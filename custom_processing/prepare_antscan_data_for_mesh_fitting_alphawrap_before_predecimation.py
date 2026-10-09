"""
Antscan mesh pipeline - alpha-wrap reconstruction core.

Replaces the Weld+holes_fill+dissolve_limit / tiled-winding-number / raw-Manifold reconstruction
cores (all four prior attempts: prepare_antscan_data_for_mesh_fitting.py,
prepare_antscan_data_for_mesh_fitting_enhanced.py, prepare_antscan_data_for_mesh_fitting_test.py,
prepare_antscan_data_for_mesh_fitting_manifold.py) with a purpose-built watertight-reconstruction
tool that carries a formal guarantee none of those four had: CGAL's 3D Alpha Wrapping
(Portaneri et al., SIGGRAPH) produces a watertight, 2-manifold, intersection-free mesh that
STRICTLY CONTAINS the input surface - so it cannot silently fuse two anatomically-distinct parts
(e.g. a leg resting near the body) the way Weld/tiled-MC/raw-Manifold all independently did on
this corpus. See WORKSPACE research notes for the full grounding (Amenta & Bern local-feature-size
theory, ManifoldPlus's documented fix for hjwdzh/Manifold's bowtie-vertex defect).

Everything upstream/downstream of reconstruction (ray-cast cleaning + island bridging,
deterministic seeding, hardened decimate_mesh, integrity/degeneracy diagnostics, the
Hausdorff-style fidelity check) was never the problem and is reused as-is from
prepare_antscan_data_for_mesh_fitting_manifold.py rather than reimplemented - see that module's
docstrings for the validation history behind each of those pieces.

STATUS (updated 2026-08-13): both external tools now build successfully on the dev machine
(custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap,
custom_processing/external/ManifoldPlus/build/manifold - see ALPHA_WRAP_SETUP.md for the two
build fixes that were needed: pointing CGAL_DIR at the git checkout directly, and forcing the
system compiler instead of the active conda env's sandboxed one). triangle_soup_wrap's CLI
contract (argv layout, relative-vs-absolute alpha/offset, always-.off output with a deterministic
name) was confirmed by reading its source directly (Alpha_wrap_3/examples/Alpha_wrap_3/
triangle_soup_wrap.cpp + output_helper.h) rather than inferred, and close_holes_via_alpha_wrap
below matches that contract exactly - see its docstring. ManifoldPlus's `--input`/`--output`/
`--depth` flags are still UNCONFIRMED (its own --help output hasn't been captured yet) - confirm
those before trusting the fallback path on a real specimen.

Neither method has been run end-to-end on a real antscan specimen through process_stl() yet - the
subprocess integration is now believed correct, not proven so. Per this project's diagnostics
rules: run compare_reconstruction_methods() on the ground-truth set in ALPHA_WRAP_SETUP.md's
validation protocol, and actually look at the exported meshes, before trusting a batch run.
"""
import bpy
import bmesh
import os
import sys
import subprocess
import tempfile
import time
import json
import numpy as np
from mathutils import Vector, Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_antscan_data_for_mesh_fitting_manifold as base

SCRIPT_VERSION = "2026-08-alpha-wrap-reconstruction-v1"

DEFAULT_ALPHA_WRAP_BINARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "external", "cgal_alpha_wrap", "build",
    "triangle_soup_wrap",
)
DEFAULT_MANIFOLDPLUS_BINARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "external", "ManifoldPlus", "build", "manifold",
)


def _derive_alpha_wrap_params(
    obj,
    edge_percentile=10,
    percentile_factor=0.4,
    gap_safety_factor=0.5,
    self_approach_gap_hops=4,
    self_approach_gap_k=100,
    offset_factor=1.0 / 30.0,
    min_alpha=None,
    max_alpha=None,
):
    """
    Derives per-specimen `alpha`/`offset` for CGAL alpha wrapping, as ABSOLUTE lengths in the
    mesh's own units. `alpha` IS this pipeline's local-feature-size target - CGAL's own docs state
    it directly ("a larger alpha will remove small features and fill large gaps, a small alpha
    will preserve those features"), so this reuses _derive_manifold_resolution's exact
    target_cell_size derivation (low-percentile edge length, further shrunk against the
    specimen's own tightest self-approach gap via base._min_self_approach_gap) and returns that
    value AS alpha, rather than converting it into an octree resolution count the way the
    Manifold-external branch had to.

    IMPORTANT: triangle_soup_wrap's actual CLI does NOT take these absolute values directly -
    confirmed by reading Alpha_wrap_3/examples/Alpha_wrap_3/triangle_soup_wrap.cpp directly (this
    repo vendors the CGAL source at build time, see ALPHA_WRAP_SETUP.md): its argv[2]/argv[3] are
    `relative_alpha`/`relative_offset`, converted internally via
    `alpha = bbox_diag_length / relative_alpha`. close_holes_via_alpha_wrap does that conversion
    (relative_alpha = bbox_diag_length / alpha) right before invoking the subprocess - this
    function stays in absolute units because that's what's directly comparable to edge lengths/
    self-approach gaps, and what the fidelity gate in process_stl compares deviations against.

    offset_factor default (1/30) matches CGAL's own documented/example default ratio between its
    relative_alpha (20) and relative_offset (600) - confirmed via the same source read above
    (`std::stod(argv[2])` defaults to 20., `std::stod(argv[3])` defaults to 600.) - i.e.
    offset = alpha * 20/600 = alpha/30. This is CGAL's stated default, not independently validated
    against this corpus yet - the research notes that motivated this file suggested trying
    offset ~ alpha * 0.3 as a starting point instead; both are untuned. Confirm empirically on the
    ground-truth set (see ALPHA_WRAP_SETUP.md's validation protocol) before trusting either on a
    full batch.

    min_alpha/max_alpha: optional clamps. Unlike Manifold's resolution (which needed a ceiling to
    avoid multi-hour runs at extreme values), alpha is a length in the mesh's own units and has no
    equivalent runaway-cost failure mode observed yet - left unclamped by default.

    Returns:
        dict: {"alpha", "offset", "p{edge_percentile}_edge", "min_self_approach_gap", "capped"}.
    """
    verts, faces = base._triangulated_verts_faces(obj)
    edge_lengths = np.linalg.norm(verts[faces[:, [1, 2, 0]]] - verts[faces], axis=-1).ravel()
    p_edge = float(np.percentile(edge_lengths, edge_percentile)) if len(edge_lengths) else 1.0
    if p_edge <= 0:
        p_edge = float(np.median(edge_lengths)) if len(edge_lengths) else 1.0
    if p_edge <= 0:
        p_edge = 1.0

    alpha = p_edge * percentile_factor

    min_gap = float("inf")
    if gap_safety_factor is not None:
        min_gap = base._min_self_approach_gap(
            verts, faces, hops=self_approach_gap_hops, k=self_approach_gap_k
        )
        if np.isfinite(min_gap):
            gap_bound_alpha = min_gap * gap_safety_factor
            if gap_bound_alpha < alpha:
                print(f"_derive_alpha_wrap_params: self-approach gap requires finer alpha than "
                      f"the edge-percentile alone gave - shrinking alpha {alpha:.4g} -> "
                      f"{gap_bound_alpha:.4g} (min_gap={min_gap:.4g} * "
                      f"safety_factor={gap_safety_factor})")
                alpha = gap_bound_alpha
        else:
            print("_derive_alpha_wrap_params: WARNING - could not determine "
                  "min_self_approach_gap; proceeding with the edge-percentile-only alpha, "
                  "unchecked against self-contact gaps.")

    capped = False
    if min_alpha is not None and alpha < min_alpha:
        alpha = min_alpha
        capped = True
    if max_alpha is not None and alpha > max_alpha:
        alpha = max_alpha
        capped = True

    offset = alpha * offset_factor

    print(f"_derive_alpha_wrap_params: p{edge_percentile}_edge={p_edge:.4g}, "
          f"min_self_approach_gap={min_gap:.4g}, alpha={alpha:.4g}, offset={offset:.4g}"
          f"{' (capped)' if capped else ''}")

    return {
        "alpha": alpha,
        "offset": offset,
        f"p{edge_percentile}_edge": p_edge,
        "min_self_approach_gap": min_gap,
        "capped": capped,
    }


def _reimport_mesh_in_place(obj, obj_path, tmp_dir, unique_tag):
    """
    Shared "replace obj's mesh data with a reconstructed OBJ on disk" idiom, factored out because
    both close_holes_via_alpha_wrap and close_holes_via_manifold_plus need it (mirrors the inline
    version in base.close_holes_via_manifold_external exactly, including the careful
    deselect-before-import to avoid touching obj itself).
    """
    bpy.ops.object.select_all(action="DESELECT")
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path)
    imported_obj = bpy.context.selected_objects[0]

    old_mesh = obj.data
    obj.data = imported_obj.data
    bpy.data.objects.remove(imported_obj)
    bpy.data.meshes.remove(old_mesh)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def close_holes_via_alpha_wrap(obj, alpha_wrap_binary_path=None, alpha=None, offset=None,
                                alpha_derivation_kwargs=None, tmp_dir=None):
    """
    Primary reconstruction step: shells out to a compiled CGAL 3D Alpha Wrapping binary
    (custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap - build via
    ALPHA_WRAP_SETUP.md) on obj's CURRENT mesh, re-imports the result in place. Unlike
    base.close_holes_via_manifold_external, no bowtie-vertex post-fix is needed: alpha wrapping's
    output is unconditionally 2-manifold and intersection-free by construction (CGAL's own
    guarantee), not just "manifold except for a documented edge case" the way raw Manifold is.

    Robust to exactly the input state clean_internal_geometry leaves behind (self-intersections,
    non-manifold junk from ray-cast cleaning, degenerate faces) - CGAL explicitly designs alpha
    wrap for "polygon soup" input, so this is intended to be called directly on the
    post-clean_internal_geometry mesh with NO Weld/EdgeSplit beforehand, same reasoning
    close_holes_via_manifold_external already established for skipping those (pre-Weld is pure
    risk of fusing close anatomical parts, no benefit to a method that rebuilds the surface from
    scratch regardless of input topology).

    CLI contract (confirmed by reading triangle_soup_wrap.cpp + output_helper.h directly, not
    inferred - see ALPHA_WRAP_SETUP.md):
      `triangle_soup_wrap <input_file> [relative_alpha=20] [relative_offset=600]`
      internally: alpha = bbox_diag_length/relative_alpha, offset = bbox_diag_length/relative_offset
      output: always written to CWD as
      `<input_basename_without_ext>_<int(relative_alpha)>_<int(relative_offset)>.off` (always
      .off, regardless of input extension - Blender has no native .off importer, so this loads
      the result via trimesh and re-exports to .obj before handing it to
      _reimport_mesh_in_place, same as every other reconstruction step in this pipeline).
      Input format is auto-detected by extension and CGAL's polygon-soup reader accepts .obj
      directly, so the existing export_mesh_to_obj input path needs no change.

    Args:
        obj (bpy.types.Object): The Blender object to rebuild in place.
        alpha_wrap_binary_path (str): Path to the compiled triangle_soup_wrap executable.
                                      Defaults to DEFAULT_ALPHA_WRAP_BINARY_PATH.
        alpha, offset (float or None): Target ABSOLUTE lengths (mesh units) - converted to the
                                       relative_alpha/relative_offset ratios the binary actually
                                       takes right before the subprocess call. None (default)
                                       derives both per-specimen via _derive_alpha_wrap_params;
                                       pass explicit floats to bypass that.
        alpha_derivation_kwargs (dict): extra kwargs forwarded to _derive_alpha_wrap_params when
                                        alpha/offset are None.
        tmp_dir (str): Directory for intermediate files. Defaults to system temp dir.

    Returns:
        dict: {"alpha", "offset", "relative_alpha", "relative_offset", "output_vertices",
              "output_faces"}, plus the full _derive_alpha_wrap_params stats under
              "alpha_derivation" when auto-derived.
    """
    if alpha_wrap_binary_path is None:
        alpha_wrap_binary_path = DEFAULT_ALPHA_WRAP_BINARY_PATH
    if not os.path.isfile(alpha_wrap_binary_path):
        raise FileNotFoundError(
            f"Alpha-wrap binary not found at {alpha_wrap_binary_path}. Build it per "
            f"custom_processing/external/ALPHA_WRAP_SETUP.md."
        )

    alpha_derivation = None
    if alpha is None or offset is None:
        print("close_holes_via_alpha_wrap: deriving per-specimen alpha/offset...")
        alpha_derivation = _derive_alpha_wrap_params(obj, **(alpha_derivation_kwargs or {}))
        alpha = alpha if alpha is not None else alpha_derivation["alpha"]
        offset = offset if offset is not None else alpha_derivation["offset"]

    tmp_dir = tmp_dir or tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    unique_tag = f"{obj.name}_{os.getpid()}"
    tmp_input = os.path.join(tmp_dir, f"_alphawrap_in_{unique_tag}.obj")

    print(f"close_holes_via_alpha_wrap: exporting current mesh ({len(obj.data.vertices)} verts, "
          f"{len(obj.data.polygons)} faces) to {tmp_input}...")
    base.export_mesh_to_obj(obj, tmp_input)

    # Convert absolute alpha/offset to the relative_alpha/relative_offset ratios the binary
    # actually consumes - same bbox-diagonal formula as the C++ source (diag_length / relative_*),
    # computed on the identical point set just exported to tmp_input.
    verts, _ = base._triangulated_verts_faces(obj)
    diag_length = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    if diag_length <= 0:
        raise RuntimeError("close_holes_via_alpha_wrap: degenerate mesh (zero bbox diagonal).")
    relative_alpha = diag_length / alpha
    relative_offset = diag_length / offset

    print(f"close_holes_via_alpha_wrap: running {alpha_wrap_binary_path} "
          f"(alpha={alpha:.4g} -> relative_alpha={relative_alpha:.4g}, "
          f"offset={offset:.4g} -> relative_offset={relative_offset:.4g})...")
    result = subprocess.run(
        [alpha_wrap_binary_path, tmp_input, str(relative_alpha), str(relative_offset)],
        capture_output=True, text=True, cwd=tmp_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Alpha-wrap reconstruction failed (exit_code={result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Exact output name per generate_output_name() in output_helper.h - int() truncates toward
    # zero same as C++'s static_cast<int>, so this must match bit-for-bit given the same inputs.
    input_stem = os.path.splitext(os.path.basename(tmp_input))[0]
    tmp_output_off = os.path.join(
        tmp_dir, f"{input_stem}_{int(relative_alpha)}_{int(relative_offset)}.off"
    )
    if not os.path.isfile(tmp_output_off):
        raise RuntimeError(
            f"Alpha-wrap reconstruction did not produce the expected output "
            f"{tmp_output_off}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print(f"close_holes_via_alpha_wrap: reconstruction complete, output={tmp_output_off}. "
          f"stdout: {result.stdout.strip()}")

    import trimesh
    off_mesh = trimesh.load(tmp_output_off, process=False)
    tmp_output_obj = os.path.join(tmp_dir, f"_alphawrap_out_{unique_tag}.obj")
    off_mesh.export(tmp_output_obj)

    _reimport_mesh_in_place(obj, tmp_output_obj, tmp_dir, unique_tag)

    os.remove(tmp_input)
    os.remove(tmp_output_off)
    os.remove(tmp_output_obj)

    print("close_holes_via_alpha_wrap: mesh state after reconstruction:")
    base.log_mesh_state(obj, "after_alpha_wrap_reconstruction")

    result_dict = {
        "alpha": alpha,
        "offset": offset,
        "relative_alpha": relative_alpha,
        "relative_offset": relative_offset,
        "output_vertices": len(obj.data.vertices),
        "output_faces": len(obj.data.polygons),
    }
    if alpha_derivation is not None:
        result_dict["alpha_derivation"] = alpha_derivation
    return result_dict


def close_holes_via_manifold_plus(obj, manifoldplus_binary_path=None, depth=8, tmp_dir=None):
    """
    Fallback reconstruction step, for use when a CGAL alpha-wrap build is impractical in a given
    environment (see module docstring / ALPHA_WRAP_SETUP.md). hjwdzh/ManifoldPlus is the same
    author's 2020 successor to the hjwdzh/Manifold tool base.close_holes_via_manifold_external
    already wraps, and its own paper documents fixing exactly the defect that forced this
    pipeline to hand-write split_nonmanifold_vertex/bowtie repair: "Manifold [Huang et al. 2018a]
    yields non-manifold vertices" (ManifoldPlus is free of that). No bowtie post-fix is applied
    here for that reason - if a future specimen turns out to need one after all, treat that as a
    signal ManifoldPlus's fix doesn't fully generalize to this corpus, not as a reason to silently
    re-add the old patcher.

    Does NOT give alpha wrap's strict-enclosure guarantee - a leg-fusion failure mode is still
    possible here in principle, which is exactly why the mandatory fidelity gate in process_stl
    runs unconditionally regardless of which reconstruction method actually ran.

    Args:
        obj (bpy.types.Object): The Blender object to rebuild in place.
        manifoldplus_binary_path (str): Path to the compiled ManifoldPlus executable. Defaults to
                                        DEFAULT_MANIFOLDPLUS_BINARY_PATH.
        depth (int): ManifoldPlus's octree depth parameter (its `--depth` flag). 8 is the value
                     shown in the tool's own usage examples - not tuned against this corpus.
        tmp_dir (str): Directory for intermediate .obj files. Defaults to system temp dir.

    Returns:
        dict: {"depth", "output_vertices", "output_faces"}.
    """
    if manifoldplus_binary_path is None:
        manifoldplus_binary_path = DEFAULT_MANIFOLDPLUS_BINARY_PATH
    if not os.path.isfile(manifoldplus_binary_path):
        raise FileNotFoundError(
            f"ManifoldPlus binary not found at {manifoldplus_binary_path}. Build it per "
            f"custom_processing/external/ALPHA_WRAP_SETUP.md."
        )

    tmp_dir = tmp_dir or tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    unique_tag = f"{obj.name}_{os.getpid()}"
    tmp_input = os.path.join(tmp_dir, f"_manifoldplus_in_{unique_tag}.obj")
    tmp_output = os.path.join(tmp_dir, f"_manifoldplus_out_{unique_tag}.obj")

    print(f"close_holes_via_manifold_plus: exporting current mesh ({len(obj.data.vertices)} "
          f"verts, {len(obj.data.polygons)} faces) to {tmp_input}...")
    base.export_mesh_to_obj(obj, tmp_input)

    print(f"close_holes_via_manifold_plus: running {manifoldplus_binary_path} (depth={depth})...")
    result = subprocess.run(
        [manifoldplus_binary_path, "--input", tmp_input, "--output", tmp_output,
         "--depth", str(depth)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.isfile(tmp_output):
        raise RuntimeError(
            f"ManifoldPlus reconstruction failed (exit_code={result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print(f"close_holes_via_manifold_plus: reconstruction complete. stdout: {result.stdout.strip()}")

    _reimport_mesh_in_place(obj, tmp_output, tmp_dir, unique_tag)

    os.remove(tmp_input)
    os.remove(tmp_output)

    print("close_holes_via_manifold_plus: mesh state after reconstruction:")
    base.log_mesh_state(obj, "after_manifold_plus_reconstruction")

    return {
        "depth": depth,
        "output_vertices": len(obj.data.vertices),
        "output_faces": len(obj.data.polygons),
    }


def process_stl(
    stl_path,
    output_dir=None,
    max_vertices=20000,
    ray_density=1000,
    secondary_rays=5,
    random_seed=42,
    min_island_faces=4,
    keep_rings=2,
    min_bridge_island_faces=50,
    max_bridge_hops=20,
    reconstruction_method="alpha_wrap",
    alpha_wrap_binary_path=None,
    alpha=None,
    offset=None,
    alpha_derivation_kwargs=None,
    manifoldplus_binary_path=None,
    manifoldplus_depth=8,
    allow_fallback=True,
    fidelity_gate_factor=3.0,
    cap_remaining_holes_flag=True,
):
    """
    Processes an STL file by importing, cleaning, reconstructing a watertight surface, and
    decimating the mesh - the alpha-wrap architecture from the research notes that motivated this
    file (section 3): ray-cast clean -> reconstruct (alpha wrap, formally guaranteed enclosure) ->
    decimate -> orient -> MANDATORY fidelity gate (every specimen, every method, no exceptions) ->
    export.

    reconstruction_method: "alpha_wrap" (default, primary per the research notes) or
    "manifold_plus" (fallback method, same CLI-wrapping pattern as
    base.close_holes_via_manifold_external but built on ManifoldPlus instead of raw Manifold).
    If reconstruction_method="alpha_wrap" and allow_fallback=True (default) and the alpha-wrap
    binary is missing or the subprocess call fails, automatically retries with manifold_plus
    before giving up - mirrors the target environment ("CGAL build impractical") the research
    notes anticipated. Set allow_fallback=False to fail loudly instead (e.g. for a validation run
    that specifically wants to know whether alpha_wrap itself works).

    No Weld/EdgeSplit is run before reconstruction under either method - both are designed to
    rebuild a fresh watertight surface directly from clean_internal_geometry's ray-cast-cleaned
    "polygon soup" output, so pre-Weld is pure fusion risk with no benefit (established by
    base.close_holes_via_manifold_external's own Weld-risk finding,
    diagnostics/sericomyrmex_weld_risk_check/).

    The fidelity gate (verify_reconstruction_fidelity against the pre-reconstruction snapshot) is
    NOT conditional on which method ran or whether a fallback fired - per the research notes'
    explicit instruction, this is the one check that must never be skipped, because aggregate
    watertightness/manifold-ness cannot see a silently-fused self-approach gap.
    """
    if reconstruction_method not in ("alpha_wrap", "manifold_plus"):
        raise ValueError(
            f"reconstruction_method must be 'alpha_wrap' or 'manifold_plus', "
            f"got {reconstruction_method!r}"
        )

    process_stl_start_time = time.time()
    base._clear_scene_mesh_objects()

    bpy.ops.wm.stl_import(filepath=stl_path)
    obj = bpy.context.selected_objects[0]

    initial_vertices = len(obj.data.vertices)
    if initial_vertices > 2000000:
        print(f"Initial vertex count: {initial_vertices}. Reducing vertices...")
        reduced_vertices = base.reduce_vertices_by_distance(obj)
        print(f"Reduced vertex count: {reduced_vertices}")

    base.find_largest_component(obj)

    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")
    obj.location = Vector((0, 0, 0))

    print("Cleaning internal geometry...")
    base.clean_internal_geometry(
        obj,
        ray_density,
        secondary_rays,
        random_seed,
        keep_rings=keep_rings,
        min_island_faces=min_island_faces,
        min_bridge_island_faces=min_bridge_island_faces,
        max_bridge_hops=max_bridge_hops,
    )

    # Snapshot the pre-reconstruction mesh to disk (not bpy.data.meshes.copy() - confirmed by
    # base.process_stl's manifold_external branch to measurably degrade Blender's per-iteration
    # performance across a full branch's worth of decimate_mesh iterations if kept as a live
    # in-memory datablock instead) and record the specimen's tightest self-approach gap. Both feed
    # the mandatory fidelity gate below, unconditionally, regardless of which reconstruction
    # method actually ends up running.
    pre_recon_backup_path = os.path.join(
        tempfile.gettempdir(), f"_alphawrap_prerecon_{obj.name}_{os.getpid()}.obj"
    )
    base.export_mesh_to_obj(obj, pre_recon_backup_path)
    verts, faces = base._triangulated_verts_faces(obj)
    pre_gap, target_a, target_b = base._min_self_approach_gap(verts, faces, return_pair=True)

    actual_reconstruction_method = reconstruction_method
    target_cell_size = None

    print(f"Rebuilding watertight surface via {reconstruction_method}...")
    try:
        if reconstruction_method == "alpha_wrap":
            recon_stats = close_holes_via_alpha_wrap(
                obj, alpha_wrap_binary_path=alpha_wrap_binary_path, alpha=alpha, offset=offset,
                alpha_derivation_kwargs=alpha_derivation_kwargs,
            )
            target_cell_size = (
                recon_stats.get("alpha_derivation", {}).get("alpha") or recon_stats.get("alpha")
            )
        else:
            recon_stats = close_holes_via_manifold_plus(
                obj, manifoldplus_binary_path=manifoldplus_binary_path, depth=manifoldplus_depth,
            )
    except (FileNotFoundError, RuntimeError) as primary_error:
        if not allow_fallback or reconstruction_method != "alpha_wrap":
            raise
        print(f"WARNING: {reconstruction_method} reconstruction failed ({primary_error}); "
              f"falling back to manifold_plus.")
        bpy.ops.object.select_all(action="DESELECT")
        try:
            bpy.ops.wm.obj_import(filepath=pre_recon_backup_path)
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=pre_recon_backup_path)
        restored_obj = bpy.context.selected_objects[0]
        old_mesh = obj.data
        obj.data = restored_obj.data
        bpy.data.objects.remove(restored_obj)
        bpy.data.meshes.remove(old_mesh)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        recon_stats = close_holes_via_manifold_plus(
            obj, manifoldplus_binary_path=manifoldplus_binary_path, depth=manifoldplus_depth,
        )
        actual_reconstruction_method = "manifold_plus_fallback_from_alpha_wrap"

    print(f"Reconstruction stats: {recon_stats}")

    removed_islands = base.filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) after reconstruction")

    remaining_vertices = base.decimate_mesh(obj, max_vertices, max_iterations=25)
    print(f"DIAGNOSTIC: holes after decimate: {base.count_holes(obj)}, "
          f"boundary_edges: {base.count_boundary_edges(obj)}")

    n_orphans_removed = base.remove_orphan_vertices(obj)
    if n_orphans_removed:
        remaining_vertices = len(obj.data.vertices)
        print(f"After remove_orphan_vertices: {remaining_vertices} vertices")

    # --- PCA reorientation / leg-down / head-direction (unchanged from base.process_stl) ---
    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    cov_matrix = np.cov(vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    rotation_matrix = Matrix(eigenvectors).to_4x4().inverted()
    obj.matrix_world = rotation_matrix @ obj.matrix_world
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    print("Aligned model with X-axis based on principal component analysis.")

    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    y_variance = np.var(vertices[:, 1])
    z_variance = np.var(vertices[:, 2])
    if y_variance < z_variance:
        rotation_matrix = Matrix.Rotation(np.pi / 2, 4, "X")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 90 degrees around X-axis to put legs down.")

    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_center = (z_min + z_max) / 2
    z_median = np.median(vertices[:, 2])
    if z_median < z_center:
        rotation_matrix = Matrix.Rotation(np.pi, 4, "X")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Flipped model 180 degrees around X-axis to ensure positive Z is up.")

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    print("Ensured positive Z is up.")

    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    num_slices = 20
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    slice_width = (x_max - x_min) / num_slices
    slice_densities = []
    for i in range(num_slices):
        slice_start = x_min + i * slice_width
        slice_end = slice_start + slice_width
        slice_vertices = vertices[(vertices[:, 0] >= slice_start) & (vertices[:, 0] < slice_end)]
        slice_volume = (
            slice_width
            * (slice_vertices[:, 1].max() - slice_vertices[:, 1].min())
            * (slice_vertices[:, 2].max() - slice_vertices[:, 2].min())
        )
        slice_density = len(slice_vertices) / slice_volume if slice_volume > 0 else 0
        slice_densities.append(slice_density)

    head_end = "start" if np.mean(slice_densities[:3]) < np.mean(slice_densities[-3:]) else "end"
    if head_end == "end":
        rotation_matrix = Matrix.Rotation(np.pi, 4, "Z")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 180 degrees around Z-axis to ensure head is in positive X direction.")

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    # --- end reorientation ---

    remaining_vertices = len(obj.data.vertices)
    print(f"Number of remaining vertices: {remaining_vertices}")

    hole_count = base.count_holes(obj)
    print(f"Number of holes in the processed mesh (before capping): {hole_count}")
    if hole_count > 0:
        # Alpha wrap/ManifoldPlus are both expected to already be watertight by construction - any
        # holes here indicate the reconstruction guarantee didn't hold for this specimen (or the
        # fallback path lost it) and this specimen needs manual review regardless of what the
        # fidelity gate below says. Keep the same capping safety net as the other pipeline
        # versions rather than assuming it will never fire.
        print(f"WARNING: {hole_count} holes present after a reconstruction method that should be "
              f"watertight by construction - flag {os.path.basename(stl_path)} for manual review.")
        base.describe_hole_locations(obj, top_n=15)
        if cap_remaining_holes_flag:
            capped, cap_skipped = base.cap_remaining_holes(obj)
            remaining_vertices = len(obj.data.vertices)
            hole_count = base.count_holes(obj)
            print(f"After cap_remaining_holes: {remaining_vertices} vertices, "
                  f"{hole_count} holes remaining (capped={capped}, skipped={cap_skipped})")

    # --- MANDATORY fidelity gate: every specimen, every method, no exceptions (research notes
    # section 3/8) - a fused mesh is still perfectly watertight/manifold, so hole_count and
    # mesh_integrity_report cannot see this failure mode; only a whole-surface distance check can.
    fidelity_gate_passed = True
    worst_fidelity_deviation = None
    fidelity = base.verify_reconstruction_fidelity(pre_recon_backup_path, obj)
    worst_fidelity_deviation = max(fidelity["pre_to_post_p99.9"], fidelity["post_to_pre_p99.9"])
    if target_cell_size is not None:
        if worst_fidelity_deviation > target_cell_size * fidelity_gate_factor:
            fidelity_gate_passed = False
            print(f"FIDELITY GATE FAILED: worst two-sided p99.9 surface deviation "
                  f"{worst_fidelity_deviation:.4g} exceeds {fidelity_gate_factor}x target_cell_size "
                  f"({target_cell_size:.4g}) - flag {os.path.basename(stl_path)} for MANUAL REVIEW "
                  f"before trusting this output (possible self-approach-gap fusion or other "
                  f"reconstruction deviation).")
        else:
            print(f"Fidelity gate passed: worst deviation {worst_fidelity_deviation:.4g} within "
                  f"{fidelity_gate_factor}x target_cell_size ({target_cell_size:.4g}).")
    else:
        print("Fidelity gate: no target_cell_size available (manifold_plus has no equivalent "
              "derived scale) - deviation logged for manual review, not used to gate pass/fail.")

    if target_a is not None:
        gap_check = base.verify_gap_preservation(obj, target_a, target_b, pre_gap,
                                                   correspondence_gate_factor=20)
        print(f"verify_gap_preservation (fast landmark pre-check): verdict={gap_check['verdict']!r}")
        if gap_check["verdict"] == "fused":
            fidelity_gate_passed = False
    os.remove(pre_recon_backup_path)
    # --- end mandatory fidelity gate ---

    face_size_cov = base.calculate_face_size_cov(obj)
    print(f"Coefficient of variation of face sizes: {face_size_cov}")

    mesh_smoothness = base.calculate_mesh_smoothness(obj)
    print(f"Average angle between face normals: {mesh_smoothness} degrees")

    integrity = base.mesh_integrity_report(obj)
    degeneracy = base.report_mesh_degeneracy(obj)
    non_manifold_vert_pct = (
        100.0 * integrity["non_manifold_verts"] / integrity["vertex_count"]
        if integrity["vertex_count"] else 0.0
    )

    process_stl_elapsed = time.time() - process_stl_start_time
    specimen_name = os.path.splitext(os.path.basename(stl_path))[0]
    print(
        f"FINAL_SUMMARY: specimen={specimen_name} approach={actual_reconstruction_method} "
        f"vertices={integrity['vertex_count']} faces={integrity['face_count']} "
        f"simple_holes={integrity['simple_holes']} boundary_edges={integrity['boundary_edges']} "
        f"non_manifold_edges={integrity['non_manifold_edges']} "
        f"non_manifold_verts={integrity['non_manifold_verts']} "
        f"non_manifold_vert_pct={non_manifold_vert_pct:.3f} "
        f"is_watertight={integrity['is_watertight']} signed_volume={integrity['signed_volume']} "
        f"face_size_cov={face_size_cov} mesh_smoothness={mesh_smoothness} "
        f"fidelity_gate_passed={fidelity_gate_passed} "
        f"worst_fidelity_deviation={worst_fidelity_deviation:.4g} "
        f"time_sec={process_stl_elapsed:.1f}"
    )

    if output_dir is None:
        output_dir = os.path.dirname(stl_path)
    os.makedirs(output_dir, exist_ok=True)

    original_file_name = os.path.splitext(os.path.basename(stl_path))[0]
    export_path = os.path.join(output_dir, f"{original_file_name}_processed.obj")

    print(f"Exporting the processed mesh to {export_path}...")
    base.export_mesh_to_obj(obj, export_path)
    print("Mesh exported successfully.")

    json_path = os.path.splitext(stl_path)[0] + ".json"
    if os.path.exists(json_path):
        print(f"Updating JSON file: {json_path}")
        with open(json_path, "r") as f:
            json_data = json.load(f)
        json_data["processed_vertex_count"] = remaining_vertices
        json_data["processed_hole_count"] = hole_count
        json_data["processed_face_size_cov"] = face_size_cov
        json_data["processed_mesh_smoothness"] = mesh_smoothness
        json_data["processed_mesh_integrity"] = integrity
        json_data["processed_mesh_degeneracy"] = degeneracy
        json_data["processed_reconstruction_method"] = actual_reconstruction_method
        json_data["processed_fidelity_gate_passed"] = fidelity_gate_passed
        json_data["processed_worst_fidelity_deviation"] = worst_fidelity_deviation
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print("JSON file updated successfully.")
    else:
        print(f"Warning: Corresponding JSON file not found at {json_path}")

    if not fidelity_gate_passed:
        print(f"*** {specimen_name}: FIDELITY GATE FAILED - exported anyway per policy (flag, "
              f"don't silently drop), but do NOT trust this output without manual visual review. ***")

    return (remaining_vertices, hole_count, face_size_cov, mesh_smoothness, fidelity_gate_passed)


def compare_reconstruction_methods(stl_path, output_dir, **kwargs):
    """
    Runs process_stl twice on the same specimen - once with reconstruction_method="alpha_wrap"
    (allow_fallback=False, so a build/runtime failure is loud rather than silently swapping in
    manifold_plus) and once with "manifold_plus" - into method-specific subdirectories. This is
    the ground-truth A/B comparison the research notes' validation protocol (step 5) calls for
    before trusting either method on a full batch: run on a handful of known specimens (compact,
    long-thin-legged, self-approach-heavy) and compare both the fidelity-gate numbers AND the
    exported OBJs opened directly in Blender/MeshLab - do not trust the log alone.
    """
    alpha_wrap_dir = os.path.join(output_dir, "alpha_wrap")
    manifold_plus_dir = os.path.join(output_dir, "manifold_plus")
    os.makedirs(alpha_wrap_dir, exist_ok=True)
    os.makedirs(manifold_plus_dir, exist_ok=True)

    print(f"--- Running reconstruction_method='alpha_wrap' on {stl_path} ---")
    alpha_wrap_result = process_stl(
        stl_path, output_dir=alpha_wrap_dir, reconstruction_method="alpha_wrap",
        allow_fallback=False, **kwargs,
    )

    print(f"--- Running reconstruction_method='manifold_plus' on {stl_path} ---")
    manifold_plus_result = process_stl(
        stl_path, output_dir=manifold_plus_dir, reconstruction_method="manifold_plus", **kwargs,
    )

    return {"alpha_wrap": alpha_wrap_result, "manifold_plus": manifold_plus_result}


def main():
    print(f"SCRIPT_VERSION={SCRIPT_VERSION}")
    start_time = time.time()

    mode = "compare"

    if bpy.context.space_data is not None and bpy.context.space_data.type == "TEXT_EDITOR":
        stl_path = bpy.path.abspath(
            "/home/fabi/dev/SMILify/custom_processing/antscan_data/Vollenhovia_sp._CASENT0745625/Vollenhovia_sp._CASENT0745625.stl"
        )
        output_dir = "/home/fabi/dev/SMILify/custom_processing/antscan_processed_alpha_wrap"
    else:
        # Optional 3rd trailing arg "alpha_wrap_only"/"manifold_plus_only"/"compare" selects the
        # mode explicitly - matches the wn_only/manifold_only convention in
        # prepare_antscan_data_for_mesh_fitting_manifold.py's main(). Bug fixed 2026-08-13: this
        # used to only recognize "alpha_wrap_only"/"manifold_plus_only" here, so passing "compare"
        # explicitly (rather than omitting the 3rd arg to get the default) fell through to the
        # 2-arg branch below and got misread as output_dir, silently shifting stl_path/output_dir
        # by one position - confirmed by reproducing it: stl_path ended up set to what was meant
        # as output_dir, and Blender's STL importer failed with "Is a directory".
        if len(sys.argv) >= 2 and sys.argv[-1] in ("alpha_wrap_only", "manifold_plus_only", "compare"):
            mode = sys.argv[-1]
            if len(sys.argv) < 4:
                print(
                    "Usage: blender --background --python "
                    "prepare_antscan_data_for_mesh_fitting_alphawrap.py -- <input_stl_path> "
                    "<output_dir> [alpha_wrap_only|manifold_plus_only|compare]"
                )
                sys.exit(1)
            stl_path = sys.argv[-3]
            output_dir = sys.argv[-2]
        else:
            if len(sys.argv) < 3:
                print(
                    "Usage: blender --background --python "
                    "prepare_antscan_data_for_mesh_fitting_alphawrap.py -- <input_stl_path> "
                    "<output_dir> [alpha_wrap_only|manifold_plus_only|compare]"
                )
                sys.exit(1)
            stl_path = sys.argv[-2]
            output_dir = sys.argv[-1]

    print(f"mode={mode}, stl_path={stl_path}, output_dir={output_dir}")

    if mode == "alpha_wrap_only":
        method_out_dir = os.path.join(output_dir, "alpha_wrap")
        os.makedirs(method_out_dir, exist_ok=True)
        process_stl(stl_path, output_dir=method_out_dir, max_vertices=50000, ray_density=1000,
                    secondary_rays=10000, random_seed=0, keep_rings=2,
                    reconstruction_method="alpha_wrap")
    elif mode == "manifold_plus_only":
        method_out_dir = os.path.join(output_dir, "manifold_plus")
        os.makedirs(method_out_dir, exist_ok=True)
        process_stl(stl_path, output_dir=method_out_dir, max_vertices=50000, ray_density=1000,
                    secondary_rays=10000, random_seed=0, keep_rings=2,
                    reconstruction_method="manifold_plus")
    else:
        compare_reconstruction_methods(
            stl_path, output_dir, max_vertices=50000, ray_density=1000, secondary_rays=10000,
            random_seed=0, keep_rings=2,
        )

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
