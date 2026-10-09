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

UPDATE (2026-08-13): close_holes_via_alpha_wrap's subprocess integration is now CONFIRMED correct
end-to-end on real data - on Metapone_emersoni_CASENT0745558, at a manually-floored alpha (see
_derive_alpha_wrap_params's min_alpha default and its docstring for the full story: the unclamped
derivation drove relative_alpha to ~1738 and got the CGAL subprocess OOM-killed), CGAL completed
in 0.63s and produced a mesh that was watertight (0 boundary/non-manifold edges) at every stage
checked - raw .off, .off->.obj conversion, and the reimported Blender mesh. This was run via a
bypass of process_stl() (now removed), not process_stl() itself, since process_stl()'s
pre-reconstruction base.decimate_mesh() step separately raises on this same specimen's non-
manifold ray-cast-cleaned soup (base.decimate_mesh() requires boundary_edges==0/non_manifold_
edges==0 before it will do anything, an unsatisfiable precondition for reconstruction input by
this architecture's own design) - that is a SEPARATE, still-open problem, not fixed by the alpha
floor. ManifoldPlus's `--input`/`--output`/`--depth` flags were separately confirmed by reading
its source (src/main.cc + src/Parser.cc) directly.

Per this project's diagnostics rules: run compare_reconstruction_methods() on the ground-truth set
in ALPHA_WRAP_SETUP.md's validation protocol, and actually look at the exported meshes, before
trusting a batch run - and resolve the pre-reconstruction decimation problem above before
process_stl() itself can be exercised on this specimen at all.
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
    offset_factor=0.4,
    min_alpha=47.0,
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

    offset_factor default CHANGED (2026-08-16, root-caused a 44%-of-corpus bisection
    timeout/OOM-fallback-to-ManifoldPlus rate on bench50_clean): previously 1/30, matching CGAL's
    own documented/example default ratio between its relative_alpha (20) and relative_offset
    (600) - confirmed via triangle_soup_wrap.cpp source read (`std::stod(argv[2])` defaults to
    20., `std::stod(argv[3])` defaults to 600.), i.e. offset = alpha * 20/600 = alpha/30.
    CGAL's docs are explicit that alpha and offset are independent controls - alpha gates whether
    the algorithm can carve through a gap (what this pipeline's bisection tunes per specimen for
    gap preservation), offset only controls how tightly the output hugs the input everywhere else
    (surface density/refinement cost, unrelated to gap preservation). alpha/30 forces an
    unnecessarily tight offset on EVERY specimen regardless of its own gap requirement, inflating
    refinement cost mesh-wide (worse on complex/many-limbed specimens) - almost certainly the
    actual driver of bisection timeouts/OOMs on this corpus, not alpha itself. CGAL's own
    wrapwrap reference tool recommends starting ratios of relative_alpha=500/relative_offset=1200
    (offset ~= alpha * 500/1200 ~= alpha * 0.417), over 10x looser than alpha/30 - offset_factor=0.4
    matches that recommendation. CONFIRMED (2026-08-16) on both ground-truth specimens named in
    this pipeline's own validation history: Cephalotes_spinosus_CASENT0744158 (previously
    bisection-timed-out on bench50_clean and fell back to ManifoldPlus) now completes cleanly as
    pure alpha_wrap in 226s, fidelity_gate_passed=True; Dorylus_sp._CASENT0744703 (this
    function's original characterization specimen) completes in 2293s with fidelity_gate_passed=
    True and gap verdict 'preserved' consistently across all 8 bisection steps at offset=0.4*alpha
    throughout - confirming the looser offset does not change gap-preservation outcomes, only cost.
    See diagnostics/offset_fix_sanity_20260816/ for the full logs.

    min_alpha default (47.0, mesh units - NOT relative_alpha) exists specifically to prevent the
    edge-percentile/gap-safety derivation from picking a pathologically fine alpha: on
    Metapone_emersoni_CASENT0745558 (220,083 verts / 431,137 faces, bbox diagonal ~941), the
    unclamped derivation produced alpha=0.5416 -> relative_alpha~1738 (~87x finer than CGAL's own
    tested example default of 20), and the CGAL subprocess was killed by the OS (exit_code=-9,
    consistent with the OOM killer - `ps` showed 6.2GB+ RAM and climbing before it disappeared)
    before producing any output at all. min_alpha=47 was confirmed (2026-08-13, on this same
    specimen) to fix that: relative_alpha=20.02, CGAL completed in 0.63s, output 528 verts/1052
    faces, watertight with 0 boundary/non-manifold edges at every stage (raw .off, .off->.obj
    conversion, and the reimported Blender mesh).

    IMPORTANT LIMITATION: 47.0 is an ABSOLUTE length validated on exactly ONE specimen. It is NOT
    yet a validated general default across the corpus - a different specimen's bbox diagonal will
    pair the same absolute floor with a different relative_alpha (a much smaller specimen could
    still end up with an impractically fine relative_alpha even with this floor in place; a much
    larger one could end up coarser than intended). This will very likely need to become a
    diagonal-relative floor (e.g. `diag_length / max_relative_alpha`) once tested on more of the
    ground-truth set - flagged here rather than silently generalized, per this pipeline's own
    diagnostics conventions. Override via alpha_derivation_kwargs={"min_alpha": ...} (or pass
    None to disable the floor entirely) until then.

    max_alpha: optional ceiling, still unclamped by default - no runaway-cost failure mode from
    an alpha that's too COARSE has been observed (unlike min_alpha's now-confirmed failure mode
    at the fine end).

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


def _estimate_relative_alpha(pre_gap, diag_length, preserved_multiplier=5.0):
    """Cheap, no-CGAL-calls estimate of the alpha resolution a specimen's tightest self-approach
    gap will likely need, computed purely from pre_gap/diag_length (both already available before
    any reconstruction runs, from base._min_self_approach_gap and the mesh's own bounding box).

    NOT a theoretically derived formula. preserved_multiplier=5.0 is literally
    _bisect_alpha_wrap_params's own alpha_preserved_start default (pre_gap*5) - today's best
    empirical guess for "roughly where this gap starts being preserved," reused here purely as a
    cheap upfront router. Added 2026-08-17 specifically to be validated/recalibrated against the
    full-corpus diagnostic table this enables (see log_corpus_diagnostics.py / FINAL_SUMMARY's
    gap_ratio/relative_alpha_est/tier fields) - treat the returned relative_alpha_est as a rough
    routing signal, not a promise, until that calibration happens.

    Returns dict: {gap_ratio, relative_alpha_est}."""
    gap_ratio = pre_gap / diag_length if diag_length > 0 else float("nan")
    alpha_est = pre_gap * preserved_multiplier
    relative_alpha_est = diag_length / alpha_est if alpha_est > 0 else float("inf")
    return {"gap_ratio": gap_ratio, "relative_alpha_est": relative_alpha_est}


# Tier thresholds - EMPIRICAL, from the bench50_clean corpus (2026-08-17): specimens that
# converged easily under the offset+bounds-widening fixes had final relative_alpha in the 60-210
# range; specimens that still timed out at 900s needed >=2500-4300 (confirmed genuine tight
# anatomical near-touches, not scan artifacts - see _classify_near_touch). These are placeholder
# cutoffs pending the calibration run _estimate_relative_alpha's docstring describes - do not
# treat as theoretically justified.
TIER_A_MAX_RELATIVE_ALPHA = 500.0    # CGAL's own wrapwrap-tool recommended default
TIER_B_MAX_RELATIVE_ALPHA = 2500.0   # empirical cliff observed 2026-08-17, see docstring above


def _classify_tier(relative_alpha_est):
    if relative_alpha_est <= TIER_A_MAX_RELATIVE_ALPHA:
        return "A"
    elif relative_alpha_est <= TIER_B_MAX_RELATIVE_ALPHA:
        return "B"
    else:
        return "C"


# Per-tier bisection kwargs - REPLACES the 2026-08-17 ALPHA_WRAP_BISECT_TIMEOUT env-var workaround
# with automatic, per-specimen routing keyed off _classify_tier(relative_alpha_est). Tier C's
# max_total_seconds is the hard per-specimen compute ceiling requested 2026-08-17: manually
# escalating 8 specimens from 300s to 900s that day showed the alpha a genuine tight near-touch
# needs can keep retreating with no guaranteed convergence point, so Tier C needs a bounded TOTAL
# budget across the WHOLE bisection (all CGAL calls combined), not just a bigger per-call timeout,
# or a single pathological specimen can consume unbounded compute. mem_cap_kb is raised for
# Tier B/C since finer alpha (higher relative_alpha) demonstrably needs more CGAL working memory
# (see the Labidus_praedator_CASENT0744237 std::bad_alloc at relative_alpha=2787 under the
# original flat 10GB cap, 2026-08-17).
_TIER_BISECTION_KWARGS = {
    "A": {"timeout": 300, "max_total_seconds": 900, "mem_cap_kb": 10_485_760},   # ~10GB
    "B": {"timeout": 900, "max_total_seconds": 1800, "mem_cap_kb": 16_777_216},  # ~16GB
    "C": {"timeout": 900, "max_total_seconds": 2700, "mem_cap_kb": 25_165_824},  # ~24GB
}


def _classify_near_touch(verts, faces, target_a, target_b):
    """Distinguish a genuine near-touch between two distinct solid surfaces (face normals at the
    two landmark vertices point roughly OPPOSITE each other, dot product near -1) from a
    duplicate/overlapping-surface artifact (normals point roughly the SAME direction, dot product
    near +1). Validated 2026-08-17 on the 4 specimens that still timed out at 900s after the
    offset+bounds fixes (Cyphoidris_afrc-tz01, Lasius_nr._fuliginosus, Nesomyrmex_angulatus,
    Strumigenys_stenorhina): all 4 showed normal_dot in [-0.99, -0.94], confirming genuine
    anatomical near-touches, not scan noise - the same reasoning this file's own docstring
    references for worker_ALT's Dorylus_sp._CASENT0744698 ("ruled out ... by inspecting local
    vertex density"), made concrete and automatic here. Cheap (pure numpy, no CGAL calls) - safe
    to run unconditionally for every Tier C specimen."""
    d_a = np.linalg.norm(verts - target_a, axis=1)
    d_b = np.linalg.norm(verts - target_b, axis=1)
    ia = int(np.argmin(d_a))
    ib = int(np.argmin(d_b))

    def vertex_normal(vi):
        face_mask = np.any(faces == vi, axis=1)
        tris = verts[faces[face_mask]]
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        n = n / norms
        avg = n.mean(axis=0)
        avg_norm = np.linalg.norm(avg)
        return (avg / avg_norm if avg_norm > 1e-9 else avg), int(face_mask.sum())

    na, nfa = vertex_normal(ia)
    nb, nfb = vertex_normal(ib)
    dot = float(np.dot(na, nb))
    if dot > 0.3:
        interpretation = "artifact_like_parallel_normals"
    elif dot < -0.3:
        interpretation = "genuine_near_touch_opposing_normals"
    else:
        interpretation = "ambiguous"
    return {"normal_dot": dot, "n_faces_at_a": nfa, "n_faces_at_b": nfb,
            "interpretation": interpretation}


# Floor for the thin-feature (SDF p5) production gate - matches compare_pipelines.py's
# SDF_THICKNESS_RATIO_FLAG constant. Applied "at least for the ManifoldPlus fallback" per explicit
# instruction 2026-08-17: the offline bench50 analysis found the fallback path's aggregate p5
# ratio catastrophic (median 0.0015-0.0047) while fidelity_gate_passed still frequently read True,
# since the existing gate has no thin-feature awareness at all. Logged (not gating) for alpha_wrap
# too, since individual alpha_wrap specimens can still show collapse even though the subset median
# is healthy post offset-fix.
THIN_FEATURE_P5_RATIO_FLOOR = 0.5


def _shape_diameter_thickness_p5(verts, faces, n_samples=1500):
    """Lightweight single-ray shape-diameter-function thin-feature thickness estimate, matching
    compare_pipelines.py's shape_diameter_thickness() (validated 2026-08-17: a cone-averaged
    cross-check confirmed the single-ray p5 percentile is NOT noise-dominated across the full
    worst-to-best range of a 9-specimen stratified sample, so this cheaper single-ray version is
    trustworthy enough for a production gate - p1 remains unreliable and is deliberately not used
    here, same reasoning as compare_pipelines.py). Runs on the largest connected component only,
    to avoid fragment/debris islands dominating the low percentile (same reasoning as
    compare_pipelines.py's largest_component()). Returns the p5 thickness value, or nan if no
    valid ray hits (e.g. degenerate/tiny mesh)."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda p: len(p.faces))
    except Exception:
        pass
    if len(mesh.faces) == 0:
        return float("nan")
    points, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_idx]
    origins = points - normals * 1e-4
    directions = -normals
    locations, index_ray, _ = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    if len(locations) == 0:
        return float("nan")
    dists = np.linalg.norm(locations - origins[index_ray], axis=1)
    return float(np.percentile(dists, 5))


def _bisect_alpha_wrap_params(
    obj_export_path,
    target_a,
    target_b,
    pre_gap,
    diag_length,
    alpha_wrap_binary_path,
    alpha_fused_start=None,
    alpha_preserved_start=None,
    max_iters=6,
    max_bound_widen_retries=4,
    timeout=300,
    mem_cap_kb=10_485_760,
    offset_alpha_ratio=2.5,
    work_dir=None,
    max_total_seconds=None,
):
    """
    Per-specimen bisection for the coarsest (cheapest) alpha that still preserves this specimen's
    own tightest self-approach gap (target_a/target_b/pre_gap - the SAME landmark process_stl
    computes once, before reconstruction, and reuses for its own mandatory fidelity/gap gate, so
    the bisection's stopping criterion and the gate that checks its output afterward are never out
    of sync).

    ADOPTED AS THE PRODUCTION alpha-selection step (2026-08-13), replacing
    _derive_alpha_wrap_params's single-formula guess. Ported from bisect_alpha_wrap.py, which ran
    3-for-3 clean (Metapone_emersoni_CASENT0745558 local STL + 2 points on
    Dorylus_sp._CASENT0744703 local STL) before this port. Motivation: a fixed edge-percentile/
    min_alpha-floor formula cannot generalize - the worker_ALT investigation (2026-08-13) found
    bbox diagonal alone varies >13x across just 5 sampled specimens (443.8 to 5949.3), so any
    single absolute-length constant is provably wrong for most of the corpus. relative_alpha (CGAL's
    own scale-invariant unit) makes bisection cheap (~6-8 CGAL calls/specimen, each a few seconds
    to low tens of seconds at these mesh sizes) and correct per-specimen, instead of one global
    constant tuned on a single mesh.

    Bounds default to alpha_fused_start=pre_gap*50 (expected coarse/fused, DOUBLED up to
    max_bound_widen_retries times if that guess turns out not coarse enough to fuse - see below)
    and alpha_preserved_start=pre_gap*5 (expected fine/preserved, HALVED up to
    max_bound_widen_retries times if that guess turns out not fine enough - see below).

    offset_alpha_ratio default CHANGED (2026-08-16) from 30.0 to 2.5 (offset = alpha/2.5 =
    alpha*0.4, was alpha/30 = alpha*0.033): root-caused a 44%-of-corpus fallback-to-ManifoldPlus
    rate on bench50_clean (15/22 explicit CGAL timeouts, 2/22 CGAL OOM-killed during bisection,
    both consistent with the same mechanism, 5/22 unrelated - see below). offset only controls
    surface-hugging tightness/refinement cost - CGAL's docs are explicit it does not gate carving,
    so it has nothing to do with what this function's bisection is actually solving for (the
    coarsest alpha that still preserves the specimen's tightest self-approach gap). Forcing
    offset=alpha/30 on every one of ~6-8 CGAL calls per specimen inflated refinement cost
    mesh-wide (worse on many-limbed specimens) independent of the gap-preservation search alpha
    was actually converging on. offset_alpha_ratio=2.5 (offset=alpha*0.4) matches CGAL's own
    wrapwrap reference tool's recommended starting ratio (relative_alpha=500/relative_offset=1200
    -> offset ~= alpha*0.417). Should NOT change gap-preservation verdicts (offset doesn't gate
    carving) - only wall-clock/memory per CGAL call. Does NOT address the other bench50_clean
    fallbacks that failed for an unrelated reason (alpha_fused_start didn't actually fuse the gap
    on those specimens - a bisection-bounds problem, not an offset/cost problem) - see the
    alpha_fused_start doubling fix below (2026-08-17).

    FIX (2026-08-17): alpha_fused_start's fixed pre_gap*15 guess did not fuse the gap on a subset
    of specimens (root cause of most of the remaining post-offset-fix ManifoldPlus fallbacks -
    confirmed on the bench50_offsetfix_20260816 corpus), so the coarse bound previously raised
    RuntimeError immediately on a bad guess instead of self-correcting. Now mirrors the
    already-validated alpha_preserved_start pattern: starts wider (pre_gap*50) and, if that still
    doesn't fuse, doubles and retries up to max_bound_widen_retries times before giving up. This is
    the same "confirm-then-bisect" shape as the preserved-side halving, just widening instead of
    narrowing since the fused bound needs to get coarser (not finer) to guarantee fusion.

    CORRECTION (2026-08-13): an earlier version of this docstring claimed the observed
    fused/preserved transition "fell inside 5x-15x" across both characterized specimens - that was
    a misread of the actual sweep data. Rechecked: Metapone preserved at ~2.17x pre_gap, local
    Dorylus preserved at ~12.93x pre_gap - the safe ratio varies specimen-to-specimen by at least
    ~6x, so pre_gap*5 is NOT a universally safe "definitely preserved" starting guess (confirmed:
    it came back fused, not preserved, on worker_ALT's Dorylus_sp._CASENT0744698 without
    clean_internal_geometry pre-cleaning - pre_gap there was a genuine isolated tight self-approach
    point, not internal-duplicate-surface noise, ruled out by inspecting local vertex density
    around target_a/target_b directly). Rather than pick a new fixed constant that could just as
    easily be wrong on the next specimen, alpha_preserved_start now self-corrects: if it comes back
    fused instead of preserved, it's halved and retried (up to max_bound_widen_retries times)
    before giving up.

    FAILS LOUDLY (raises RuntimeError) instead of silently returning an unverified alpha if:
      - the coarse bound doesn't actually fuse the gap (bounds are wrong for this specimen), or
      - no preserving fine bound is found within max_bound_widen_retries halvings, or
      - any CGAL subprocess call times out, OOMs (mem_cap_kb, matching the sweep's safeguard), or
        otherwise errors during bisection.
    Per explicit instruction: a specimen that can't be bisected cleanly gets flagged for manual
    alpha selection, not silently handed a value nobody verified.

    Reuses the LAST confirmed-preserved CGAL output (the final `hi`) as the actual reconstruction
    result rather than re-running CGAL a final time - that .off is already sitting on disk from
    the bisection itself.

    Returns:
        dict: {"alpha", "offset", "off_path", "history", "alpha_fused_start",
              "alpha_preserved_start", "total_seconds", "peak_memory_kb"}. Caller is responsible
              for cleaning up work_dir/off_path after consuming it.
    """
    import re as _re
    import trimesh
    from scipy.spatial import cKDTree

    work_dir = work_dir or tempfile.mkdtemp(prefix="alpha_bisect_")
    os.makedirs(work_dir, exist_ok=True)

    alpha_fused_start = alpha_fused_start if alpha_fused_start is not None else pre_gap * 50.0
    alpha_preserved_start = (
        alpha_preserved_start if alpha_preserved_start is not None else pre_gap * 5.0
    )
    if alpha_fused_start <= alpha_preserved_start:
        raise RuntimeError(
            f"_bisect_alpha_wrap_params: alpha_fused_start ({alpha_fused_start:.4g}) must be "
            f"coarser (larger) than alpha_preserved_start ({alpha_preserved_start:.4g})."
        )

    history = []
    bisection_start_time = time.time()
    peak_memory_kb = [None]  # mutable cell, updated by _run(); max across all calls

    def _check_ceiling(next_label):
        # Hard per-specimen compute ceiling (2026-08-17): a bounded TOTAL budget across the whole
        # bisection, not just a per-call timeout - see _TIER_BISECTION_KWARGS docstring for why
        # (a genuine tight near-touch can keep demanding a finer alpha with no guaranteed
        # convergence point; without this, one pathological specimen could consume unbounded
        # compute across repeated bound-widening/bisection iterations even if each individual CGAL
        # call stays under its own `timeout`).
        if max_total_seconds is None:
            return
        elapsed = time.time() - bisection_start_time
        if elapsed > max_total_seconds:
            raise RuntimeError(
                f"_bisect_alpha_wrap_params: hard compute ceiling exceeded "
                f"({elapsed:.0f}s > max_total_seconds={max_total_seconds}s) before {next_label} - "
                f"flag this specimen for manual alpha selection rather than continuing to escalate."
            )

    def _run(alpha, label):
        _check_ceiling(label)
        offset = alpha / offset_alpha_ratio
        relative_alpha = diag_length / alpha
        relative_offset = diag_length / offset
        input_base = os.path.splitext(os.path.basename(obj_export_path))[0]
        predicted_path = os.path.join(
            work_dir, f"{input_base}_{int(relative_alpha)}_{int(relative_offset)}.off"
        )
        out_path = os.path.join(work_dir, f"{label}.off")
        # Wrapped in `/usr/bin/time -v` to capture peak resident memory per CGAL call (2026-08-17,
        # part of the per-specimen diagnostic table) - its report goes to stderr alongside CGAL's
        # own stderr, parsed out below rather than replacing capture_output entirely.
        cmd = (
            f"ulimit -v {mem_cap_kb}; cd {work_dir} && /usr/bin/time -v {alpha_wrap_binary_path} "
            f"{obj_export_path} {relative_alpha} {relative_offset}"
        )
        call_start = time.time()
        try:
            result = subprocess.run(
                ["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"_bisect_alpha_wrap_params: CGAL timed out after {timeout}s at {label} "
                f"(alpha={alpha:.4g}, relative_alpha={relative_alpha:.4g}) - flag this specimen "
                f"for manual alpha selection."
            )
        call_seconds = time.time() - call_start
        mem_match = _re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", result.stderr)
        call_peak_kb = int(mem_match.group(1)) if mem_match else None
        if call_peak_kb is not None and (peak_memory_kb[0] is None or call_peak_kb > peak_memory_kb[0]):
            peak_memory_kb[0] = call_peak_kb
        if result.returncode != 0 or not os.path.isfile(predicted_path):
            raise RuntimeError(
                f"_bisect_alpha_wrap_params: CGAL failed at {label} (alpha={alpha:.4g}, "
                f"relative_alpha={relative_alpha:.4g}, exit_code={result.returncode}) - flag this "
                f"specimen for manual alpha selection.\nstdout: {result.stdout[-500:]}\n"
                f"stderr: {result.stderr[-500:]}"
            )
        os.replace(predicted_path, out_path)

        mesh = trimesh.load(out_path, process=False)
        tree = cKDTree(mesh.vertices)
        _, ia = tree.query(target_a)
        _, ib = tree.query(target_b)
        gap_after = float(np.linalg.norm(mesh.vertices[ia] - mesh.vertices[ib]))
        if ia == ib:
            verdict = "fused"
        elif gap_after < pre_gap * 0.3:
            verdict = "suspect"
        else:
            verdict = "preserved"
        history.append({"label": label, "alpha": alpha, "offset": offset, "verdict": verdict,
                         "gap_after": gap_after, "call_seconds": call_seconds,
                         "call_peak_memory_kb": call_peak_kb})
        print(f"_bisect_alpha_wrap_params: {label} alpha={alpha:.4g} "
              f"relative_alpha={relative_alpha:.4g} -> verdict={verdict} gap_after={gap_after:.4g} "
              f"call_seconds={call_seconds:.1f} call_peak_memory_kb={call_peak_kb}")
        return verdict, out_path

    v_lo, lo_off_path = _run(alpha_fused_start, "bound_fused")
    fused_widen_attempt = 0
    while v_lo != "fused" and fused_widen_attempt < max_bound_widen_retries:
        fused_widen_attempt += 1
        alpha_fused_start = alpha_fused_start * 2.0
        print(f"_bisect_alpha_wrap_params: coarse bound didn't fuse the gap (verdict={v_lo!r}) - "
              f"doubling to alpha={alpha_fused_start:.4g} and retrying (widen attempt "
              f"{fused_widen_attempt}/{max_bound_widen_retries})...")
        v_lo, lo_off_path = _run(alpha_fused_start, f"bound_fused_widen{fused_widen_attempt}")
    if v_lo != "fused":
        raise RuntimeError(
            f"_bisect_alpha_wrap_params: coarse bound alpha={alpha_fused_start:.4g} did NOT fuse "
            f"the gap after {max_bound_widen_retries} doublings (verdict={v_lo!r}) - bisection "
            f"bounds are wrong for this specimen, flag for manual alpha selection."
        )
    hi = alpha_preserved_start
    v_hi, hi_off_path = _run(hi, "bound_preserved")
    widen_attempt = 0
    while v_hi != "preserved" and widen_attempt < max_bound_widen_retries:
        widen_attempt += 1
        hi = hi / 2.0
        print(f"_bisect_alpha_wrap_params: fine bound didn't preserve the gap (verdict={v_hi!r}) - "
              f"halving to alpha={hi:.4g} and retrying (widen attempt "
              f"{widen_attempt}/{max_bound_widen_retries})...")
        v_hi, hi_off_path = _run(hi, f"bound_preserved_widen{widen_attempt}")
    if v_hi != "preserved":
        raise RuntimeError(
            f"_bisect_alpha_wrap_params: no preserving fine bound found after "
            f"{max_bound_widen_retries} halvings (last tried alpha={hi:.4g}, verdict={v_hi!r}) - "
            f"flag this specimen for manual alpha selection."
        )
    alpha_preserved_start = hi

    lo = alpha_fused_start
    best_off_path = hi_off_path
    for i in range(max_iters):
        mid = (lo + hi) / 2.0
        verdict, off_path = _run(mid, f"iter{i}")
        if verdict == "fused":
            lo = mid
        elif verdict == "preserved":
            hi = mid
            best_off_path = off_path
        else:
            raise RuntimeError(
                f"_bisect_alpha_wrap_params: inconclusive verdict {verdict!r} at iter{i} "
                f"(alpha={mid:.4g}) - flag this specimen for manual alpha selection."
            )

    alpha = hi
    offset = alpha / offset_alpha_ratio
    total_seconds = time.time() - bisection_start_time
    print(f"_bisect_alpha_wrap_params: converged, coarsest-safe alpha={alpha:.6g} "
          f"(alpha/pre_gap={alpha / pre_gap:.4g}) after {max_iters} iterations. "
          f"total_seconds={total_seconds:.1f} peak_memory_kb={peak_memory_kb[0]}")

    return {
        "alpha": alpha,
        "offset": offset,
        "off_path": best_off_path,
        "history": history,
        "alpha_fused_start": alpha_fused_start,
        "alpha_preserved_start": alpha_preserved_start,
        "total_seconds": total_seconds,
        "peak_memory_kb": peak_memory_kb[0],
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
                                alpha_derivation_kwargs=None, tmp_dir=None,
                                target_a=None, target_b=None, pre_gap=None,
                                use_bisection=True, bisection_kwargs=None):
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

    ALPHA SELECTION (updated 2026-08-13): three ways alpha/offset get set, in priority order:
      1. Explicit alpha AND offset passed in - used as-is, nothing else runs.
      2. use_bisection=True (default) and target_a/target_b/pre_gap are given and finite -
         _bisect_alpha_wrap_params finds the coarsest (cheapest) alpha that still preserves this
         specimen's own tightest self-approach gap, ~6-8 CGAL calls. This is now the PRODUCTION
         path - see _bisect_alpha_wrap_params's docstring for why a fixed formula/constant cannot
         generalize across specimens whose scale varies >13x (worker_ALT investigation,
         2026-08-13).
      3. Fallback: _derive_alpha_wrap_params's edge-percentile/min_alpha-floor formula, used only
         when no landmark is available (e.g. the specimen has no detectable self-approach gap) or
         use_bisection=False is passed explicitly.

    Args:
        obj (bpy.types.Object): The Blender object to rebuild in place.
        alpha_wrap_binary_path (str): Path to the compiled triangle_soup_wrap executable.
                                      Defaults to DEFAULT_ALPHA_WRAP_BINARY_PATH.
        alpha, offset (float or None): Target ABSOLUTE lengths (mesh units). Explicit floats
                                       bypass both bisection and formula derivation.
        alpha_derivation_kwargs (dict): extra kwargs forwarded to _derive_alpha_wrap_params (the
                                        formula-based fallback path only).
        tmp_dir (str): Directory for intermediate files. Defaults to system temp dir.
        target_a, target_b, pre_gap: the specimen's tightest self-approach gap landmark, as
                                     computed by process_stl BEFORE reconstruction (same landmark
                                     its own mandatory fidelity gate uses afterward). Required for
                                     the bisection path.
        use_bisection (bool): Set False to force the old formula-based derivation even when a
                              landmark is available (e.g. for A/B comparison against the bisection
                              path).
        bisection_kwargs (dict): extra kwargs forwarded to _bisect_alpha_wrap_params.

    Returns:
        dict: {"alpha", "offset", "relative_alpha", "relative_offset", "output_vertices",
              "output_faces"}, plus "alpha_bisection" (bisection history/bounds) or
              "alpha_derivation" (formula stats) depending on which path ran.
    """
    if alpha_wrap_binary_path is None:
        alpha_wrap_binary_path = DEFAULT_ALPHA_WRAP_BINARY_PATH
    if not os.path.isfile(alpha_wrap_binary_path):
        raise FileNotFoundError(
            f"Alpha-wrap binary not found at {alpha_wrap_binary_path}. Build it per "
            f"custom_processing/external/ALPHA_WRAP_SETUP.md."
        )

    tmp_dir = tmp_dir or tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    unique_tag = f"{obj.name}_{os.getpid()}"
    tmp_input = os.path.join(tmp_dir, f"_alphawrap_in_{unique_tag}.obj")

    print(f"close_holes_via_alpha_wrap: exporting current mesh ({len(obj.data.vertices)} verts, "
          f"{len(obj.data.polygons)} faces) to {tmp_input}...")
    base.export_mesh_to_obj(obj, tmp_input)

    # bbox-diagonal, needed either way (bisection's relative_alpha conversion, or the direct
    # subprocess call below) - same formula as the C++ source (diag_length / relative_*), computed
    # on the identical point set just exported to tmp_input.
    verts, _ = base._triangulated_verts_faces(obj)
    diag_length = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    if diag_length <= 0:
        raise RuntimeError("close_holes_via_alpha_wrap: degenerate mesh (zero bbox diagonal).")

    alpha_derivation = None
    bisection_result = None
    reused_off_path = None

    if alpha is not None and offset is not None:
        pass  # explicit override - nothing else to derive
    elif (use_bisection and target_a is not None and target_b is not None
          and pre_gap is not None and np.isfinite(pre_gap)):
        print("close_holes_via_alpha_wrap: bisecting per-specimen alpha (production method)...")
        bisection_result = _bisect_alpha_wrap_params(
            tmp_input, target_a, target_b, pre_gap, diag_length, alpha_wrap_binary_path,
            **(bisection_kwargs or {}),
        )
        alpha = bisection_result["alpha"]
        offset = bisection_result["offset"]
        reused_off_path = bisection_result["off_path"]
    else:
        print("close_holes_via_alpha_wrap: no self-approach-gap landmark available (or "
              "use_bisection=False) - falling back to formula-based _derive_alpha_wrap_params...")
        alpha_derivation = _derive_alpha_wrap_params(obj, **(alpha_derivation_kwargs or {}))
        alpha = alpha if alpha is not None else alpha_derivation["alpha"]
        offset = offset if offset is not None else alpha_derivation["offset"]

    relative_alpha = diag_length / alpha
    relative_offset = diag_length / offset

    if reused_off_path is not None:
        print(f"close_holes_via_alpha_wrap: reusing bisection's final CGAL output "
              f"({reused_off_path}) - alpha={alpha:.4g} -> relative_alpha={relative_alpha:.4g}, "
              f"offset={offset:.4g} -> relative_offset={relative_offset:.4g}.")
        tmp_output_off = reused_off_path
    else:
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
        # zero same as C++'s static_cast<int>, so this must match bit-for-bit given the same
        # inputs.
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
    os.remove(tmp_output_obj)
    if bisection_result is not None:
        import shutil
        shutil.rmtree(os.path.dirname(bisection_result["off_path"]), ignore_errors=True)
    else:
        os.remove(tmp_output_off)

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
    if bisection_result is not None:
        result_dict["alpha_bisection"] = {
            k: v for k, v in bisection_result.items() if k != "off_path"
        }
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


def _decimate_with_gap_preservation(
    obj, max_vertices, target_a, target_b, pre_gap,
    max_vertices_growth_factor=2.0, max_growth_attempts=4, max_vertices_hard_cap=1_000_000,
    correspondence_gate_factor=20, max_decimate_iterations=25,
):
    """
    Wraps base.decimate_mesh() (NEVER modified directly - standing constraint on this shared,
    validated function) with a gap-preservation retry.

    2026-08-13 finding: decimate_mesh()'s own retry logic only guards against topology damage
    (boundary/non-manifold edges) - it has zero self-approach-gap awareness, the same generic-
    edge-collapse risk already documented for Weld/remove_doubles elsewhere in this pipeline.
    Confirmed on worker_ALT's Dorylus_sp._CASENT0744698: bisection converged to a fine alpha
    (needed because this specimen's pre_gap=0.925 is tiny relative to its scale), producing
    1.2M vertices; a checkpoint immediately after reconstruction (before decimation) showed the
    gap correctly preserved (verdict='preserved', gap_after=1.2) - but decimating that same mesh
    down to max_vertices=50,000 (a ~23x reduction) fused it (verdict='fused', gap_after=0)
    despite decimate_mesh() reporting clean topology (0 boundary/non-manifold edges) throughout.
    So a specimen can pass every check decimate_mesh() itself performs while still silently
    destroying the one landmark process_stl's own gate exists to protect.

    POLICY (2026-08-13, confirmed): max_vertices is a SOFT target, not a correctness criterion.
    fidelity_gate_passed (gap preservation + surface-deviation check) is the actual signal for
    whether reconstruction is correct - hitting max_vertices is not, and overrunning it is not a
    defect to eliminate. Observed on the first 5 worker_ALT specimens tested at the same
    max_vertices=50,000 starting point: final vertex counts ranged 6,998 to 419,925 (60x spread) -
    that's specimen-dependent self-approach-gap geometry (biology), not pipeline malfunction. When
    the soft target conflicts with gap preservation, gap preservation wins - this function's whole
    job is making that trade correctly, not apologizing for the vertex count it produces.

    max_vertices_hard_cap IS a real limit, but a different kind: a compute/memory bound on the
    retry ladder itself (decimation cost on a multi-hundred-thousand-vertex working set, not a
    quality target). If gap preservation still can't be reached by this ceiling, give up loudly -
    same fail-loud philosophy as bisection - rather than growing without bound.

    Fix: after decimating to max_vertices, check gap preservation. If fused, retry from a FRESH
    pristine copy of the pre-decimation mesh at a looser (max_vertices *= growth_factor, capped at
    max_vertices_hard_cap) vertex ceiling - less aggressive reduction gives the collapse algorithm
    more room to avoid the gap - up to max_growth_attempts times. Retrying at the SAME max_vertices
    would be pointless: decimate_mesh() is deterministic given the same input/target, so only a
    looser target can change the outcome. FAILS LOUDLY-adjacent (does not raise, matching
    process_stl's existing "export anyway, flag, don't silently drop" policy) if no looser cap
    within the attempt/hard-cap budget preserves the gap - "gap_preserved": False signals
    process_stl's caller to treat this the same as any other fidelity-gate failure.

    Args:
        obj (bpy.types.Object): The Blender object to decimate in place.
        max_vertices (int): Starting SOFT target vertex count (grows on retry, never shrinks).
        target_a, target_b, pre_gap: the specimen's tightest self-approach gap landmark.
        max_vertices_growth_factor (float): Multiplier applied to max_vertices on each retry.
        max_growth_attempts (int): Cap on retries before giving up.
        max_vertices_hard_cap (int): Absolute ceiling on the retry ladder - compute/memory bound,
                                     not a quality target. Retries never exceed this regardless of
                                     max_growth_attempts.
        correspondence_gate_factor (float): Forwarded to base.verify_gap_preservation.
        max_decimate_iterations (int): Forwarded to base.decimate_mesh's own max_iterations.

    Returns:
        dict: {"vertex_count", "max_vertices_used", "attempts", "gap_check", "gap_preserved"}.
    """
    pristine_mesh = obj.data.copy()
    attempt = 0
    current_max_vertices = max_vertices
    while True:
        attempt_mesh = pristine_mesh.copy()
        old_mesh = obj.data
        obj.data = attempt_mesh
        bpy.data.meshes.remove(old_mesh)

        vertex_count = base.decimate_mesh(obj, current_max_vertices,
                                           max_iterations=max_decimate_iterations)
        gap_check = base.verify_gap_preservation(
            obj, target_a, target_b, pre_gap, correspondence_gate_factor=correspondence_gate_factor
        )
        print(f"_decimate_with_gap_preservation: attempt {attempt} at max_vertices="
              f"{current_max_vertices:,} -> {vertex_count:,} verts, gap verdict="
              f"{gap_check['verdict']!r}")

        if gap_check["verdict"] != "fused":
            bpy.data.meshes.remove(pristine_mesh)
            return {
                "vertex_count": vertex_count, "max_vertices_used": current_max_vertices,
                "attempts": attempt + 1, "gap_check": gap_check, "gap_preserved": True,
            }

        hit_hard_cap = current_max_vertices >= max_vertices_hard_cap
        if attempt >= max_growth_attempts or hit_hard_cap:
            bpy.data.meshes.remove(pristine_mesh)
            reason = (f"hard cap {max_vertices_hard_cap:,} reached" if hit_hard_cap
                      else f"{attempt + 1} attempts exhausted")
            print(f"_decimate_with_gap_preservation: WARNING - gap still fused ({reason}, last "
                  f"tried max_vertices={current_max_vertices:,}) - giving up, flag this specimen "
                  f"for manual review.")
            return {
                "vertex_count": vertex_count, "max_vertices_used": current_max_vertices,
                "attempts": attempt + 1, "gap_check": gap_check, "gap_preserved": False,
            }

        attempt += 1
        current_max_vertices = min(
            int(current_max_vertices * max_vertices_growth_factor), max_vertices_hard_cap
        )
        print(f"_decimate_with_gap_preservation: gap fused - retrying from pristine with looser "
              f"cap max_vertices={current_max_vertices:,} (attempt {attempt}/"
              f"{max_growth_attempts}, hard_cap={max_vertices_hard_cap:,})...")


def _decimate_with_fidelity_stopping(
    obj, pre_decimation_path, target_cell_size=None,
    target_a=None, target_b=None, pre_gap=None,
    fidelity_stop_factor=3.0, edge_percentile_fallback=10,
    ratio_step=0.8, min_ratio_step=0.99, max_retries_per_step=6, max_iterations=25,
    correspondence_gate_factor=20,
):
    """
    Decimates obj in place via Blender's DECIMATE/COLLAPSE modifier - the same primitive
    base.decimate_mesh() uses internally (never modified directly - standing constraint on that
    shared, validated function) - but with a fundamentally different stopping philosophy: fidelity
    is the PRIMARY stop condition, vertex count is not tracked as a target at all.

    POLICY (2026-08-13, confirmed): downstream (mesh fitting) doesn't need a fixed vertex count -
    _decimate_with_gap_preservation already established max_vertices as a soft target, not a
    correctness criterion. This function goes one step further and drops the vertex-count target
    entirely from the STOPPING decision (process_stl still uses a soft max_vertices to decide
    whether to attempt decimation at all - no point running this on an already-small mesh - but
    once started, only fidelity/gap/topology decide when to stop).

    At each successful (topology-clean) decimation step, checks base.verify_reconstruction_fidelity
    against pre_decimation_path - the mesh state RIGHT AFTER reconstruction, before ANY
    decimation - NOT the pre-RECONSTRUCTION snapshot process_stl's own fidelity gate uses. This
    isolates decimation's own error from reconstruction's, the same isolation principle already
    established for the gap-preservation checkpoint (see process_stl's "DIAGNOSTIC CHECKPOINT"
    comment). Stops - keeping the last step that stayed within fidelity_stop_factor *
    target_cell_size - the moment the next step would exceed it. Also checks self-approach gap
    preservation at each accepted step if target_a/target_b/pre_gap are given (same check
    _decimate_with_gap_preservation uses, folded in here rather than run as a separate pass).

    Reimplements decimate_mesh()'s own topology-retry pattern directly (gentler ratio on
    boundary/non-manifold damage, same step-halving logic, same pristine-copy-per-attempt
    discipline) rather than calling decimate_mesh() - required, since decimate_mesh() has no
    fidelity awareness and the constraint against modifying it directly still applies.

    target_cell_size: if None, derived from pre_decimation_path's own p{edge_percentile_fallback}
    edge length - gives manifold_plus reconstructions (which have no bisected/derived alpha to use
    directly, unlike alpha_wrap) the same "local edge length" scale to hold decimation to.

    Args:
        obj (bpy.types.Object): The Blender object to decimate in place.
        pre_decimation_path (str): Path to the mesh state immediately after reconstruction, before
                                   any decimation - the fidelity baseline this function measures
                                   its own error against.
        target_cell_size (float or None): Scale fidelity_stop_factor is multiplied against.
        target_a, target_b, pre_gap: optional self-approach gap landmark to also guard.
        fidelity_stop_factor (float): Same semantics/default as process_stl's fidelity_gate_factor.
        edge_percentile_fallback (float): Used to derive target_cell_size when not given.
        ratio_step, min_ratio_step, max_retries_per_step: mirror decimate_mesh()'s own step-size/
                                                           retry parameters.
        max_iterations (int): Cap on outer decimation steps.
        correspondence_gate_factor (float): Forwarded to base.verify_gap_preservation.

    Returns:
        dict: {"vertex_count", "stop_reason", "worst_fidelity_deviation", "fidelity_ceiling",
              "target_cell_size", "gap_check"}.
    """
    if target_cell_size is None:
        verts, faces = base._triangulated_verts_faces(obj)
        edge_lengths = np.linalg.norm(verts[faces[:, [1, 2, 0]]] - verts[faces], axis=-1).ravel()
        target_cell_size = (float(np.percentile(edge_lengths, edge_percentile_fallback))
                             if len(edge_lengths) else 1.0)
        print(f"_decimate_with_fidelity_stopping: no target_cell_size given - derived "
              f"{target_cell_size:.4g} from p{edge_percentile_fallback} edge length of the "
              f"pre-decimation mesh.")

    fidelity_ceiling = fidelity_stop_factor * target_cell_size
    print(f"_decimate_with_fidelity_stopping: fidelity ceiling = {fidelity_stop_factor}x"
          f"{target_cell_size:.4g} = {fidelity_ceiling:.4g}")

    last_gap_check = None
    if target_a is not None:
        last_gap_check = base.verify_gap_preservation(
            obj, target_a, target_b, pre_gap, correspondence_gate_factor=correspondence_gate_factor
        )
    last_deviation = 0.0
    last_vertex_count = len(obj.data.vertices)
    stop_reason = "max_iterations_reached"

    for iteration in range(max_iterations):
        accepted_mesh = obj.data
        ratio = ratio_step
        retry = 0
        step_succeeded = False

        while True:
            attempt_mesh = accepted_mesh.copy()
            obj.data = attempt_mesh

            modifier = obj.modifiers.new(name="Decimate", type="DECIMATE")
            modifier.decimate_type = "COLLAPSE"
            modifier.use_symmetry = False
            modifier.use_collapse_triangulate = True
            modifier.use_dissolve_boundaries = False
            modifier.ratio = ratio
            bpy.context.view_layer.objects.active = obj
            try:
                bpy.ops.object.modifier_apply(modifier="Decimate")
            except RuntimeError as e:
                if "Modifiers cannot be applied to multi-user data" in str(e):
                    bpy.ops.object.make_single_user(object=True, obdata=True, material=False,
                                                     animation=False)
                    bpy.ops.object.modifier_apply(modifier="Decimate")
                else:
                    raise

            integrity = base.mesh_integrity_report(obj)
            if integrity["boundary_edges"] == 0 and integrity["non_manifold_edges"] == 0:
                step_succeeded = True
                break

            print(f"_decimate_with_fidelity_stopping: iter{iteration} retry{retry} "
                  f"ratio={ratio:.4f} damaged topology "
                  f"(boundary_edges={integrity['boundary_edges']}, "
                  f"non_manifold_edges={integrity['non_manifold_edges']}) - retrying gentler.")
            obj.data = accepted_mesh
            bpy.data.meshes.remove(attempt_mesh)
            retry += 1
            if ratio > min_ratio_step or retry > max_retries_per_step:
                break
            ratio = 1.0 - (1.0 - ratio) / 2.0

        if not step_succeeded:
            stop_reason = "topology_damage_no_progress"
            break

        gap_check = None
        if target_a is not None:
            gap_check = base.verify_gap_preservation(
                obj, target_a, target_b, pre_gap,
                correspondence_gate_factor=correspondence_gate_factor
            )
        fidelity = base.verify_reconstruction_fidelity(pre_decimation_path, obj)
        deviation = max(fidelity["pre_to_post_p99.9"], fidelity["post_to_pre_p99.9"])
        gap_ok = gap_check is None or gap_check["verdict"] != "fused"
        fidelity_ok = deviation <= fidelity_ceiling
        vertex_count = len(obj.data.vertices)

        print(f"_decimate_with_fidelity_stopping: iter{iteration} ratio={ratio:.4f} -> "
              f"{vertex_count:,} verts, deviation={deviation:.4g} (ceiling={fidelity_ceiling:.4g}), "
              f"gap_verdict={gap_check['verdict'] if gap_check else 'n/a'!r}")

        if fidelity_ok and gap_ok:
            bpy.data.meshes.remove(accepted_mesh)
            last_gap_check = gap_check
            last_deviation = deviation
            last_vertex_count = vertex_count
            continue

        rejected_mesh = obj.data
        obj.data = accepted_mesh
        bpy.data.meshes.remove(rejected_mesh)
        stop_reason = "fidelity_exceeded" if not fidelity_ok else "gap_would_fuse"
        break

    print(f"_decimate_with_fidelity_stopping: stopped ({stop_reason}) at "
          f"{last_vertex_count:,} verts, worst_deviation={last_deviation:.4g} "
          f"(ceiling={fidelity_ceiling:.4g}).")

    return {
        "vertex_count": last_vertex_count,
        "stop_reason": stop_reason,
        "worst_fidelity_deviation": last_deviation,
        "fidelity_ceiling": fidelity_ceiling,
        "target_cell_size": target_cell_size,
        "gap_check": last_gap_check,
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
    run_internal_cleaning=None,
    reconstruction_method="alpha_wrap",
    alpha_wrap_binary_path=None,
    alpha=None,
    offset=None,
    alpha_derivation_kwargs=None,
    use_bisection=True,
    bisection_kwargs=None,
    manifoldplus_binary_path=None,
    manifoldplus_depth=8,
    allow_fallback=True,
    fidelity_gate_factor=3.0,
    cap_remaining_holes_flag=True,
    visualization_max_vertices=None,
):
    """
    Processes an STL file by importing, cleaning, reconstructing a watertight surface at FULL
    RESOLUTION, decimating (only if still needed), and re-orienting the mesh - the alpha-wrap
    architecture from the research notes that motivated this file (section 3): ray-cast clean ->
    reconstruct (alpha wrap, formally guaranteed enclosure, on the full-resolution cleaned mesh,
    no pre-reconstruction reduction of any kind) -> decimate to max_vertices only if reconstruction
    left the mesh above that target -> MANDATORY fidelity gate (every specimen, every method, no
    exceptions) -> orient -> export.

    REVERTED 2026-08-13 (see below) from a brief attempt at decimating BEFORE reconstruction -
    that broke on non-manifold ray-cast-cleaned input, see the inline comment at the removed call
    site for the full story. The actual, safe fix for reconstruction cost on a large
    full-resolution input is _derive_alpha_wrap_params's min_alpha floor, not pre-reconstruction
    decimation - confirmed on Metapone_emersoni_CASENT0745558 (220,083 verts / 431,137 faces):
    the unclamped derivation drove relative_alpha to ~1738 and the CGAL subprocess was OOM-killed;
    min_alpha=47 (now the default) fixed it - relative_alpha=20.02, CGAL completed in 0.63s,
    output watertight (0 boundary/non-manifold edges) at every stage checked.

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

    The fidelity gate (verify_reconstruction_fidelity against the decimated pre-reconstruction
    snapshot - the actual input reconstruction received, NOT the original cleaned mesh, so
    ordinary decimation error isn't misattributed to reconstruction) is NOT conditional on which
    method ran or whether a fallback fired - per the research notes' explicit instruction, this is
    the one check that must never be skipped, because aggregate watertightness/manifold-ness
    cannot see a silently-fused self-approach gap. It also now runs BEFORE the PCA/orientation
    block rather than after - base.verify_gap_preservation requires the same coordinate frame the
    target landmark was recorded in, which the orientation block's baked-in rotations break; see
    the inline comment at the gate itself for the full explanation of that fix. A second,
    non-gating end-to-end check (clean reference vs. final) is also logged for the record.

    POLICY (2026-08-13, confirmed): max_vertices is a SOFT target for the reconstruction/
    decimation pipeline, not a correctness criterion - fidelity_gate_passed (gap preservation +
    surface-deviation check) is the real signal. See _decimate_with_gap_preservation's docstring
    for the full reasoning and the 60x final-vertex-count spread observed across the first 5
    worker_ALT specimens tested (all at the SAME max_vertices starting point) - that variance is
    specimen-dependent self-approach-gap geometry, not a defect. If you need a specific vertex
    count for a downstream concern that is NOT reconstruction correctness (visualization, storage
    size), use visualization_max_vertices to get a SEPARATE, clearly-labeled export decimated to
    that count with no gap-preservation guarantee - do not conflate that need with this function's
    own max_vertices, which stays gap-preservation-first.

    visualization_max_vertices (int or None): if set and the final gap-preserving mesh exceeds
    this count, an ADDITIONAL `<name>_processed_viz.obj` is exported - a plain decimate_mesh() pass
    (no gap-preservation retry) purely for downstream uses that want a small, fast-to-load mesh and
    don't care about the self-approach-gap guarantee. Does not affect remaining_vertices,
    fidelity_gate_passed, the main `<name>_processed.obj` export, or the JSON record - a separate
    concern, a separate step, a separate file.

    run_internal_cleaning: whether to run clean_internal_geometry's ray-cast pass at all.
    None (default) auto-decides by input extension: True for .stl (unchanged behavior - the raw
    antscan scans this step was built for genuinely contain internal debris/noise that needs
    stripping), False for .obj (worker_ALT specimens - already low-poly, already mostly a single
    external shell per the 2026-08-13 connectivity check, so there is little/no real internal
    noise to remove). Pass an explicit True/False to override either way.

    Backstory (2026-08-13): clean_internal_geometry casts rays from a surrounding sphere AIMED AT
    THE OBJECT'S CENTROID (only a narrow +-20 degree jitter cone for the "secondary" rays) and
    deletes any face never hit. That reliably samples convex-ish mass (head/thorax/gaster) but
    systematically under-samples thin, multi-jointed, curved appendages that bend away from a
    straight line to the centroid (ant legs) - confirmed visually on
    Dorylus_sp._CASENT0744698 (worker_ALT): the pre-reconstruction diagnostic mesh already showed
    legs chopped into disconnected fragments and holes in the thorax/gaster BEFORE alpha-wrap ever
    ran, and clean_internal_geometry cut ~72% of the input's faces (35,293->11,601 verts) despite
    the input having almost no real debris to remove. Per user confirmation, other pipeline runs on
    other (raw STL) data have produced acceptable legs through this same function, so the fix here
    is to skip the step for input that doesn't need it, NOT to rewrite the shared ray-casting
    algorithm (that would need re-validation across every pipeline variant that imports it).
    """
    if reconstruction_method not in ("alpha_wrap", "manifold_plus"):
        raise ValueError(
            f"reconstruction_method must be 'alpha_wrap' or 'manifold_plus', "
            f"got {reconstruction_method!r}"
        )

    process_stl_start_time = time.time()
    base._clear_scene_mesh_objects()

    # Despite the name (kept for backward compat with existing call sites/JSON-sidecar naming
    # below), stl_path also accepts .obj input - needed for worker_ALT specimens (Google Drive
    # scan-time-processed .obj files, not raw .stl - see 2026-08-13 investigation), which are
    # already low-poly and mostly-single-component but NOT watertight, so still need this full
    # pipeline (find/filter components, clean_internal_geometry, reconstruction) run on them from
    # scratch exactly like a raw .stl would.
    input_ext = os.path.splitext(stl_path)[1].lower()
    if input_ext == ".stl":
        bpy.ops.wm.stl_import(filepath=stl_path)
    elif input_ext == ".obj":
        bpy.ops.object.select_all(action="DESELECT")
        try:
            bpy.ops.wm.obj_import(filepath=stl_path)
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=stl_path)
    else:
        raise ValueError(
            f"process_stl: unsupported input extension {input_ext!r} for {stl_path} - expected "
            f".stl or .obj."
        )
    obj = bpy.context.selected_objects[0]

    initial_vertices = len(obj.data.vertices)
    if initial_vertices > 2000000:
        print(f"Initial vertex count: {initial_vertices}. Reducing vertices...")
        reduced_vertices = base.reduce_vertices_by_distance(obj)
        print(f"Reduced vertex count: {reduced_vertices}")

    base.filter_small_components(obj, min_faces=min_island_faces)

    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")
    obj.location = Vector((0, 0, 0))

    # See run_internal_cleaning's docstring entry above for the full 2026-08-13 backstory: this
    # step's centroid-aimed ray-casting systematically fragments thin curved appendages (legs) -
    # confirmed on worker_ALT's Dorylus_sp._CASENT0744698. Auto-skip for .obj (worker_ALT - already
    # low-poly/single-shell, little real internal debris to remove) unless explicitly overridden;
    # unchanged (always runs) for .stl, since other pipeline runs on raw STL data have produced
    # acceptable legs through this same function per user confirmation.
    should_clean_internal = (
        run_internal_cleaning if run_internal_cleaning is not None else (input_ext == ".stl")
    )
    if should_clean_internal:
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
    else:
        print(f"Skipping clean_internal_geometry (run_internal_cleaning="
              f"{run_internal_cleaning!r}, input_ext={input_ext!r}).")

    # CLEAN reference: snapshot immediately after ray-cast cleaning. NOTE: as of the 2026-08-13
    # revert below (no pre-reconstruction decimation anymore), this is now byte-identical to the
    # PRE-RECONSTRUCTION reference exported a few lines down - reconstruction runs directly on
    # this same full-resolution mesh with nothing in between. Kept as a separate snapshot/file
    # anyway (not aliased to the same path) so the fidelity gate and the end-to-end diagnostic
    # below keep working unchanged off their own named variables - deliberately not touched by
    # this revert, which is scoped to removing the decimation call only.
    clean_reference_path = os.path.join(
        tempfile.gettempdir(), f"_alphawrap_cleanref_{obj.name}_{os.getpid()}.obj"
    )
    base.export_mesh_to_obj(obj, clean_reference_path)

    # REVERTED 2026-08-13: this used to decimate to max_vertices HERE, before reconstruction, to
    # keep the auto-derived alpha from getting pathologically fine on a large full-resolution
    # input. Removed because it broke on non-manifold input: base.decimate_mesh() requires
    # boundary_edges==0 AND non_manifold_edges==0 BEFORE it will make any progress at all (an
    # ABSOLUTE precondition, not "no worse than the input"), and clean_internal_geometry's output
    # is deliberately NOT manifold - that non-manifoldness is inherent to this pipeline's design
    # (Weld/EdgeSplit before reconstruction risks fusing close anatomical parts, which is the
    # entire reason alpha wrap is used here instead). On Metapone_emersoni_CASENT0745558
    # (220,083 verts / 431,137 faces, 39,852 boundary edges, 12,814 non-manifold edges right after
    # clean_internal_geometry) this made decimate_mesh raise "made ZERO progress" on every call,
    # before reconstruction ever ran - not a tuning problem, an unsatisfiable precondition.
    # base.decimate_mesh() itself is unchanged; do not reintroduce a pre-reconstruction call to it.
    # The actual, safe fix for reconstruction cost/memory on a large full-resolution input is
    # _derive_alpha_wrap_params's min_alpha floor (see its docstring) - confirmed on this same
    # specimen: the unclamped derivation drove relative_alpha to ~1738 and the CGAL subprocess was
    # OOM-killed (exit_code=-9); min_alpha=47 (now the default) fixed it - relative_alpha=20.02,
    # CGAL completed in 0.63s, output watertight at every stage checked. Reconstruction below now
    # always runs on the full-resolution mesh; decimation happens ONLY after, and only if needed
    # (see "Final decimation is now conditional" further down) - by then the mesh is already
    # watertight by construction, so base.decimate_mesh()'s manifold precondition is satisfied.
    pre_recon_vertex_count = len(obj.data.vertices)
    pre_recon_face_count = len(obj.data.polygons)
    pre_recon_hole_count = base.count_holes(obj)
    pre_recon_boundary_edges = base.count_boundary_edges(obj)
    pre_recon_integrity = base.mesh_integrity_report(obj)
    pre_recon_degeneracy = base.report_mesh_degeneracy(obj)
    print(f"DIAGNOSTIC pre-reconstruction input (what CGAL/ManifoldPlus actually receive, full "
          f"resolution - no pre-reconstruction decimation): vertices={pre_recon_vertex_count} "
          f"faces={pre_recon_face_count} holes={pre_recon_hole_count} "
          f"boundary_edges={pre_recon_boundary_edges}")

    # PRE-RECONSTRUCTION reference: the actual mesh handed to the reconstruction method (not
    # bpy.data.meshes.copy() - confirmed by base.process_stl's manifold_external branch to
    # measurably degrade Blender's per-iteration performance if kept as a live in-memory
    # datablock instead) - now the same full-resolution mesh as clean_reference_path above (see
    # note there). Also records the specimen's tightest self-approach gap, feeding the mandatory
    # fidelity gate below, unconditionally, regardless of which reconstruction method actually
    # ends up running.
    pre_recon_backup_path = os.path.join(
        tempfile.gettempdir(), f"_alphawrap_prerecon_{obj.name}_{os.getpid()}.obj"
    )
    base.export_mesh_to_obj(obj, pre_recon_backup_path)
    verts, faces = base._triangulated_verts_faces(obj)
    pre_gap, target_a, target_b = base._min_self_approach_gap(verts, faces, return_pair=True)

    # --- Geometry-aware tier routing (2026-08-17) ---
    # Cheap, no-CGAL-calls estimate of bisection cost/feasibility (see _estimate_relative_alpha's
    # docstring for why gap_ratio/relative_alpha_est, not raw pre_gap, is the right predictor) -
    # routes each specimen into Tier A/B/C BEFORE any CGAL subprocess runs, and auto-builds
    # per-specimen timeout/mem_cap_kb/max_total_seconds from that tier. REPLACES the
    # ALPHA_WRAP_BISECT_TIMEOUT env-var workaround with a principled per-specimen mechanism -
    # caller-supplied bisection_kwargs (if any) still take precedence, so explicit overrides keep
    # working.
    diag_length_est = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    gap_estimate = _estimate_relative_alpha(pre_gap, diag_length_est)
    tier = _classify_tier(gap_estimate["relative_alpha_est"])
    if bisection_kwargs is None:
        bisection_kwargs = dict(_TIER_BISECTION_KWARGS[tier])
    near_touch_diag = None
    if tier == "C":
        near_touch_diag = _classify_near_touch(verts, faces, target_a, target_b)
        print(f"DIAGNOSTIC Tier C near-touch check: {near_touch_diag}")
    pre_recon_thickness_p5 = _shape_diameter_thickness_p5(verts, faces)
    print(f"DIAGNOSTIC geometry-aware routing: pre_gap={pre_gap:.6g} diag_length={diag_length_est:.6g} "
          f"gap_ratio={gap_estimate['gap_ratio']:.6g} "
          f"relative_alpha_est={gap_estimate['relative_alpha_est']:.6g} tier={tier} "
          f"bisection_kwargs={bisection_kwargs} pre_recon_thickness_p5={pre_recon_thickness_p5:.6g}")
    # --- end geometry-aware tier routing ---

    # TEMPORARY DIAGNOSTIC (2026-08-13): persist a copy of the pre-reconstruction mesh (post-clean,
    # pre-alpha-wrap) so it can be visually compared against the final reconstructed output -
    # pre_recon_backup_path/clean_reference_path are both deleted before process_stl returns, so
    # without this copy there is no way to tell whether damage (holes, thin/broken legs) already
    # existed before reconstruction ran, or was introduced by alpha wrap itself. Added specifically
    # to investigate the worker_ALT Dorylus_sp._CASENT0744698 leg/hole artifact (bisection passed
    # every automated gate - watertight, fidelity, gap-preservation - but the exported mesh
    # visually shows large through-holes and thin wire-like legs). Remove once that's resolved -
    # not meant to be a permanent pipeline output.
    diagnostic_dir = os.path.join(tempfile.gettempdir(), "alphawrap_prerecon_diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    diagnostic_specimen_name = os.path.splitext(os.path.basename(stl_path))[0]
    prerecon_diagnostic_path = os.path.join(
        diagnostic_dir, f"{diagnostic_specimen_name}_PRE_RECONSTRUCTION.obj"
    )
    import shutil
    shutil.copy2(pre_recon_backup_path, prerecon_diagnostic_path)
    print(f"DIAGNOSTIC: pre-reconstruction mesh (post-clean, pre-alpha-wrap) saved to "
          f"{prerecon_diagnostic_path} for visual comparison against the final output.")

    actual_reconstruction_method = reconstruction_method
    target_cell_size = None

    print(f"Rebuilding watertight surface via {reconstruction_method}...")
    try:
        if reconstruction_method == "alpha_wrap":
            recon_stats = close_holes_via_alpha_wrap(
                obj, alpha_wrap_binary_path=alpha_wrap_binary_path, alpha=alpha, offset=offset,
                alpha_derivation_kwargs=alpha_derivation_kwargs,
                target_a=target_a, target_b=target_b, pre_gap=pre_gap,
                use_bisection=use_bisection, bisection_kwargs=bisection_kwargs,
            )
            # target_cell_size reflects the alpha reconstruction actually ran with - bisected
            # (production path, see close_holes_via_alpha_wrap) or formula-derived (fallback) -
            # the actual target resolution reconstruction was asked for on the input it actually
            # received, which is the logically correct value to gate against below.
            target_cell_size = (
                recon_stats.get("alpha_bisection", {}).get("alpha")
                or recon_stats.get("alpha_derivation", {}).get("alpha")
                or recon_stats.get("alpha")
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

    # TEMPORARY DIAGNOSTIC (2026-08-13): checkpoint gap-preservation immediately after
    # reconstruction, BEFORE filter_small_components/decimation run - isolates whether a
    # bisection-converged "preserved" verdict flips to "fused" during the .off->.obj->Blender
    # reimport round-trip itself, or only later (filter_small_components deleting a sliver
    # component one landmark vertex sat on, or decimation collapsing near it). Added to chase
    # down worker_ALT Dorylus_sp._CASENT0744698: bisection converged (approach=alpha_wrap, no
    # fallback) but process_stl's own post-hoc gap check still came back 'fused'. Remove once
    # resolved.
    if target_a is not None:
        checkpoint_gap = base.verify_gap_preservation(obj, target_a, target_b, pre_gap,
                                                        correspondence_gate_factor=20)
        print(f"DIAGNOSTIC CHECKPOINT (immediately after reconstruction, before "
              f"filter_small_components/decimation): verify_gap_preservation="
              f"{checkpoint_gap}")

    removed_islands = base.filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) after reconstruction")

    # PRE-DECIMATION diagnostic snapshot (2026-08-13) - the mesh state decimation will actually
    # start from, right after reconstruction/filter_small_components, before any decimation runs.
    # Needed to prove whether visible damage in a final output originates in reconstruction
    # (bisection choosing too coarse an alpha) or in decimation - without this, that question can
    # only be answered by in-memory checkpoints, not by actually opening the mesh. Also serves as
    # _decimate_with_fidelity_stopping's own fidelity baseline below. Persisted (not cleaned up)
    # for visual comparison, same convention as the PRE_RECONSTRUCTION diagnostic above.
    pre_decimation_diagnostic_path = os.path.join(
        diagnostic_dir, f"{diagnostic_specimen_name}_POST_RECONSTRUCTION_PRE_DECIMATION.obj"
    )
    base.export_mesh_to_obj(obj, pre_decimation_diagnostic_path)
    print(f"DIAGNOSTIC: post-reconstruction, pre-decimation mesh saved to "
          f"{pre_decimation_diagnostic_path} for visual comparison.")

    # Decimation happens ONLY here, after reconstruction - the mesh is watertight by construction
    # at this point (alpha wrap/ManifoldPlus's whole purpose), so base.decimate_mesh()'s manifold
    # precondition is satisfied, unlike the pre-reconstruction call this file used to make (see
    # the removed-call comment earlier in this function). Conditional rather than unconditional:
    # with min_alpha now flooring how fine reconstruction's own target gets, its output is often
    # already well under max_vertices on its own (confirmed: 528 verts on
    # Metapone_emersoni_CASENT0745558 at min_alpha=47, vs. a 50,000 target) - base.decimate_mesh
    # itself also no-ops below its 1.05x threshold, but this makes the skip explicit and logged
    # rather than silent. max_vertices here only gates WHETHER decimation is attempted at all (no
    # point running it on an already-small mesh) - it does NOT drive how far decimation goes once
    # started; see _decimate_with_fidelity_stopping's docstring (2026-08-13 policy).
    post_recon_vertices = len(obj.data.vertices)
    if post_recon_vertices > max_vertices * 1.05:
        print(f"Post-reconstruction mesh ({post_recon_vertices:,} verts) exceeds soft target "
              f"max_vertices={max_vertices:,} - decimating with fidelity as the stopping "
              f"criterion (not vertex count)...")
        # _decimate_with_gap_preservation (vertex-count-driven, superseded 2026-08-13) had no
        # awareness of whole-surface fidelity, only the one tracked landmark gap - confirmed on
        # Pheidole_nodus_CASENT0877547 that a reconstruction-stage coarse alpha alone can already
        # produce a large worst_fidelity_deviation while still passing the single-gap check.
        # _decimate_with_fidelity_stopping folds the same gap check in AND stops decimating the
        # moment whole-surface deviation crosses fidelity_stop_factor * target_cell_size,
        # regardless of whether max_vertices was reached.
        decimation_result = _decimate_with_fidelity_stopping(
            obj, pre_decimation_diagnostic_path, target_cell_size=target_cell_size,
            target_a=target_a, target_b=target_b, pre_gap=pre_gap,
            fidelity_stop_factor=fidelity_gate_factor,
        )
        remaining_vertices = decimation_result["vertex_count"]
        print(f"Decimation-with-fidelity-stopping: {decimation_result}")
    else:
        remaining_vertices = post_recon_vertices
        print(f"Post-reconstruction mesh ({post_recon_vertices:,} verts) already at/under soft "
              f"target max_vertices={max_vertices:,} - skipping decimation.")
    print(f"DIAGNOSTIC: holes after post-reconstruction decimation step: {base.count_holes(obj)}, "
          f"boundary_edges: {base.count_boundary_edges(obj)}")

    n_orphans_removed = base.remove_orphan_vertices(obj)
    if n_orphans_removed:
        remaining_vertices = len(obj.data.vertices)
        print(f"After remove_orphan_vertices: {remaining_vertices} vertices")

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
    #
    # BUG FIXED 2026-08-13: this gate now runs HERE - after reconstruction/post-decimation/hole-
    # capping, but BEFORE the PCA/orientation block below - not after it as an earlier version of
    # this file did. base.verify_gap_preservation's own docstring states the precondition this
    # violated explicitly: "Must be called on obj's CURRENT mesh in the SAME coordinate frame
    # target_a/target_b were recorded in - i.e. before any PCA realignment/rotation stage runs."
    # The orientation block calls bpy.ops.object.transform_apply(rotation=True) multiple times,
    # which BAKES rotation into vertex coordinates - running these checks after that point would
    # compare two different orientations of the same shape as if they were different geometry,
    # making every distance the gate computes meaningless (not just occasionally wrong - the old
    # ordering guaranteed a coordinate-frame mismatch on any specimen that needed any rotation at
    # all). base.verify_reconstruction_fidelity has the identical dependency (raw absolute-
    # coordinate nearest-surface distance, no registration step), so it shares the fix.
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

    # --- Thin-feature (SDF p5) production gate (2026-08-17) ---
    # Promotes the thickness_p5_ratio metric from compare_pipelines.py (offline-only until now)
    # into the production gate itself. Validated 2026-08-17: a cone-averaged cross-check confirmed
    # the single-ray p5 percentile is not noise-dominated. GATING (forces fidelity_gate_passed to
    # False) at least for the ManifoldPlus fallback path per explicit instruction - the offline
    # bench50 analysis found that path's aggregate p5 ratio catastrophic (median 0.0015-0.0047)
    # while fidelity_gate_passed still frequently read True, since until now nothing in the
    # production gate had any thin-feature awareness at all. Logged (not gating) for alpha_wrap,
    # since the alpha_wrap-only subset's aggregate p5 is healthy post offset-fix, but individual
    # specimens can still occasionally show collapse.
    post_verts, post_faces = base._triangulated_verts_faces(obj)
    post_recon_thickness_p5 = _shape_diameter_thickness_p5(post_verts, post_faces)
    if (pre_recon_thickness_p5 and np.isfinite(pre_recon_thickness_p5) and pre_recon_thickness_p5 > 0
            and np.isfinite(post_recon_thickness_p5)):
        thin_feature_p5_ratio = post_recon_thickness_p5 / pre_recon_thickness_p5
    else:
        thin_feature_p5_ratio = float("nan")
    thin_feature_gate_passed = True
    is_manifold_plus_path = "manifold_plus" in actual_reconstruction_method
    if (np.isfinite(thin_feature_p5_ratio) and thin_feature_p5_ratio < THIN_FEATURE_P5_RATIO_FLOOR
            and is_manifold_plus_path):
        thin_feature_gate_passed = False
        fidelity_gate_passed = False
        print(f"THIN-FEATURE GATE FAILED: p5 thickness ratio (post-reconstruction/pre-reconstruction) "
              f"{thin_feature_p5_ratio:.4g} is below {THIN_FEATURE_P5_RATIO_FLOOR} on the "
              f"ManifoldPlus fallback path - flag {os.path.basename(stl_path)} for MANUAL REVIEW, "
              f"thin anatomy (legs/antennae) likely collapsed.")
    else:
        print(f"Thin-feature gate: p5 thickness ratio (post/pre) = {thin_feature_p5_ratio:.4g} "
              f"(floor={THIN_FEATURE_P5_RATIO_FLOOR}, gating={is_manifold_plus_path}).")
    # --- end thin-feature production gate ---

    # Optional, NON-GATING end-to-end diagnostic: the gate above isolates reconstruction error
    # from decimation error by comparing against the decimated pre-reconstruction reference; this
    # additionally reports clean-reference-vs-final deviation (same pre-orientation coordinate
    # frame, so it's a valid comparison too) purely for the record. Not used for pass/fail - it
    # conflates two different error sources (decimation + reconstruction) by design, which is
    # exactly why it must not replace the gate above, only supplement it.
    end_to_end_fidelity = base.verify_reconstruction_fidelity(clean_reference_path, obj)
    end_to_end_worst_deviation = max(
        end_to_end_fidelity["pre_to_post_p99.9"], end_to_end_fidelity["post_to_pre_p99.9"]
    )
    print(f"End-to-end fidelity (clean reference vs. final, informational only): "
          f"worst deviation={end_to_end_worst_deviation:.4g}")
    os.remove(clean_reference_path)
    # --- end mandatory fidelity gate ---

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

    # remaining_vertices/hole_count were already established above (before orientation, alongside
    # the mandatory fidelity gate) - orientation only rotates the mesh, it cannot change vertex or
    # hole count, so both are still accurate here without recomputing.
    print(f"Number of remaining vertices: {remaining_vertices}")

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

    # POLICY (2026-08-13, confirmed): ManifoldPlus has no strict-enclosure guarantee (unlike
    # alpha_wrap by construction - see close_holes_via_manifold_plus's own docstring), so any
    # specimen actually reconstructed via it - deliberate reconstruction_method="manifold_plus" OR
    # the automatic fallback after alpha_wrap fails/times out - gets an explicit, separate flag
    # here, regardless of fidelity_gate_passed. A passed fidelity gate on a manifold_plus output is
    # NOT equivalent to a passed gate on an alpha_wrap output - it just means the fast checks this
    # pipeline can run didn't catch a problem, not that the same enclosure guarantee holds.
    used_weaker_reconstruction_guarantee = "manifold_plus" in actual_reconstruction_method
    if used_weaker_reconstruction_guarantee:
        print(f"*** {specimen_name}: reconstructed via {actual_reconstruction_method!r} - "
              f"ManifoldPlus has NO strict-enclosure guarantee (unlike alpha_wrap). Flag for "
              f"manual visual review regardless of fidelity_gate_passed - do not treat this as "
              f"equivalent to an alpha_wrap result. ***")

    bisection_stats = recon_stats.get("alpha_bisection", {}) if isinstance(recon_stats, dict) else {}
    bisection_total_seconds = bisection_stats.get("total_seconds")
    bisection_peak_memory_kb = bisection_stats.get("peak_memory_kb")
    print(
        f"FINAL_SUMMARY: specimen={specimen_name} approach={actual_reconstruction_method} "
        f"pre_recon_vertices={pre_recon_vertex_count} pre_recon_faces={pre_recon_face_count} "
        f"vertices={integrity['vertex_count']} faces={integrity['face_count']} "
        f"simple_holes={integrity['simple_holes']} boundary_edges={integrity['boundary_edges']} "
        f"non_manifold_edges={integrity['non_manifold_edges']} "
        f"non_manifold_verts={integrity['non_manifold_verts']} "
        f"non_manifold_vert_pct={non_manifold_vert_pct:.3f} "
        f"is_watertight={integrity['is_watertight']} signed_volume={integrity['signed_volume']} "
        f"face_size_cov={face_size_cov} mesh_smoothness={mesh_smoothness} "
        f"fidelity_gate_passed={fidelity_gate_passed} "
        f"worst_fidelity_deviation={worst_fidelity_deviation:.4g} "
        f"end_to_end_worst_deviation={end_to_end_worst_deviation:.4g} "
        f"time_sec={process_stl_elapsed:.1f} "
        f"pre_gap={pre_gap:.6g} diag_length={diag_length_est:.6g} "
        f"gap_ratio={gap_estimate['gap_ratio']:.6g} "
        f"relative_alpha_est={gap_estimate['relative_alpha_est']:.6g} tier={tier} "
        f"bisection_total_seconds={bisection_total_seconds} "
        f"bisection_peak_memory_kb={bisection_peak_memory_kb} "
        f"near_touch_normal_dot={near_touch_diag['normal_dot'] if near_touch_diag else None} "
        f"near_touch_interpretation={near_touch_diag['interpretation'] if near_touch_diag else None} "
        f"pre_recon_thickness_p5={pre_recon_thickness_p5:.6g} "
        f"post_recon_thickness_p5={post_recon_thickness_p5:.6g} "
        f"thin_feature_p5_ratio={thin_feature_p5_ratio:.6g} "
        f"thin_feature_gate_passed={thin_feature_gate_passed}"
    )

    if output_dir is None:
        output_dir = os.path.dirname(stl_path)
    os.makedirs(output_dir, exist_ok=True)

    original_file_name = os.path.splitext(os.path.basename(stl_path))[0]
    export_path = os.path.join(output_dir, f"{original_file_name}_processed.obj")

    print(f"Exporting the processed mesh to {export_path}...")
    base.export_mesh_to_obj(obj, export_path)
    print("Mesh exported successfully.")

    # SEPARATE, opt-in downstream-convenience export (2026-08-13 policy) - decoupled from
    # reconstruction correctness on purpose. This is a plain decimate_mesh() pass with NO
    # gap-preservation retry, so it does not carry the same guarantee as the main export above;
    # it exists only for callers that need a specific, small vertex count for something that isn't
    # reconstruction correctness (visualization, storage) and are fine trading that guarantee away
    # for it. Never touches remaining_vertices/fidelity_gate_passed/the main export/the JSON record.
    if visualization_max_vertices is not None and remaining_vertices > visualization_max_vertices:
        print(f"Creating a SEPARATE visualization export decimated to "
              f"{visualization_max_vertices:,} vertices (plain decimate_mesh, NO gap-preservation "
              f"guarantee - does not affect the main export or fidelity_gate_passed)...")
        viz_mesh = obj.data.copy()
        viz_obj = obj.copy()
        viz_obj.data = viz_mesh
        bpy.context.collection.objects.link(viz_obj)
        bpy.ops.object.select_all(action="DESELECT")
        viz_obj.select_set(True)
        bpy.context.view_layer.objects.active = viz_obj
        viz_vertex_count = base.decimate_mesh(viz_obj, visualization_max_vertices,
                                               max_iterations=25)
        viz_export_path = os.path.join(output_dir, f"{original_file_name}_processed_viz.obj")
        base.export_mesh_to_obj(viz_obj, viz_export_path)
        print(f"Visualization export ({viz_vertex_count:,} verts, gap preservation NOT "
              f"guaranteed) written to {viz_export_path}")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.data.objects.remove(viz_obj)
        bpy.data.meshes.remove(viz_mesh)

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
        json_data["processed_used_weaker_reconstruction_guarantee"] = used_weaker_reconstruction_guarantee
        json_data["processed_fidelity_gate_passed"] = fidelity_gate_passed
        json_data["processed_worst_fidelity_deviation"] = worst_fidelity_deviation
        json_data["processed_end_to_end_worst_deviation"] = end_to_end_worst_deviation
        json_data["processed_pre_reconstruction_vertex_count"] = pre_recon_vertex_count
        json_data["processed_pre_reconstruction_face_count"] = pre_recon_face_count
        json_data["processed_pre_reconstruction_integrity"] = pre_recon_integrity
        json_data["processed_pre_reconstruction_degeneracy"] = pre_recon_degeneracy
        json_data["processed_reconstruction_stats"] = recon_stats
        json_data["processed_geometry_routing"] = {
            "pre_gap": pre_gap, "diag_length": diag_length_est,
            "gap_ratio": gap_estimate["gap_ratio"],
            "relative_alpha_est": gap_estimate["relative_alpha_est"], "tier": tier,
            "bisection_kwargs": bisection_kwargs, "near_touch_diagnostic": near_touch_diag,
        }
        json_data["processed_thin_feature_gate"] = {
            "pre_recon_thickness_p5": pre_recon_thickness_p5,
            "post_recon_thickness_p5": post_recon_thickness_p5,
            "thin_feature_p5_ratio": thin_feature_p5_ratio,
            "thin_feature_gate_passed": thin_feature_gate_passed,
        }
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
        # bisection_kwargs left as None here (2026-08-17): process_stl now auto-derives
        # per-specimen timeout/mem_cap_kb/max_total_seconds from the geometry-aware tier
        # (_classify_tier(relative_alpha_est)) right where pre_gap is computed - this replaces the
        # earlier ALPHA_WRAP_BISECT_TIMEOUT env-var workaround, which set one global override for
        # every specimen in a run regardless of its own geometry.
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
