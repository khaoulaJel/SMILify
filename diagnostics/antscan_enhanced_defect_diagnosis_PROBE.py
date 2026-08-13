"""
Diagnoses the two specimens that hit decimate_mesh's pre-existing-topology-defect
RuntimeError in the 2026-08-11 antscan-enhanced batch run (Philidris_sp._CASENT0878052,
Centromyrmex_brachycola_CASENT0744052).

Reproduces process_stl's exact pipeline up to (but not including) decimate_mesh, using the
same kwargs the production batch job used, then:
  1. Checks the RAW scan's own bad-edge count, before any processing - tells us whether the
     defect is already present in the source data or introduced/left unrepaired by the pipeline.
  2. Captures the FULL bad-edge list at the pre-decimate_mesh state (report_bad_edges only
     prints `limit` edges by default, but always returns the complete list).
  3. Groups bad edges into connected loops via union-find on shared vertex coordinates, to
     distinguish "many small unclosed holes" (holes_fill should have closed these) from
     "one or two large defects" (more consistent with missing/fused scan geometry).

Run: blender --background --python-expr "import sys; sys.path.insert(0, '<blender_python_packages>')" \
     --python diagnostics/antscan_enhanced_defect_diagnosis_PROBE.py
"""
import bpy
import importlib.util
import os
from collections import defaultdict

ENHANCED_SCRIPT = "/p/home/jusers/jellal1/jureca/SMILify/custom_processing/prepare_antscan_data_for_mesh_fitting_enhanced.py"
spec = importlib.util.spec_from_file_location("enhanced_mod", ENHANCED_SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # guarded by `if __name__ == "__main__"`, safe to import

# Same kwargs main() actually passes to process_stl in production (NOT process_stl's own
# defaults, which differ: max_vertices=20000, secondary_rays=5, random_seed=42).
COMMON_KWARGS = dict(max_vertices=50000, ray_density=1000, secondary_rays=10000, random_seed=0, keep_rings=2)
MIN_ISLAND_FACES = 4
MIN_BRIDGE_ISLAND_FACES = 50
MAX_BRIDGE_HOPS = 20

SPECIMENS = [
    "/p/scratch/cias-7/jellal1/antscan/antscan_data/Philidris_sp._CASENT0878052/Philidris_sp._CASENT0878052.stl",
    "/p/scratch/cias-7/jellal1/antscan/antscan_data/Centromyrmex_brachycola_CASENT0744052/Centromyrmex_brachycola_CASENT0744052.stl",
]


def analyze_loop_sizes(bad_vert_coords):
    """Groups bad edges into connected loops via union-find on shared vertex coords (rounded to
    5dp - same bmesh vertex object reused across edges, so exact equality would also work, this
    is just defensive). Returns sorted list of edge-counts per loop, largest first."""
    edges = list(zip(bad_vert_coords[0::2], bad_vert_coords[1::2]))
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def key(c):
        return tuple(round(v, 5) for v in c)

    for a, b in edges:
        ka, kb = key(a), key(b)
        parent.setdefault(ka, ka)
        parent.setdefault(kb, kb)
        union(ka, kb)

    edge_counts = defaultdict(int)
    for a, _ in edges:
        edge_counts[find(key(a))] += 1

    return sorted(edge_counts.values(), reverse=True)


for stl_path in SPECIMENS:
    name = os.path.basename(stl_path)
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.wm.stl_import(filepath=stl_path)
    except AttributeError:
        mod.ensure_addon_enabled("io_mesh_stl")
        bpy.ops.import_mesh.stl(filepath=stl_path)
    obj = bpy.context.selected_objects[0]

    initial_vertices = len(obj.data.vertices)
    print(f"initial_vertices={initial_vertices}")

    print("--- RAW SCAN (before any processing) ---")
    raw_count, _ = mod.report_bad_edges(obj, limit=5)

    if initial_vertices > 2_000_000:
        mod.reduce_vertices_by_distance(obj)
    mod.find_largest_component(obj)
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")
    obj.location = mod.Vector((0, 0, 0))

    print("--- running clean_internal_geometry (ray-cast) ---")
    mod.clean_internal_geometry(
        obj,
        COMMON_KWARGS["ray_density"],
        COMMON_KWARGS["secondary_rays"],
        COMMON_KWARGS["random_seed"],
        keep_rings=COMMON_KWARGS["keep_rings"],
        min_island_faces=MIN_ISLAND_FACES,
        min_bridge_island_faces=MIN_BRIDGE_ISLAND_FACES,
        max_bridge_hops=MAX_BRIDGE_HOPS,
    )

    print("--- running apply_modifiers (includes holes_fill) ---")
    mod.apply_modifiers(obj, fill_holes_sides=0, min_island_faces=MIN_ISLAND_FACES)
    mod.filter_small_components(obj, min_faces=MIN_ISLAND_FACES)

    print("--- PRE-DECIMATE_MESH state (this is what raised the RuntimeError in production) ---")
    pre_count, pre_coords = mod.report_bad_edges(obj, limit=5)
    loop_sizes = analyze_loop_sizes(pre_coords)
    print(f"pre_decimate_bad_edges={pre_count}")
    print(f"n_distinct_loops={len(loop_sizes)}")
    print(f"loop_sizes (edge count, largest first, top 20): {loop_sizes[:20]}")
    if loop_sizes:
        print(f"largest_loop={loop_sizes[0]} edges, smallest_loop={loop_sizes[-1]} edges")
        n_small = sum(1 for s in loop_sizes if s <= 20)
        n_large = sum(1 for s in loop_sizes if s > 200)
        print(f"n_loops_<=20_edges={n_small}, n_loops_>200_edges={n_large}")

print("\nDIAGNOSIS_DONE")
