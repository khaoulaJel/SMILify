# Manifold (external mesh-repair tool) setup

[Manifold](https://github.com/hjwdzh/Manifold) is used as an alternative/supplementary
watertight-reconstruction step in the antscan mesh-prep pipeline
(`custom_processing/prepare_antscan_data_for_mesh_fitting_manifold.py`,
`hole_fill_method="manifold_external"`). It is a third-party tool, not vendored in this repo.

## Build

```bash
bash custom_processing/external/setup_manifold.sh
```

Requires `cmake` and a C++ build toolchain. Produces
`custom_processing/external/Manifold/build/manifold`, which is `.gitignore`d - rebuild it on
each machine that needs it.

## Usage

CLI signature: `manifold input.obj output.obj [resolution=20000] [-s]`. `resolution` is the
octree leaf-node count; 20000 is the default validated in
`diagnostics/MANIFOLD_REPAIR_METHODOLOGY.md` across 10 antscan specimens.

## Known defect signature and required fix

Manifold's raw octree reconstruction can leave isolated "bowtie" non-manifold vertices at thin
tapering features (leg/mandible/antenna tips) - two disjoint, individually-closed face-fans
meeting at exactly one point, with **zero** non-manifold edges (invisible to an edge-based
`face_count > 2` check). See `diagnostics/MANIFOLD_REPAIR_METHODOLOGY.md` for the full validated
diagnostic and repair procedure; the fix (`split_nonmanifold_vertex` - true vertex duplication,
not `bmesh.ops.split_edges` on the vertex's incident edges, which shreds each fan's own internal
connectivity) is implemented inline in
`custom_processing/prepare_antscan_data_for_mesh_fitting_manifold.py`.

**Important caveat:** the fix creates two vertices at identical coordinates. Any loader that
merges vertices by position (e.g. trimesh's default `trimesh.load()`, `process=True`) will
silently undo it. `pytorch3d.io.load_obj` (used by `fitter_3d/utils.py`'s `load_meshes`) and
Blender's own OBJ importer do **not** merge by position, so the fix is durable for this repo's
actual fitting pipeline. Always validate with `trimesh.load(path, process=False)`, not the
default.

## Required environment fix: scipy import inside Blender

**On Blender 4.2 LTS (current project standard - see root `CLAUDE.md`): no LD_PRELOAD needed.**
Blender 4.2's official Linux build bundles its own fully self-contained Python (3.11) with a
separate site-packages dir - `scipy` just needs to be installed into *that* interpreter once per
machine:

```bash
<path-to-blender-4.2>/4.2/python/bin/python3.11 -m pip install scipy
```

(e.g. `~/opt/blender-4.2.23-linux-x64/4.2/python/bin/python3.11 -m pip install scipy` on
khaoula's machine.) Verified 2026-08-13: `blender --background --python-expr "from scipy.spatial
import cKDTree"` succeeds with no other environment changes after this. No `LD_PRELOAD` or
conda-env cross-linking required or wanted here - keep Blender's Python and the `pytorch3d` conda
env fully separate.

This matters for every code path in this pipeline that imports `scipy` from inside Blender's own
Python process, not just `manifold_external`: also `decimate_mesh`'s
`check_bad_verts_edge_length_percentile` and `close_holes_via_winding_number`'s
`_min_self_approach_gap`.

<details>
<summary>Historical note: the old Blender 3.0.1 LD_PRELOAD workaround (obsolete, kept for context)</summary>

The previously-installed Blender 3.0.1 (an apt package) turned out to launch using the
`pytorch3d` conda env's own Python 3.10 interpreter as its embedded Python (`sys.executable`
pointed straight at `.../envs/pytorch3d/bin/python3.10`) rather than a bundled one - so it picked
up that env's compiled `scipy` extensions (`scipy.spatial._distance_pybind`), which required
`CXXABI_1.3.15`. Both Blender's own bundled `libstdc++.so.6` *and* the system one
(`/lib/x86_64-linux-gnu/`, max `CXXABI_1.3.13`) predated that, so the fix was to preload only the
conda env's own `libstdc++.so.6` before launching Blender (not the whole conda `lib/` dir on
`LD_LIBRARY_PATH` - tried first, broke unrelated Blender libraries `libcaca`/`libgdal` by
shadowing their dependencies):

```bash
export LD_PRELOAD="/home/khaoula/miniconda3/envs/pytorch3d/lib/libstdc++.so.6"
```

This was specific to that apt package's unusual (non-bundled-Python) build and does not apply to
Blender 4.2's official self-contained build.

</details>
