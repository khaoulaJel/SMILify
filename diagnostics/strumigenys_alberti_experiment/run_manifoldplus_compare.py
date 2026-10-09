"""Run ManifoldPlus directly on the same pre_reconstruction.obj used for the alpha_wrap sweep, and
compare topology/watertightness/runtime/memory against it, plus Chamfer/Hausdorff vs the raw
pre_reconstruction mesh (both "how much did each method change the input" and, once the sweep
picks a winning alpha_wrap output, vs that too).

Usage: python3 run_manifoldplus_compare.py
"""
import os
import re
import subprocess
import time
import numpy as np
import trimesh
from scipy.spatial import cKDTree

INPUT_OBJ = "/home/nao48500/SMILify/diagnostics/eyeball_groundtruth_20260817/06_FIDFAIL_Strumigenys_alberti/pre_reconstruction.obj"
MANIFOLDPLUS_BINARY = "/home/nao48500/SMILify/custom_processing/external/ManifoldPlus/build/manifold"
WORK_DIR = "/home/nao48500/SMILify/diagnostics/strumigenys_alberti_experiment/manifoldplus_work"
os.makedirs(WORK_DIR, exist_ok=True)

out_path = os.path.join(WORK_DIR, "manifoldplus_out.obj")
depth = 8
cmd = f"/usr/bin/time -v {MANIFOLDPLUS_BINARY} --input {INPUT_OBJ} --output {out_path} --depth {depth}"
t0 = time.time()
result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=1800)
elapsed = time.time() - t0
mem_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", result.stderr)
peak_kb = int(mem_match.group(1)) if mem_match else None

print(f"ManifoldPlus: exit={result.returncode} elapsed={elapsed:.1f}s peak_memory_kb={peak_kb}")
if result.returncode != 0 or not os.path.isfile(out_path):
    print("stdout:", result.stdout[-500:])
    print("stderr:", result.stderr[-500:])
else:
    mesh = trimesh.load(out_path, process=False)
    print(f"ManifoldPlus output: verts={len(mesh.vertices)} faces={len(mesh.faces)} "
          f"watertight={mesh.is_watertight} n_components={mesh.body_count} "
          f"winding_consistent={mesh.is_winding_consistent}")

    orig = trimesh.load(INPUT_OBJ, process=False)
    # Chamfer distance vs pre_reconstruction input (both directions), sample-based
    n_samples = 20000
    pts_a, _ = trimesh.sample.sample_surface(mesh, n_samples)
    pts_b, _ = trimesh.sample.sample_surface(orig, n_samples)
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_a2b, _ = tree_b.query(pts_a)
    d_b2a, _ = tree_a.query(pts_b)
    chamfer = float(d_a2b.mean() + d_b2a.mean())
    hausdorff = float(max(np.percentile(d_a2b, 99.9), np.percentile(d_b2a, 99.9)))
    print(f"ManifoldPlus vs pre_reconstruction: chamfer_mean={chamfer:.4g} hausdorff_p99.9={hausdorff:.4g}")
