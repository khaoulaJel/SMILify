"""
Blender-headless gap-preservation re-check for one specimen, run via:
    blender --background --python-exit-code 1 --python gapcheck_blender.py -- <name> <alpha> <prerecon_obj> <out_json>

Uses the REAL base._triangulated_verts_faces / base._min_self_approach_gap (same function, same
environment as the original bisection run) to get target_a/target_b/pre_gap - avoids any risk of
a standalone numpy reimplementation picking a different (but equally-minimal) point pair on a
bilaterally-symmetric specimen than Blender's original in-memory vertex ordering did. Everything
after landmark detection (the actual CGAL call + gap-after query) matches
_bisect_alpha_wrap_params._run() exactly and does NOT need bpy, so it's done with trimesh/scipy
directly, same as the production bisection code.
"""
import sys
import os
import json
import subprocess

import bpy
import numpy as np

sys.path.insert(0, "/rwthfs/rz/cluster/home/nao48500/SMILify/custom_processing")
import prepare_antscan_data_for_mesh_fitting_manifold as base

ALPHA_WRAP_BINARY = "/rwthfs/rz/cluster/home/nao48500/SMILify/custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap"
NEW_OFFSET_FACTOR = 0.4
TIMEOUT = 300
MEM_CAP_KB = 10_485_760

argv = sys.argv[sys.argv.index("--") + 1:]
name, alpha_str, prerecon_path, out_json = argv
alpha = float(alpha_str)

bpy.ops.object.select_all(action="DESELECT")
try:
    bpy.ops.wm.obj_import(filepath=prerecon_path)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=prerecon_path)
obj = bpy.context.selected_objects[0]

verts, faces = base._triangulated_verts_faces(obj)
pre_gap, target_a, target_b = base._min_self_approach_gap(verts, faces, return_pair=True)
diag_length = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))

print(f"GAPCHECK: {name} alpha={alpha:.6g} pre_gap={pre_gap:.6g} target_a={list(target_a)} "
      f"target_b={list(target_b)} diag_length={diag_length:.6g}")

offset = alpha * NEW_OFFSET_FACTOR
relative_alpha = diag_length / alpha
relative_offset = diag_length / offset

work_dir = os.path.join(os.path.dirname(out_json), "work", name)
os.makedirs(work_dir, exist_ok=True)
input_base = os.path.splitext(os.path.basename(prerecon_path))[0]
predicted_path = os.path.join(work_dir, f"{input_base}_{int(relative_alpha)}_{int(relative_offset)}.off")
out_path = os.path.join(work_dir, f"{name}_newoffset.off")

cmd = (
    f"ulimit -v {MEM_CAP_KB}; cd {work_dir} && {ALPHA_WRAP_BINARY} "
    f"{prerecon_path} {relative_alpha} {relative_offset}"
)
result = {"specimen": name, "alpha": alpha, "pre_gap": float(pre_gap), "offset": offset,
          "relative_alpha": relative_alpha, "relative_offset": relative_offset}
try:
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=TIMEOUT)
except subprocess.TimeoutExpired:
    result["verdict"] = "ERROR"
    result["error"] = f"CGAL timed out after {TIMEOUT}s"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    sys.exit(0)

if proc.returncode != 0 or not os.path.isfile(predicted_path):
    result["verdict"] = "ERROR"
    result["error"] = f"CGAL failed exit={proc.returncode}: {proc.stderr[-300:]}"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    sys.exit(0)
os.replace(predicted_path, out_path)

import trimesh
from scipy.spatial import cKDTree

out_mesh = trimesh.load(out_path, process=False)
tree = cKDTree(out_mesh.vertices)
_, ia = tree.query(target_a)
_, ib = tree.query(target_b)
gap_after = float(np.linalg.norm(out_mesh.vertices[ia] - out_mesh.vertices[ib]))
if ia == ib:
    verdict = "fused"
elif gap_after < pre_gap * 0.3:
    verdict = "suspect"
else:
    verdict = "preserved"

result.update({
    "gap_after": gap_after, "verdict": verdict,
    "output_vertices": len(out_mesh.vertices), "output_faces": len(out_mesh.faces),
})
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"GAPCHECK RESULT: {result}")
