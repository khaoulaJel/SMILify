"""
Blender-headless OLD-vs-NEW offset gap-preservation comparison for one specimen, run via:
    blender --background --python-exit-code 1 --python gapcheck_blender2.py -- <name> <alpha> <prerecon_obj> <out_json>

Runs the CGAL call TWICE at the SAME alpha - once at the OLD offset ratio (alpha/30), once at the
NEW one (alpha*0.4) - using target_a/target_b/pre_gap derived ONCE, consistently, from the SAME
reimported mesh (via the real base._min_self_approach_gap, not a standalone reimplementation).
This isolates the offset's actual effect from an orthogonal confound: re-deriving target_a/target_b
via an OBJ text-export/reimport round trip introduces floating-point noise that can, on its own,
flip a borderline specimen's verdict regardless of offset - confirmed by testing CGAL's own
determinism directly (same relative_alpha/relative_offset run twice = byte-identical .off, so CGAL
itself is not the source of any discrepancy against the original log). Comparing OLD vs NEW within
this one consistent pipeline is the only way to isolate the real offset effect from that noise.
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
OLD_OFFSET_FACTOR = 1.0 / 30.0
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

print(f"GAPCHECK: {name} alpha={alpha:.6g} pre_gap={pre_gap:.6g} diag_length={diag_length:.6g}")

import trimesh
from scipy.spatial import cKDTree


def run_one(offset_factor, label):
    offset = alpha * offset_factor
    relative_alpha = diag_length / alpha
    relative_offset = diag_length / offset
    work_dir = os.path.join(os.path.dirname(out_json), "work", name, label)
    os.makedirs(work_dir, exist_ok=True)
    input_base = os.path.splitext(os.path.basename(prerecon_path))[0]
    predicted_path = os.path.join(work_dir, f"{input_base}_{int(relative_alpha)}_{int(relative_offset)}.off")
    out_path = os.path.join(work_dir, f"{label}.off")
    cmd = (
        f"ulimit -v {MEM_CAP_KB}; cd {work_dir} && {ALPHA_WRAP_BINARY} "
        f"{prerecon_path} {relative_alpha} {relative_offset}"
    )
    r = {"offset": offset, "relative_alpha": relative_alpha, "relative_offset": relative_offset}
    try:
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        r["verdict"] = "ERROR"
        r["error"] = f"CGAL timed out after {TIMEOUT}s"
        return r
    if proc.returncode != 0 or not os.path.isfile(predicted_path):
        r["verdict"] = "ERROR"
        r["error"] = f"CGAL failed exit={proc.returncode}: {proc.stderr[-300:]}"
        return r
    os.replace(predicted_path, out_path)

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
    r.update({"gap_after": gap_after, "verdict": verdict,
              "output_vertices": len(out_mesh.vertices), "output_faces": len(out_mesh.faces)})
    return r


old_result = run_one(OLD_OFFSET_FACTOR, "old")
new_result = run_one(NEW_OFFSET_FACTOR, "new")

result = {
    "specimen": name, "alpha": alpha, "pre_gap": float(pre_gap), "diag_length": diag_length,
    "old": old_result, "new": new_result,
    "flipped": old_result.get("verdict") != new_result.get("verdict"),
}
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"GAPCHECK RESULT: {result}")
