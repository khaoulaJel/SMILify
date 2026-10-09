#!/usr/bin/env python3
"""
Standalone (no Blender/bpy) gap-preservation re-check for the 28 specimens that already
succeeded with alpha_wrap under the OLD offset ratio (offset=alpha/30). Re-runs ONLY the CGAL
triangle_soup_wrap call at each specimen's ALREADY-KNOWN, frozen alpha (no re-bisection) with the
NEW offset ratio (offset=alpha*0.4), then checks whether the gap-preservation verdict flips.

_min_self_approach_gap is copied verbatim from
prepare_antscan_data_for_mesh_fitting_manifold.py (that file imports bpy at module level, so it
can't be imported directly outside Blender) - pure numpy/scipy, no bpy dependency, so this
reproduces the exact same target_a/target_b/pre_gap deterministically from the same frozen
PRE_RECONSTRUCTION.obj diagnostic snapshot process_stl already saved for each specimen.
"""
import json
import os
import subprocess
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

ALPHA_WRAP_BINARY = "/rwthfs/rz/cluster/home/nao48500/SMILify/custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap"
NEW_OFFSET_FACTOR = 0.4  # offset = alpha * 0.4 (was alpha/30 ~= alpha*0.0333)
TIMEOUT = 300
MEM_CAP_KB = 10_485_760


def _min_self_approach_gap(verts, faces, hops=4, k=100, return_pair=False):
    referenced = np.unique(faces.ravel())
    if len(referenced) < len(verts):
        remap = -np.ones(len(verts), dtype=np.int64)
        remap[referenced] = np.arange(len(referenced))
        verts = verts[referenced]
        faces = remap[faces]

    n = len(verts)
    edges = set()
    for f in faces:
        a, b, c = f
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    edges = np.array(list(edges))

    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row))
    A = csr_matrix((data, (row, col)), shape=(n, n))

    R = A.copy()
    cur = A.copy()
    for _ in range(hops - 1):
        cur = cur.dot(A)
        R = R + cur
    R = (R > 0)

    tree = cKDTree(verts)
    min_gap = float("inf")
    best_pair = None
    for i in range(n):
        dists, idxs = tree.query(verts[i], k=k)
        excl = set(R.getrow(i).indices.tolist())
        excl.add(i)
        for d, j in zip(dists, idxs):
            if j not in excl:
                if d < min_gap:
                    min_gap = float(d)
                    best_pair = (verts[i].copy(), verts[j].copy())
                break
    if not return_pair:
        return min_gap
    if best_pair is None:
        return min_gap, None, None
    return min_gap, best_pair[0], best_pair[1]


def check_one(name, alpha, prerecon_path, work_dir):
    mesh = trimesh.load(prerecon_path, process=False)
    verts = mesh.vertices
    faces = mesh.faces

    pre_gap, target_a, target_b = _min_self_approach_gap(verts, faces, return_pair=True)
    diag_length = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))

    offset = alpha * NEW_OFFSET_FACTOR
    relative_alpha = diag_length / alpha
    relative_offset = diag_length / offset

    os.makedirs(work_dir, exist_ok=True)
    input_base = os.path.splitext(os.path.basename(prerecon_path))[0]
    predicted_path = os.path.join(
        work_dir, f"{input_base}_{int(relative_alpha)}_{int(relative_offset)}.off"
    )
    out_path = os.path.join(work_dir, f"{name}_newoffset.off")

    cmd = (
        f"ulimit -v {MEM_CAP_KB}; cd {work_dir} && {ALPHA_WRAP_BINARY} "
        f"{prerecon_path} {relative_alpha} {relative_offset}"
    )
    try:
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"specimen": name, "error": f"CGAL timed out after {TIMEOUT}s", "verdict": "ERROR"}

    if result.returncode != 0 or not os.path.isfile(predicted_path):
        return {"specimen": name, "error": f"CGAL failed exit={result.returncode}: {result.stderr[-300:]}",
                "verdict": "ERROR"}
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

    return {
        "specimen": name, "alpha": alpha, "pre_gap": float(pre_gap), "offset": offset,
        "relative_alpha": relative_alpha, "relative_offset": relative_offset,
        "gap_after": gap_after, "verdict": verdict,
        "output_vertices": len(out_mesh.vertices), "output_faces": len(out_mesh.faces),
    }


def main():
    gapdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(gapdir, "manifest.json")) as f:
        manifest = json.load(f)

    work_dir = os.path.join(gapdir, "work")
    results = []
    for i, (name, alpha, prerecon_path) in enumerate(manifest):
        print(f"[{i+1}/{len(manifest)}] {name} (alpha={alpha:.6g})...", flush=True)
        r = check_one(name, alpha, prerecon_path, os.path.join(work_dir, name))
        print(f"  -> {r}", flush=True)
        results.append(r)

    with open(os.path.join(gapdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    flipped = [r for r in results if r.get("verdict") not in ("preserved",)]
    print(f"\n{'='*60}\n{len(results)-len(flipped)}/{len(results)} still preserved.")
    if flipped:
        print(f"{len(flipped)} FLIPPED or ERRORED - flagged specimens:")
        for r in flipped:
            print(f"  {r['specimen']}: verdict={r.get('verdict')} error={r.get('error')}")
    else:
        print("No flips. All 28 confirmed preserved under the new offset ratio.")


if __name__ == "__main__":
    main()
