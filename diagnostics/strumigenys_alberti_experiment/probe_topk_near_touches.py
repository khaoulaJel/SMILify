"""_min_self_approach_gap only returns the SINGLE globally-tightest self-approach gap. For
Strumigenys_alberti that turned out to be a parallel-normal (normal_dot=0.9995) artifact point,
not the leg-gaster fusion the user observed visually - meaning the pipeline's bisection was
guarding the wrong landmark entirely. This finds the top-K tightest per-vertex self-approach gaps
(same hop-exclusion algorithm, just keeping the K best instead of only the global minimum),
computes normal_dot for each, and reports enough spatial context (distance from mesh centroid,
local vertex density) to identify which candidates are genuine near-touches and which region of
the ant they're likely on.

Usage: blender --background --python probe_topk_near_touches.py
"""
import sys
sys.path.insert(0, "/home/nao48500/SMILify")
sys.path.insert(0, "/home/nao48500/SMILify/custom_processing")

import bpy
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import prepare_antscan_data_for_mesh_fitting_manifold as base
import prepare_antscan_data_for_mesh_fitting_alphawrap as aw

PATH = "/home/nao48500/SMILify/diagnostics/eyeball_groundtruth_20260817/06_FIDFAIL_Strumigenys_alberti/pre_reconstruction.obj"
K_TOP = 15
HOPS = 4
K_NEIGHBORS = 100

bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.wm.obj_import(filepath=PATH)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=PATH)
obj = bpy.context.selected_objects[0]
verts, faces = base._triangulated_verts_faces(obj)

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
A = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
R = A.copy()
cur = A.copy()
for _ in range(HOPS - 1):
    cur = cur.dot(A)
    R = R + cur
R = (R > 0)

tree = cKDTree(verts)
candidates = []  # (gap, i, j)
for i in range(n):
    dists, idxs = tree.query(verts[i], k=K_NEIGHBORS)
    excl = set(R.getrow(i).indices.tolist())
    excl.add(i)
    for d, j in zip(dists, idxs):
        if j not in excl:
            candidates.append((float(d), i, j))
            break

candidates.sort(key=lambda c: c[0])

centroid = verts.mean(axis=0)


def vertex_normal(vi):
    face_mask = np.any(faces == vi, axis=1)
    tris = verts[faces[face_mask]]
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(nrm, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    nrm = nrm / norms
    avg = nrm.mean(axis=0)
    avg_norm = np.linalg.norm(avg)
    return avg / avg_norm if avg_norm > 1e-9 else avg


seen_pairs = set()
n_reported = 0
for gap, i, j in candidates:
    key = (min(i, j), max(i, j))
    if key in seen_pairs:
        continue
    seen_pairs.add(key)
    na = vertex_normal(i)
    nb = vertex_normal(j)
    dot = float(np.dot(na, nb))
    pa, pb = verts[i], verts[j]
    mid = (pa + pb) / 2
    dist_from_centroid = float(np.linalg.norm(mid - centroid))
    print(f"TOPK_CANDIDATE gap={gap:.4g} normal_dot={dot:+.4f} "
          f"a={list(pa)} b={list(pb)} dist_from_centroid={dist_from_centroid:.4g} "
          f"interp={'GENUINE' if dot < -0.3 else ('ARTIFACT' if dot > 0.3 else 'AMBIG')}")
    n_reported += 1
    if n_reported >= K_TOP:
        break

print(f"TOPK_CENTROID centroid={list(centroid)}")
bmin, bmax = verts.min(axis=0), verts.max(axis=0)
print(f"TOPK_BBOX min={list(bmin)} max={list(bmax)}")
