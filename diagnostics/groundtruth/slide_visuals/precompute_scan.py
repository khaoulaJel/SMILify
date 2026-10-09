"""Precompute a display-cleaned crop of the RAW scan mesh around the mandible for the GIF.

The raw scan is genuinely noisy right at the mandible apex (a known failure mode: thin,
occluded tips are the hardest thing for photogrammetry to capture -- this is real scan data,
not a bug). Anchored Laplacian smoothing (boundary vertices held fixed, interior relaxed)
tames it for a clean presentation render without touching the actual landmark/vertex analysis,
which was already computed from the raw mesh in extract_mesh_case.py.
"""
import numpy as np
import trimesh
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

OUT = "/w0/tmp/nao48500/login23-1_353989/claude-54434/-rwthfs-rz-cluster-home-nao48500-SMILify/9f42922d-7ac6-494a-aafb-70b3f65325c7/scratchpad/mesh_case"
d = np.load(f"{OUT}/case.npz")
V = d["fitted_verts"]
F = d["faces"].astype(np.int64)
if F.ndim == 3:
    F = F[0]
T = d["target_verts"]
FT = d["target_faces"].astype(np.int64)
human_pt = np.array(d["human_landmark_fitter_frame"])
old_pt = V[int(d["old_idx"])]
ZOOM_CENTER = np.stack([human_pt, old_pt]).mean(0)
CROP_RADIUS = 0.22


def crop(verts, faces, center, radius):
    mask = np.linalg.norm(verts - center, axis=1) < radius
    tri = faces
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], 0)
    valid = mask[e[:, 0]] & mask[e[:, 1]]
    e = e[valid]
    n = len(verts)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)); A = A + A.T
    _, labels = connected_components(A, directed=False)
    idxs = np.nonzero(mask)[0]
    seed = idxs[np.argmin(np.linalg.norm(verts[idxs] - center, axis=1))]
    keep = labels == labels[seed]
    return faces[keep[faces].all(1)]


FT_zoom = crop(T, FT, ZOOM_CENTER, CROP_RADIUS)
uniq = np.unique(FT_zoom)
remap = -np.ones(len(T), dtype=np.int64)
remap[uniq] = np.arange(len(uniq))
sub_V = T[uniq].copy()
sub_F = remap[FT_zoom]

mesh = trimesh.Trimesh(vertices=sub_V, faces=sub_F, process=False)
edge_face_count = {}
for e in map(tuple, mesh.edges_sorted):
    edge_face_count[e] = edge_face_count.get(e, 0) + 1
boundary_verts = {v for e, c in edge_face_count.items() if c == 1 for v in e}
boundary_mask = np.zeros(len(sub_V), bool)
boundary_mask[list(boundary_verts)] = True

adj = mesh.vertex_neighbors
Vs = sub_V.copy()
lam = 0.5
for _ in range(25):
    newV = Vs.copy()
    for i in range(len(Vs)):
        if boundary_mask[i]:
            continue
        nbrs = adj[i]
        if not nbrs:
            continue
        newV[i] = Vs[i] + lam * (Vs[nbrs].mean(0) - Vs[i])
    Vs = newV

np.savez(f"{OUT}/scan_smooth_final.npz", verts=Vs, faces=mesh.faces)
print("saved", f"{OUT}/scan_smooth_final.npz", Vs.shape, mesh.faces.shape)
