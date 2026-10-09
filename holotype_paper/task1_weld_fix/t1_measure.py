"""Task 1 measurements (PROTOCOL.md): each version's output vs the raw AntScan scan it came from.

Per specimen and per output (V0, V1, V2, and the 2024 worker mesh for the provenance check P0):
  align     rigid (rotation + translation, no scale -- the script applies no scale) output -> raw,
            ICP from 24 principal-axis start candidates, best kept; residual reported (alignment QC)
  fidelity  output surface -> raw surface distance, p50 / p99 / max, % of raw outer bbox diagonal
            (geometry NOT present in the input = P1 symptom)
  coverage  fraction of the raw OUTER surface farther than tau = 0.5% diag from the output
            (lost anatomy = P2 symptom). Outer = a raw surface sample from which at least one of 26
            rays escapes the mesh (embree), i.e. visible from outside; internal geometry excluded.
  pieces    connected components; face share of the largest; share of faces in components < 1%
            of the total (debris)
Writes out/measure_<specimen>.json. Usage: python t1_measure.py <specimen> [<specimen> ...]
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("RAW_ROOT", "/hpcwork/nao48500/antscan_data") + "/{s}/{s}.stl"
OUT = "/hpcwork/nao48500/holotype_task1/{v}/{s}/{s}_processed.obj"
WORKER = "/hpcwork/nao48500/worker_ALT/{s}_processed.obj"
TAU = 0.005
RNG = np.random.default_rng(0)


def outer_samples(raw, n=100_000):
    pts, fi = trimesh.sample.sample_surface(raw, n, seed=1)
    dirs = _sphere_dirs(26)
    eps = 1e-4 * raw.scale
    outer = np.zeros(len(pts), bool)
    for d in dirs:
        idx = np.where(~outer)[0]
        if not len(idx):
            break
        origins = pts[idx] + eps * d
        hit = raw.ray.intersects_any(origins, np.repeat(d[None], len(idx), 0))
        outer[idx[~hit]] = True
    return pts, outer


def _sphere_dirs(k):
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def pca_frame(P):
    c = P.mean(0)
    w, v = np.linalg.eigh(np.cov((P - c).T))
    return c, v[:, ::-1]


def align(out_pts, raw_outer_pts):
    """Best rigid transform out -> raw over 24 proper sign/permutation start candidates + ICP."""
    co, Fo = pca_frame(out_pts)
    cr, Fr = pca_frame(raw_outer_pts)
    tree = cKDTree(raw_outer_pts)
    sub = out_pts[RNG.choice(len(out_pts), min(5000, len(out_pts)), replace=False)]
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R0 = Fr @ (np.diag(signs) @ np.eye(3)[list(perm)]) @ Fo.T
            if np.linalg.det(R0) < 0:
                continue
            T = np.eye(4); T[:3, :3] = R0; T[:3, 3] = cr - R0 @ co
            M, _, cost = trimesh.registration.icp(sub, raw_outer_pts, initial=T, threshold=1e-6,
                                                  max_iterations=40, reflection=False, scale=False)
            if best is None or cost < best[1]:
                best = (M, cost)
    M = best[0]
    moved = trimesh.transform_points(out_pts, M)
    d, _ = tree.query(moved)
    return M, float(np.median(d))


def pieces(mesh):
    """Pieces by shared vertices/edges (Blender's 'loose parts'). trimesh.face_adjacency only links
    faces across edges used by exactly two faces, so on non-manifold meshes it shatters one piece into
    thousands (found 2026-10-09: 6,974 'pieces' vs 40 real ones). Also reports non-manifold edges
    (> 2 faces; a signature of wrongly merged surfaces) and boundary edges (1 face; open holes)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    mesh = mesh.copy(); mesh.merge_vertices()
    e = mesh.edges_unique; n = len(mesh.vertices)
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    counts = np.bincount(lab[mesh.faces[:, 0]]); counts = counts[counts > 0]
    small = counts < 0.01 * len(mesh.faces)
    per_edge = np.bincount(mesh.edges_unique_inverse)
    return dict(n_components=int(len(counts)), largest_share=float(counts.max() / len(mesh.faces)),
                debris_face_share=float(counts[small].sum() / len(mesh.faces)),
                n_components_ge1pct=int((~small).sum()), n_faces=int(len(mesh.faces)),
                nonmanifold_edges=int((per_edge > 2).sum()), boundary_edges=int((per_edge == 1).sum()))


def measure(s):
    t0 = time.time()
    raw = trimesh.load(RAW.format(s=s), process=True)
    rp, outer = outer_samples(raw)
    ro = rp[outer]
    diag = float(np.linalg.norm(ro.max(0) - ro.min(0)))
    raw_dense, _ = trimesh.sample.sample_surface(raw, 1_000_000, seed=2)
    raw_tree = cKDTree(raw_dense)
    res = dict(specimen=s, raw_faces=int(len(raw.faces)), outer_fraction=float(outer.mean()),
               outer_diag=diag, tau=TAU)
    for v, path in (("V0", OUT), ("V1", OUT), ("V2", OUT), ("W2024", WORKER)):
        p = path.format(v=v, s=s)
        if not os.path.exists(p):
            res[v] = dict(missing=True)
            continue
        m = trimesh.load(p, process=False)
        op, _ = trimesh.sample.sample_surface(m, 200_000, seed=3)
        M, resid = align(op, ro)
        op_raw = trimesh.transform_points(op, M)
        dfid, _ = raw_tree.query(op_raw)
        dcov, _ = cKDTree(op_raw).query(ro)
        r = dict(align_resid_pct=100 * resid / diag,
                 fid_p50_pct=100 * float(np.percentile(dfid, 50)) / diag,
                 fid_p99_pct=100 * float(np.percentile(dfid, 99)) / diag,
                 fid_max_pct=100 * float(dfid.max()) / diag,
                 uncovered_outer_pct=100 * float((dcov > TAU * diag).mean()),
                 transform=M.tolist())
        r.update(pieces(m))
        res[v] = r
        print(f"[measure] {s} {v}: align {r['align_resid_pct']:.3f}%  fid p99 {r['fid_p99_pct']:.3f}%  "
              f"uncovered {r['uncovered_outer_pct']:.2f}%  comps {r['n_components']} "
              f"(>=1%: {r['n_components_ge1pct']}, debris {100*r['debris_face_share']:.2f}%) "
              f"nonmanifold {r['nonmanifold_edges']} boundary {r['boundary_edges']}", flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "out", f"measure_{s}.json"), "w"), indent=1)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        measure(s)
