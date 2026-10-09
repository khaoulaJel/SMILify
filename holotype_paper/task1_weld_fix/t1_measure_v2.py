"""Task 1 measurements, v2: each version's output vs the part of the scan the pipeline actually receives.

Why v2 (2026-10-09, FINDINGS.md): raw AntScan STLs hold thousands of loose pieces, among them flat
mounting plates up to 17-27% of the scan's area (e.g. Cephalotes minutus, Strumigenys appretiata).
Every version discards all of them in its FIRST step (find_largest_component, unchanged by the fix).
v1 scored against the whole raw scan, so (a) those pieces counted as "lost" for every version and
(b) they shifted the raw PCA frame, and single-pass ICP then failed for all versions on some scans
(Cephalotes minutus p99 5.5% of the diagonal) or for one version only (Discothyrea V1: residual 0.53%
in a flipped frame while the near-identical V2 output aligns at 0.16%).

  reference  largest piece of the raw scan by vertex count, vertex/edge connectivity (the definition
             find_largest_component uses); outer = visible from outside the reference (26 rays)
  dropped    raw OUTER surface (whole scan) farther than tau from the reference: what the first step
             removes; the same for every version, reported once
  align      pass 1: 24 principal-axis starts per output; pass 2: every output re-started from every
             other output's pass-1 transform; best = lowest median output->reference distance over
             20,000 points
  fidelity / coverage / pieces as in t1_measure.py, against the reference
Writes out/measure_v2_<specimen>.json. Usage: python t1_measure_v2.py <specimen> [<specimen> ...]
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t1_measure import OUT, RAW, TAU, WORKER, outer_samples, pca_frame, pieces  # noqa: E402

VERSIONS = (("V0", OUT), ("V1", OUT), ("V2", OUT), ("W2024", WORKER))


def largest_piece(raw):
    """Largest connected piece by vertex count (vertex/edge connectivity), as the pipeline keeps it."""
    e = raw.edges_unique
    n = len(raw.vertices)
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    keep = np.argmax(np.bincount(lab))
    fmask = lab[raw.faces[:, 0]] == keep
    ref = raw.submesh([np.where(fmask)[0]], append=True)
    return ref, float(fmask.mean())


def icp(pts, target, init, iters):
    M, _, _ = trimesh.registration.icp(pts, target, initial=init, threshold=1e-6, max_iterations=iters,
                                       reflection=False, scale=False)
    return M


def score(M, pts, tree):
    return float(np.median(tree.query(trimesh.transform_points(pts, M))[0]))


def pca_starts(out_pts, ref_pts):
    co, Fo = pca_frame(out_pts)
    cr, Fr = pca_frame(ref_pts)
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R0 = Fr @ (np.diag(signs) @ np.eye(3)[list(perm)]) @ Fo.T
            if np.linalg.det(R0) < 0:
                continue
            T = np.eye(4)
            T[:3, :3] = R0
            T[:3, 3] = cr - R0 @ co
            yield T


def main(s):
    t0 = time.time()
    rng = np.random.default_rng(0)
    raw = trimesh.load(RAW.format(s=s), process=True)
    ref, ref_face_share = largest_piece(raw)
    rp_full, outer_full = outer_samples(raw)
    rp, outer = outer_samples(ref)
    ro = rp[outer]
    diag = float(np.linalg.norm(ro.max(0) - ro.min(0)))
    ref_outer_tree = cKDTree(ro)
    dense, _ = trimesh.sample.sample_surface(ref, 1_000_000, seed=2)
    dense_tree = cKDTree(dense)
    dropped = dense_tree.query(rp_full[outer_full])[0] > TAU * diag
    res = dict(specimen=s, raw_faces=int(len(raw.faces)), ref_faces=int(len(ref.faces)),
               ref_face_share=ref_face_share, outer_diag=diag, tau=TAU,
               dropped_at_input_outer_pct=float(100 * dropped.mean()))
    outs = {}
    for v, path in VERSIONS:
        p = path.format(v=v, s=s)
        if os.path.exists(p):
            m = trimesh.load(p, process=False)
            op, _ = trimesh.sample.sample_surface(m, 200_000, seed=3)
            outs[v] = (m, op, op[rng.choice(len(op), 5000, replace=False)],
                       op[rng.choice(len(op), 20000, replace=False)])
        else:
            res[v] = dict(missing=True)
    best = {}
    for v, (_, _, sub, sc) in outs.items():
        cands = [icp(sub, ro, T, 40) for T in pca_starts(sub, ro)]
        best[v] = min(((score(M, sc, ref_outer_tree), M, "pca") for M in cands), key=lambda x: x[0])
    for v, (_, _, sub, sc) in outs.items():
        for w in outs:
            if w != v:
                M = icp(sub, ro, best[w][1], 60)
                c = score(M, sc, ref_outer_tree)
                if c < best[v][0]:
                    best[v] = (c, M, f"seed_from_{w}")
    for v, (m, op, _, _) in outs.items():
        resid, M, src = best[v]
        op_ref = trimesh.transform_points(op, M)
        dfid = dense_tree.query(op_ref)[0]
        dcov = cKDTree(op_ref).query(ro)[0]
        r = dict(align_resid_pct=100 * resid / diag, align_source=src,
                 fid_p50_pct=100 * float(np.percentile(dfid, 50)) / diag,
                 fid_p99_pct=100 * float(np.percentile(dfid, 99)) / diag,
                 fid_max_pct=100 * float(dfid.max()) / diag,
                 uncovered_outer_pct=100 * float((dcov > TAU * diag).mean()),
                 transform=M.tolist())
        r.update(pieces(m))
        res[v] = r
        print(f"[measure_v2] {s} {v}: align {r['align_resid_pct']:.3f}% ({src})  fid p99 {r['fid_p99_pct']:.3f}%  "
              f"uncovered {r['uncovered_outer_pct']:.2f}%", flush=True)
    res["seconds"] = time.time() - t0
    print(f"[measure_v2] {s}: reference = {100 * ref_face_share:.1f}% of raw faces; dropped at input "
          f"{res['dropped_at_input_outer_pct']:.2f}% of the raw outer surface", flush=True)
    json.dump(res, open(os.path.join(HERE, "out", f"measure_v2_{s}.json"), "w"), indent=1)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
