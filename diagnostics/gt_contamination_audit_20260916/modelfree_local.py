"""Arm A2 + diagnostics -- model-free landmark transfer, with the articulation removed.

Arm A (modelfree_pairwise.py) registered WHOLE scans with one similarity transform. That
model is far weaker than the production fit, which has articulated pose, a shape space and
a free-form deform. Its 162 % WL is therefore NOT evidence about correspondence -- it is
evidence that a single rigid+scale map cannot relate two ant species. This script:

  D1  DIAGNOSIS: is the global ICP at a wrong minimum, or is a global similarity simply
      inadequate? Compare the chamfer under the ICP transform with the chamfer under the
      transform obtained from the ANNOTATIONS (landmark Procrustes). If the annotation-
      driven transform is no better, ICP converged fine and the model is the limit.

  A2  HEAD-LOCAL geometric registration: crop both scans to the head (region defined by the
      head landmarks with the TARGET LANDMARK EXCLUDED, so it never sees its own answer),
      multi-start ICP-with-scale head-to-head, transfer the held-out landmark. This gives
      the articulated freedom the whole-body arm lacked, still with no model and with no
      test-time use of the target annotation.

  O   ORACLE variants -- the best ANY transform of that class could do, computed from the
      annotations directly (leave-the-target-landmark-out throughout):
        O-sim    similarity (rot+uniform scale+transl) on all other landmarks
        O-head   similarity on the other HEAD landmarks only  (articulation removed)
        O-aff    full affine on all other landmarks           (crude shape space)
      plus CONSENSUS (average the 10 transferred points, the analogue of R1's consensus
      vertex) and a NULL (predict the landmark at the centroid of the aligned configuration)
      so the numbers can be judged against the scale of the landmark cloud itself.
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
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LM_DIR = os.path.join(REPO, "annotation", "landmarks")
MESH_DIR = "/hpcwork/nao48500/worker_alt_data"
SIDS = sorted(json.load(open(os.path.join(
    REPO, "diagnostics/joint_alignment_benchmark/data/gt_joints_fitframe.json"))).keys())

fix = lambda v: np.array([v[0], v[2], -v[1]], float)  # noqa: E731
HEAD = ["clypeal_ant_mid", "cephalic_post_mid", "head_width_l", "head_width_r",
        "antennal_insertion_l", "antennal_insertion_r", "mandibular_apex_l",
        "mandibular_apex_r", "scape_apex_l", "scape_apex_r"]
NPTS = 40000          # dense body sample; the head crop keeps a few thousand of them
ICP_ITERS = 60
TRIM = 0.85
rng = np.random.default_rng(0)


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = float((d * np.diag(W)).sum() / S.var(0).sum()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def affine(src, dst):
    """Least-squares affine y = A x + b (no reflection guard: this is an upper bound)."""
    X = np.hstack([src, np.ones((len(src), 1))])
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)
    return M


def apply_sim(P, s, R, t):
    return (s * (R @ P.T)).T + t


def icp_sim(src, dst_tree, dst_pts, s, R, t, iters):
    for _ in range(iters):
        P = apply_sim(src, s, R, t)
        d, idx = dst_tree.query(P, workers=1)
        keep = d <= np.quantile(d, TRIM)
        s, R, t = umeyama(src[keep], dst_pts[idx[keep]])
    P = apply_sim(src, s, R, t)
    d, _ = dst_tree.query(P, workers=1)
    return s, R, t, float(np.mean(d))


def bidir_chamfer(PA, B_pts, B_tree):
    dab, _ = B_tree.query(PA, workers=1)
    dba, _ = cKDTree(PA).query(B_pts, workers=1)
    return float(0.5 * (dab.mean() + dba.mean()))


def pca_axes(P):
    w, V = np.linalg.eigh(np.cov(P.T))
    V = V[:, np.argsort(w)[::-1]]
    for a in range(3):
        if (((P - P.mean(0)) @ V[:, a]) ** 3).sum() < 0:
            V[:, a] *= -1
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return V


def rot_about(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


FLIPS = [np.diag(v).astype(float) for v in itertools.product((1, -1), repeat=3)
         if np.linalg.det(np.diag(v)) > 0]

# ------------------------------------------------------------------ load
spec = {}
for sid in SIDS:
    m = trimesh.load(os.path.join(MESH_DIR, f"{sid}_processed.obj"), process=False)
    o = np.asarray(m.vertices, float)
    diag = float(np.linalg.norm(o.max(0) - o.min(0)))
    d = json.load(open(os.path.join(LM_DIR, f"{sid}_traits.json")))["landmarks"]
    tree_full = cKDTree(o)
    L = {k: fix(v["original"]) for k, v in d.items()}
    med_cor = float(np.median([tree_full.query(p)[0] for p in L.values()])) / diag * 100.0
    assert med_cor < 0.5, f"{sid}: corrected landmarks {med_cor:.3f}% diag off-surface"
    WL = float(np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"]))
    P = np.asarray(trimesh.sample.sample_surface(m, NPTS, seed=0)[0], float)
    c = P.mean(0)
    r = float(np.sqrt(((P - c) ** 2).sum(1).mean()))
    Pn = (P - c) / r
    spec[sid] = dict(P=Pn, tree=cKDTree(Pn), L={k: (p - c) / r for k, p in L.items()},
                     WL=WL / r, med_cor=med_cor)
    print(f"[lm] {sid:<46} on-surface {med_cor:.3f}%  WL/r {WL/r:.4f}")
sys.stdout.flush()

pairs = list(itertools.combinations(SIDS, 2))
rows, diag_rows = [], []
t0 = time.time()

# ------------------------------------------------------------------ D1 : global ICP diagnosis
prev = json.load(open(os.path.join(HERE, "modelfree_pairwise.json")))
icp_ch = prev["chamfer_pct_wl"]
for a, b in pairs:
    A, B = spec[a], spec[b]
    sh = sorted(set(A["L"]) & set(B["L"]))
    s, R, t = umeyama(np.stack([A["L"][k] for k in sh]), np.stack([B["L"][k] for k in sh]))
    # 6000-point subsample so this chamfer is directly comparable with Arm A's (same density)
    Pa, Pb = A["P"][:6000], B["P"][:6000]
    ch_lm = bidir_chamfer(apply_sim(Pa, s, R, t), Pb, cKDTree(Pb)) / B["WL"] * 100.0
    diag_rows.append(dict(a=a, b=b, chamfer_icp=icp_ch[f"{a}|{b}"], chamfer_landmark=ch_lm))
med_icp = np.median([r["chamfer_icp"] for r in diag_rows])
med_lm = np.median([r["chamfer_landmark"] for r in diag_rows])
print(f"\n[D1] median bidirectional chamfer, %WL:  ICP {med_icp:.2f}   "
      f"annotation-driven similarity {med_lm:.2f}")
print(f"[D1] ICP better on {sum(r['chamfer_icp'] < r['chamfer_landmark'] for r in diag_rows)}"
      f"/{len(diag_rows)} pairs\n")
sys.stdout.flush()

# ------------------------------------------------------------------ A2 : head-local ICP
for n, (a, b) in enumerate(pairs):
    A, B = spec[a], spec[b]
    shared = sorted((set(A["L"]) & set(B["L"])) & set(HEAD))
    for k in shared:
        others = [q for q in shared if q != k]
        if len(others) < 4:
            continue
        crops, trees = {}, {}
        ok = True
        for sid_ in (a, b):
            S_ = spec[sid_]
            X = np.stack([S_["L"][q] for q in others])
            cH = X.mean(0)
            rad = 1.35 * float(np.linalg.norm(X - cH, axis=1).max())
            sel = np.linalg.norm(S_["P"] - cH, axis=1) < rad
            if sel.sum() < 300:
                ok = False
                break
            crops[sid_] = S_["P"][sel] - cH          # crop, centred on its own head
            trees[sid_] = cKDTree(crops[sid_])
        if not ok:
            continue
        for src, dst in ((a, b), (b, a)):
            Ps, Pd, Td = crops[src], crops[dst], trees[dst]
            Vs, Vd = pca_axes(Ps), pca_axes(Pd)
            best = None
            for Dm in FLIPS:
                for ang in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
                    R0 = Vd @ rot_about(np.array([1.0, 0, 0]), ang) @ Dm @ Vs.T
                    cand = icp_sim(Ps, Td, Pd, 1.0, R0, np.zeros(3), iters=10)
                    if best is None or cand[3] < best[3]:
                        best = cand
            s, R, t, _ = icp_sim(Ps, Td, Pd, best[0], best[1], best[2], iters=ICP_ITERS)
            # the crop offsets must be undone in the same frame the landmarks live in
            cs = np.stack([spec[src]["L"][q] for q in others]).mean(0)
            cd = np.stack([spec[dst]["L"][q] for q in others]).mean(0)
            p = apply_sim((spec[src]["L"][k] - cs)[None], s, R, t)[0] + cd
            ch = bidir_chamfer(apply_sim(Ps, s, R, t), Pd, Td) / spec[dst]["WL"] * 100.0
            rows.append(dict(src=src, dst=dst, lm=k, method="head_icp",
                             err=float(np.linalg.norm(p - spec[dst]["L"][k]))
                             / spec[dst]["WL"] * 100.0,
                             chamfer=ch, pred=(p).tolist()))
    print(f"[A2] {n+1:>3}/{len(pairs)}  {a[:20]:<20} <-> {b[:20]:<20}  ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

# ------------------------------------------------------------------ O : oracle transforms
for a, b in pairs:
    for src, dst in ((a, b), (b, a)):
        A, B = spec[src], spec[dst]
        shared = sorted(set(A["L"]) & set(B["L"]))
        for k in shared:
            others = [q for q in shared if q != k]
            S_ = np.stack([A["L"][q] for q in others])
            D_ = np.stack([B["L"][q] for q in others])
            WLd = B["WL"]
            tgt = B["L"][k]

            s, R, t = umeyama(S_, D_)
            p = apply_sim(A["L"][k][None], s, R, t)[0]
            rows.append(dict(src=src, dst=dst, lm=k, method="O-sim",
                             err=float(np.linalg.norm(p - tgt)) / WLd * 100.0,
                             pred=p.tolist()))
            # NULL: predict at the centroid of the aligned configuration
            rows.append(dict(src=src, dst=dst, lm=k, method="NULL-centroid",
                             err=float(np.linalg.norm(D_.mean(0) - tgt)) / WLd * 100.0))

            hs = [q for q in others if q in HEAD]
            if k in HEAD and len(hs) >= 4:
                s, R, t = umeyama(np.stack([A["L"][q] for q in hs]),
                                  np.stack([B["L"][q] for q in hs]))
                p = apply_sim(A["L"][k][None], s, R, t)[0]
                rows.append(dict(src=src, dst=dst, lm=k, method="O-head",
                                 err=float(np.linalg.norm(p - tgt)) / WLd * 100.0,
                                 pred=p.tolist()))
            if len(others) >= 5:
                M = affine(S_, D_)
                p = np.hstack([A["L"][k], 1.0]) @ M
                rows.append(dict(src=src, dst=dst, lm=k, method="O-aff",
                                 err=float(np.linalg.norm(p - tgt)) / WLd * 100.0,
                                 pred=p.tolist()))

json.dump(dict(built="2026-09-16", specimens=SIDS, npts=NPTS,
               diag_global_icp=diag_rows, rows=rows),
          open(os.path.join(HERE, "modelfree_local.json"), "w"), indent=1)
print(f"\n[out] modelfree_local.json  ({len(rows)} rows, {time.time()-t0:.0f}s)")
