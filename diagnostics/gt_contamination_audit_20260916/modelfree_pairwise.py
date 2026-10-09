"""Arm A -- landmark correspondence measured specimen-to-specimen on the RAW SCANS.

No parametric model, no fit, anywhere in this script.

Estimand
--------
For each ordered pair of specimens (i -> j) and each landmark k annotated on BOTH:
    e_ijk = || T_ij( x_ik ) - x_jk ||  /  WL_j   x 100 %
where T_ij is the best similarity transform (rotation + uniform scale + translation)
between the raw scan surfaces of i and j, estimated from GEOMETRY ALONE (multi-start
ICP-with-scale over PCA-axis initialisations), and x is the expert annotation in the
corrected frame obj = (x, z, -y).

This is the model-free analogue of R1's "how far is the best template vertex from the
annotated anatomy". If e is comparable to R1's ~24 % WL, R1's number is dominated by a
genuine absence of correspondence; if it is far lower, R1 is dominated by fit error.

Three estimators are reported, and they are different estimands -- do not mix them:
  (a) PAIRWISE   : mean over source specimens of e_ijk           (1-vs-1 transfer)
  (b) CONSENSUS  : error of the mean of the 10 transferred points (n-vs-1, noise-averaged;
                   the closest analogue to R1's consensus VERTEX)
  (c) ORACLE     : the same as (a) but with T_ij computed by Umeyama similarity Procrustes
                   ON THE LANDMARKS THEMSELVES (leave-k-out, so k never votes for its own
                   transform). This is the LOWER BOUND achievable by ANY similarity
                   transform, i.e. the part of the residual that no registration algorithm
                   can remove -- the intrinsic non-similarity of the landmark configurations.

Registration quality is reported alongside (bidirectional chamfer, % WL) so a failed
registration is never silently read as absent correspondence.
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

fix = lambda v: np.array([v[0], v[2], -v[1]], float)  # noqa: E731  Blender Z-up -> obj frame

NPTS = 6000          # surface samples used for registration
ICP_ITERS = 60
TRIM = 0.85          # trimmed ICP: use the closest 85 % of correspondences each iteration
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ similarity utilities
def umeyama(src, dst, with_scale=True):
    """Least-squares similarity transform mapping src -> dst. Returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = float((d * np.diag(W)).sum() / S.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim(P, s, R, t):
    return (s * (R @ P.T)).T + t


def icp_sim(src, dst_tree, dst_pts, s, R, t, iters=ICP_ITERS):
    """Trimmed ICP with uniform scale. src/dst in their own normalised frames."""
    for _ in range(iters):
        P = apply_sim(src, s, R, t)
        d, idx = dst_tree.query(P, workers=1)
        keep = d <= np.quantile(d, TRIM)
        s, R, t = umeyama(src[keep], dst_pts[idx[keep]])
    P = apply_sim(src, s, R, t)
    d, _ = dst_tree.query(P, workers=1)
    return s, R, t, float(np.mean(d))


def chamfer(A_pts, A_tree, B_pts, B_tree):
    dab, _ = B_tree.query(A_pts, workers=1)
    dba, _ = A_tree.query(B_pts, workers=1)
    return float(0.5 * (dab.mean() + dba.mean()))


# ------------------------------------------------------------------ load specimens
spec = {}
for sid in SIDS:
    m = trimesh.load(os.path.join(MESH_DIR, f"{sid}_processed.obj"), process=False)
    o = np.asarray(m.vertices, float)
    diag = float(np.linalg.norm(o.max(0) - o.min(0)))
    d = json.load(open(os.path.join(LM_DIR, f"{sid}_traits.json")))["landmarks"]

    tree_full = cKDTree(o)
    L, off_cor, off_raw = {}, {}, {}
    for k, v in d.items():
        pc = fix(v["original"])
        off_cor[k] = float(tree_full.query(pc)[0]) / diag * 100.0
        off_raw[k] = float(tree_full.query(np.array(v["original"], float))[0]) / diag * 100.0
        L[k] = pc
    med_cor = float(np.median(list(off_cor.values())))
    # THE DISCRIMINATING ASSERT (uncorrected is ~2.27 % of the diagonal)
    assert med_cor < 0.5, f"{sid}: corrected landmarks {med_cor:.3f}% diag off-surface"

    WL = float(np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"]))

    # GEOMETRY-ONLY normalisation: centroid + RMS radius. Deliberately NOT WL-based,
    # so no landmark information leaks into the registration.
    P = np.asarray(trimesh.sample.sample_surface(m, NPTS, seed=0)[0], float)
    c = P.mean(0)
    r = float(np.sqrt(((P - c) ** 2).sum(1).mean()))
    Pn = (P - c) / r
    Ln = {k: (p - c) / r for k, p in L.items()}
    WLn = WL / r                                   # Weber's length in the normalised frame

    # principal axes, sign-disambiguated by the 3rd moment so the frame is deterministic
    Cv = np.cov((Pn).T)
    w, Vv = np.linalg.eigh(Cv)
    Vv = Vv[:, np.argsort(w)[::-1]]
    for a in range(3):
        if ((Pn @ Vv[:, a]) ** 3).sum() < 0:
            Vv[:, a] *= -1
    if np.linalg.det(Vv) < 0:
        Vv[:, 2] *= -1

    spec[sid] = dict(P=Pn, tree=cKDTree(Pn), L=Ln, WL=WLn, axes=Vv,
                     med_cor=med_cor, med_raw=float(np.median(list(off_raw.values()))),
                     n_lm=len(L), diag=diag)
    print(f"[lm] {sid:<46} n_lm {len(L):>2}  on-surface corrected {med_cor:6.3f}%  "
          f"raw {spec[sid]['med_raw']:6.3f}%  WL/r {WLn:.4f}")

print(f"\n[lm] {len(spec)} specimens; median on-surface: corrected "
      f"{np.median([a['med_cor'] for a in spec.values()]):.4f}%  raw "
      f"{np.median([a['med_raw'] for a in spec.values()]):.4f}%  (of mesh diagonal)\n")

# ------------------------------------------------------------------ pairwise registration
FLIPS = []
for sx in (1, -1):
    for sy in (1, -1):
        for sz in (1, -1):
            Dm = np.diag([sx, sy, sz]).astype(float)
            if np.linalg.det(Dm) > 0:
                FLIPS.append(Dm)

pairs = list(itertools.combinations(SIDS, 2))
CKPT = os.path.join(HERE, "modelfree_pairwise_ckpt.json")
res = {}
if os.path.exists(CKPT):
    raw = json.load(open(CKPT))
    for k, v in raw.items():
        a, b = k.split("|")
        res[(a, b)] = dict(s=v["s"], R=np.array(v["R"]), t=np.array(v["t"]),
                           chamfer_pct_wl=v["chamfer_pct_wl"])
    print(f"[resume] {len(res)} pairs loaded from checkpoint")

t0 = time.time()
for n, (a, b) in enumerate(pairs):
    if (a, b) in res:
        continue
    A, B = spec[a], spec[b]
    best = None
    for Dm in FLIPS:                              # 4 proper-rotation axis alignments
        R0 = B["axes"] @ Dm @ A["axes"].T
        cand = icp_sim(A["P"], B["tree"], B["P"], 1.0, R0, np.zeros(3), iters=12)
        if best is None or cand[3] < best[3]:
            best = cand
    s, R, t, _ = icp_sim(A["P"], B["tree"], B["P"], best[0], best[1], best[2], iters=ICP_ITERS)
    PA = apply_sim(A["P"], s, R, t)
    ch = chamfer(PA, cKDTree(PA), B["P"], B["tree"])
    res[(a, b)] = dict(s=s, R=R, t=t, chamfer_pct_wl=ch / B["WL"] * 100.0)
    print(f"[reg] {n+1:>3}/{len(pairs)}  {a[:22]:<22} -> {b[:22]:<22}  "
          f"chamfer {ch / B['WL'] * 100:6.2f} %WL  ({time.time()-t0:.0f}s)")
    sys.stdout.flush()
    json.dump({f"{k[0]}|{k[1]}": dict(s=v["s"], R=v["R"].tolist(), t=v["t"].tolist(),
                                      chamfer_pct_wl=v["chamfer_pct_wl"])
              for k, v in res.items()}, open(CKPT, "w"))

# ------------------------------------------------------------------ landmark transfer
rows = []          # one per (src, dst, landmark, method)
for (a, b), r in res.items():
    for src, dst, inv in ((a, b, False), (b, a, True)):
        A, B = spec[src], spec[dst]
        if inv:
            s = 1.0 / r["s"]
            R = r["R"].T
            t = -s * R @ r["t"]
        else:
            s, R, t = r["s"], r["R"], r["t"]
        shared = sorted(set(A["L"]) & set(B["L"]))
        for k in shared:
            p = apply_sim(A["L"][k][None], s, R, t)[0]
            e = float(np.linalg.norm(p - B["L"][k])) / B["WL"] * 100.0
            psurf = B["P"][B["tree"].query(p)[1]]
            e_surf = float(np.linalg.norm(psurf - B["L"][k])) / B["WL"] * 100.0
            rows.append(dict(src=src, dst=dst, lm=k, method="icp", err=e, err_surf=e_surf,
                             chamfer=r["chamfer_pct_wl"], pred=p.tolist()))

        # ORACLE: similarity Procrustes on the landmarks, leave-k-out
        for k in shared:
            others = [q for q in shared if q != k]
            if len(others) < 4:
                continue
            S_ = np.stack([A["L"][q] for q in others])
            D_ = np.stack([B["L"][q] for q in others])
            so, Ro, to = umeyama(S_, D_)
            p = apply_sim(A["L"][k][None], so, Ro, to)[0]
            rows.append(dict(src=src, dst=dst, lm=k, method="oracle",
                             err=float(np.linalg.norm(p - B["L"][k])) / B["WL"] * 100.0,
                             err_surf=None, chamfer=r["chamfer_pct_wl"], pred=p.tolist()))

json.dump(dict(
    built="2026-09-16", n_specimens=len(spec), specimens=SIDS,
    npts=NPTS, icp_iters=ICP_ITERS, trim=TRIM,
    onsurface={k: dict(corrected=v["med_cor"], raw=v["med_raw"]) for k, v in spec.items()},
    chamfer_pct_wl={f"{a}|{b}": v["chamfer_pct_wl"] for (a, b), v in res.items()},
    rows=rows), open(os.path.join(HERE, "modelfree_pairwise.json"), "w"), indent=1)
print(f"\n[out] modelfree_pairwise.json  ({len(rows)} rows, {time.time()-t0:.0f}s)")
