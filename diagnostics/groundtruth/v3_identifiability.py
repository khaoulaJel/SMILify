"""V3 -- is the anterior pathology a pose/shape IDENTIFIABILITY degeneracy?

See PREREGISTRATION_V3_identifiability.md. No ground truth is used anywhere in this file.

Core measurement: at each specimen's PRODUCTION parameters, the surface changes reachable by
anterior POSE and by SHAPE, restricted to anterior vertices. If those two subspaces nearly
coincide, a surface objective cannot tell pose from shape there, and the fitter may trade one for
the other at flat chamfer. The gaster is the control region -- a degeneracy everywhere would be
uninformative.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
sys.path.insert(0, HERE)

import pickle  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
import trait_extract as TX  # noqa: E402

ANTERIOR = ["b_h", "ma_r", "ma_l", "an_1_r", "an_2_r", "an_3_r", "an_1_l", "an_2_l", "an_3_l"]
GASTER = ["b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]
BOUNDS = {"HW/HL": (0.40, 2.00), "SL/HL": (0.20, 3.00), "PetL/WL": (0.02, 0.80),
          "ML/HL": (0.10, 2.00), "GL/WL": (0.50, 4.00)}
SEED = 20260907


def principal_angles(A, B):
    """Principal angles (degrees) between the column spans of A and B. Each span is
    orthonormalised first, so the differing units of radians and betas are irrelevant."""
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    jn = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    M = TX.load_model(); lm = TX.load_landmarks(M)

    # region vertex membership, by dominant joint -- the same construction trait_extract uses
    def region_verts(joints):
        return np.where(np.isin(M["dominant"], [M["jnames"].index(j) for j in joints]))[0]
    REG = {"anterior": (ANTERIOR, region_verts(ANTERIOR)),
           "gaster": (GASTER, region_verts(GASTER))}
    for k, (js, vi) in REG.items():
        print(f"{k}: {len(js)} joints, {len(vi)} vertices", flush=True)

    # ---- load the corpus -------------------------------------------------------------
    labels, PAR = [], {k: [] for k in ["betas", "global_rot", "joint_rot", "trans",
                                       "log_beta_scales", "betas_trans", "deform_verts"]}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        labels += [str(x) for x in d["labels"]]
        for k in PAR:
            PAR[k].append(np.asarray(d[k], dtype=np.float32))
    PAR = {k: np.concatenate(v) for k, v in PAR.items()}
    n = len(labels)
    v2 = json.load(open(os.path.join(HERE, "v2_results.json")))
    flag = np.array([l in set(v2["flagged_labels_with"]) for l in labels])
    print(f"{n} fits, {flag.sum()} flagged", flush=True)

    # ---- SEVERITY DISTRIBUTION (no GT) ------------------------------------------------
    # V2 verified traits_Z8.npz reproduces a fresh recompute to 0.000e+00, so read it rather
    # than reloading 757 meshes and redoing every Kabsch alignment.
    tz = np.load(os.path.join(HERE, "traits_Z8.npz"))
    tzl = [str(x) for x in tz["labels"]]
    ordz = [tzl.index(l) for l in labels]
    tw = {k: np.asarray(tz[k])[ordz] for k in tz.files if k != "labels"}
    R = {k: tw[k.split("/")[0]] / np.maximum(tw[k.split("/")[1]], 1e-12) for k in BOUNDS}
    sev = np.zeros(n)
    for k, (lo, hi) in BOUNDS.items():
        over = np.maximum(np.maximum(lo - R[k], R[k] - hi), 0) / (hi - lo)
        sev = np.maximum(sev, over)
    s_f = sev[flag]
    print("\n=== severity of the flagged (max fractional overshoot past the bound) ===")
    for q in (10, 25, 50, 75, 90, 100):
        print(f"  p{q:<3d} {np.percentile(s_f, q):.3f}")
    print(f"  fraction of flagged within 25% of the bound: {(s_f < 0.25).mean():.2f}")
    print(f"  fraction of flagged over 2x the bound width: {(s_f > 2.0).mean():.2f}")

    # ---- SUBSPACE GEOMETRY -------------------------------------------------------------
    sel = np.concatenate([np.where(flag)[0],
                          rng.choice(np.where(~flag)[0], size=int(flag.sum()), replace=False)])
    NB = PAR["betas"].shape[1]
    cols = {k: NB + 3 * len(js) for k, (js, _) in REG.items()}
    B = 1 + 2 * max(cols.values())
    smal = SMAL3DFitter(batch_size=B, device=dev)

    def verts_batch(i, region, eps):
        """One forward pass containing the base fit and all +/- perturbations."""
        js, _ = REG[region]
        ji = [jn.index(j) - 1 for j in js]          # joint_rot excludes the root
        with torch.no_grad():
            for k, v in PAR.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(torch.as_tensor(
                        np.repeat(v[i:i + 1], B, axis=0), device=dev))
            r = 1
            for k in range(NB):
                for sgn in (+1, -1):
                    smal.betas[r, k] += sgn * eps; r += 1
            for j in ji:
                for a in range(3):
                    for sgn in (+1, -1):
                        smal.joint_rot[r, j, a] += sgn * eps; r += 1
            V = smal().detach().cpu().numpy().astype(np.float64)
        return V, NB, len(ji)

    def subspaces(i, region, eps):
        V, nb, nj = verts_batch(i, region, eps)
        vi = REG[region][1]
        D = (V[1::2, vi] - V[2::2, vi]) / (2 * eps)          # central differences
        D = D.reshape(D.shape[0], -1).T                       # (3*nverts, ncols)
        return D[:, :nb], D[:, nb:nb + 3 * nj]

    res = {r: [] for r in REG}
    conv = []
    for c, i in enumerate(sel):
        for region in REG:
            Sb, St = subspaces(i, region, 1e-2)
            ang = principal_angles(St, Sb)
            res[region].append((float(ang.min()), int((ang < 10).sum()), len(ang)))
        if c < 5:   # numerical convergence check on the first few
            Sb2, St2 = subspaces(i, "anterior", 1e-3)
            conv.append(abs(principal_angles(St2, Sb2).min() - res["anterior"][-1][0]))
        if c % 25 == 0:
            print(f"  {c}/{len(sel)}", flush=True)

    numer_ok = max(conv) < 2.0
    print(f"\n[NUMERICAL CHECK] smallest principal angle shifts at most {max(conv):.2f} deg "
          f"between step sizes 1e-2 and 1e-3 (bar <2.0)  ->  "
          f"{'PASS' if numer_ok else 'FAIL -- DIFFERENCING NOT CONVERGED, RUN IS VOID'}")

    A = {r: np.array([x[0] for x in res[r]]) for r in REG}
    NL = {r: np.array([x[1] for x in res[r]]) for r in REG}
    print(f"\n=== smallest principal angle between the POSE and SHAPE subspaces (degrees) ===")
    print(f"{'region':>10}{'median':>9}{'p10':>8}{'p90':>8}{'#angles<10deg (med)':>22}")
    for r in REG:
        print(f"{r:>10}{np.median(A[r]):>9.2f}{np.percentile(A[r],10):>8.2f}"
              f"{np.percentile(A[r],90):>8.2f}{np.median(NL[r]):>22.0f}")
    h1 = np.median(A["anterior"]) < 20 and (np.median(A["gaster"]) - np.median(A["anterior"])) >= 10
    print(f"[H1 anterior is degenerate] anterior {np.median(A['anterior']):.2f} deg vs gaster "
          f"{np.median(A['gaster']):.2f} deg  ->  {'PASS' if h1 else 'FAIL'}")

    from scipy.stats import mannwhitneyu
    fl = flag[sel]
    U = mannwhitneyu(A["anterior"][fl], A["anterior"][~fl], alternative="less")
    h2 = U.pvalue < 0.05
    print(f"\n[H2 degeneracy tracks pathology] anterior min-angle flagged "
          f"{np.median(A['anterior'][fl]):.2f} vs unflagged "
          f"{np.median(A['anterior'][~fl]):.2f} deg, one-sided p={U.pvalue:.4g}  ->  "
          f"{'PASS' if h2 else 'FAIL'}")

    # ---- beta- vs theta-induced ANTERIOR JOINT displacement ----------------------------
    print("\n=== does changing SHAPE move the anterior joints as much as changing POSE? ===")
    Jr = torch.as_tensor(np.asarray(M["Jr"]), device=dev, dtype=torch.float32)
    ai = [jn.index(j) for j in ANTERIOR]
    dj_b, dj_t = [], []
    for i in sel[:40]:
        V, nb, nj = verts_batch(i, "anterior", 1e-2)
        Jt = torch.einsum("ij,njk->nik", Jr, torch.as_tensor(V, device=dev,
                                                             dtype=torch.float32)).cpu().numpy()
        d = (Jt[1::2, ai] - Jt[2::2, ai]) / 0.02
        dj_b.append(np.linalg.norm(d[:nb], axis=-1).mean())
        dj_t.append(np.linalg.norm(d[nb:], axis=-1).mean())
    print(f"  mean anterior joint displacement per unit beta : {np.mean(dj_b):.5f}")
    print(f"  mean anterior joint displacement per radian    : {np.mean(dj_t):.5f}")
    print(f"  ratio beta/theta: {np.mean(dj_b)/max(np.mean(dj_t),1e-12):.3f}")

    # ---- per-PC anterior trait table ----------------------------------------------------
    print("\n=== each beta PC at +/-2 sd: effect on the anterior traits (template pose) ===")
    sd = np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64)))
    Bp = 2 * NB + 1
    sm2 = SMAL3DFitter(batch_size=Bp, device=dev)
    with torch.no_grad():
        for k in range(NB):
            sm2.betas[1 + 2 * k, k] = float(2 * sd[k])
            sm2.betas[2 + 2 * k, k] = float(-2 * sd[k])
        Vp = sm2().detach().cpu().numpy().astype(np.float64)
    tp = TX.traits(Vp, M, lm)
    Rp = {k: tp[k.split("/")[0]] / np.maximum(tp[k.split("/")[1]], 1e-12) for k in ("ML/HL", "HW/HL")}
    pcs = sorted(range(NB), key=lambda k: -abs(Rp["ML/HL"][1 + 2 * k] - Rp["ML/HL"][2 + 2 * k]))
    print(f"{'PC':>4}{'ML/HL -2sd':>12}{'+2sd':>8}{'range':>8}{'HW/HL -2sd':>12}{'+2sd':>8}{'range':>8}")
    for k in pcs[:8]:
        a, b = Rp["ML/HL"][2 + 2 * k], Rp["ML/HL"][1 + 2 * k]
        c, e = Rp["HW/HL"][2 + 2 * k], Rp["HW/HL"][1 + 2 * k]
        print(f"{k:>4}{a:>12.3f}{b:>8.3f}{abs(b-a):>8.3f}{c:>12.3f}{e:>8.3f}{abs(e-c):>8.3f}")

    # ---- violation predictor -------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ji_all = [jn.index(j) - 1 for j in ANTERIOR]
    X = np.concatenate([PAR["betas"], PAR["joint_rot"][:, ji_all].reshape(n, -1)], axis=1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.1))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    auc = cross_val_score(clf, X, flag, cv=cv, scoring="roc_auc").mean()
    null = np.mean([cross_val_score(clf, X, rng.permutation(flag), cv=cv,
                                    scoring="roc_auc").mean() for _ in range(5)])
    h3 = auc >= 0.75
    print(f"\n[H3 violations predictable from parameters] 5-fold AUC {auc:.3f} "
          f"(permutation null {null:.3f}, bar >=0.75)  ->  {'PASS' if h3 else 'FAIL'}")

    json.dump({"numerical_check_deg": float(max(conv)), "numerical_ok": bool(numer_ok),
               "severity_pct": {str(q): float(np.percentile(s_f, q)) for q in (10,25,50,75,90,100)},
               "min_angle": {r: [float(x) for x in A[r]] for r in REG},
               "n_angles_below10": {r: [int(x) for x in NL[r]] for r in REG},
               "flagged_in_sample": [bool(x) for x in fl],
               "median_min_angle": {r: float(np.median(A[r])) for r in REG},
               "mw_p": float(U.pvalue), "auc": float(auc), "auc_null": float(null),
               "dj_beta": float(np.mean(dj_b)), "dj_theta": float(np.mean(dj_t)),
               "bars": {"numerical": bool(numer_ok), "H1": bool(h1), "H2": bool(h2), "H3": bool(h3)}},
              open(os.path.join(HERE, "v3_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
