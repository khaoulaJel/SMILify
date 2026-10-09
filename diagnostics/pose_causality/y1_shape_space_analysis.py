"""Y1 step 2 -- does pose magnitude degrade the learnable shape structure recovered by the fitter?

Bar in PREREGISTRATION_Y1_pose_causality.md, fixed before any Y1 number was seen.

THE METRIC
----------
Leave-one-out PCA reconstruction of REST-SPACE SHAPED GEOMETRY, matching REPORT.md section 6.4:

    v_shaped_i = v_template + deform_verts_i + shapedirs . betas_i

`smal_torch.__call__` composes exactly this before skinning, so it is the specimen's shape with
pose removed. `log_beta_scales` and `betas_trans` are deliberately NOT folded in -- they act during
skinning, per joint, so they are pose-space quantities (build_shape_space.py's own reasoning).

    spread   = LOO error predicting the population MEAN (k=0)
    gen@k    = LOO error reconstructing from the top k modes of the other n-1 specimens
    reported = gen@k / spread     (1.0 = no better than the mean; 0.0 = perfect)

THE TRAP THIS AVOIDS
--------------------
Run on GROUND-TRUTH parameters this metric is pose-INVARIANT by construction, because pose is
removed analytically. So GT is used ONLY as the instrument's positive control (check 5.1); the
endpoint is computed from the FITTED parameters, where the causal chain
pose -> fit degrades -> correspondence degrades -> shape space loses structure can actually act.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402

KS = [1, 5, 10, 20, 25, 40]


def rest_space_shaped(v_template, shapedirs, betas, deform_verts=None):
    """(n, V, 3) rest-space shaped geometry. shapedirs is (V,3,B) or (B,V,3)."""
    sd = np.asarray(shapedirs)
    if sd.ndim != 3:
        raise SystemExit(f"unexpected shapedirs ndim {sd.ndim}")
    if sd.shape[0] == v_template.shape[0]:          # (V,3,B)
        blend = np.einsum("vab,nb->nva", sd, betas)
    else:                                            # (B,V,3)
        blend = np.einsum("bva,nb->nva", sd, betas)
    out = v_template[None] + blend
    if deform_verts is not None:
        out = out + deform_verts
    return out


def loo_gen_over_spread(X, ks):
    """Leave-one-out PCA reconstruction error, and its ratio to the population spread.

    X: (n, D). Returns (spread, {k: gen_k}, {k: ratio}).
    For each held-out i, PCA is fit on the OTHER n-1 specimens only -- the held-out specimen never
    contributes to the mean or the modes, which is what makes this a generalisation measure rather
    than an in-sample fit.
    """
    n = X.shape[0]
    errs = {0: []}
    for k in ks:
        errs[k] = []
    for i in range(n):
        rest = np.delete(X, i, axis=0)
        mu = rest.mean(0)
        R = rest - mu
        # economy SVD; components are rows of Vt
        try:
            _, _, Vt = np.linalg.svd(R, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        d = X[i] - mu
        errs[0].append(np.sqrt((d ** 2).mean()))
        for k in ks:
            kk = min(k, Vt.shape[0])
            P = Vt[:kk]
            recon = P.T @ (P @ d)
            errs[k].append(np.sqrt(((d - recon) ** 2).mean()))
    spread = float(np.mean(errs[0]))
    gen = {k: float(np.mean(errs[k])) for k in ks}
    ratio = {k: (gen[k] / spread if spread > 1e-12 else float("nan")) for k in ks}
    return spread, gen, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="diagnostics/pose_causality/corpora")
    ap.add_argument("--runs_root", default="diagnostics/moonshot/runs")
    ap.add_argument("--run_prefix", default="Y1_ps")
    ap.add_argument("--out_dir", default="diagnostics/pose_causality/out_Y1")
    args = ap.parse_args()
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)

    ladder = json.load(open(os.path.join(REPO, args.corpora, "ladder.json")))
    n_betas = int(ladder["n_betas"])
    scales = sorted([k for k in ladder if k != "n_betas"], key=float)

    rows, gt_rows = [], []
    for s in scales:
        gt = np.load(os.path.join(REPO, args.corpora, f"ps{float(s):.2f}", "ground_truth.npz"))
        betas_gt = gt["betas"].astype(np.float64)
        verts_gt = gt["verts"].astype(np.float64)
        deg = ladder[s]["mean_deg_per_joint"]

        # ---- check 5.1: instrument validation on GROUND TRUTH -------------------------------
        Xgt = rest_space_shaped(v_template, shapedirs, betas_gt).reshape(len(betas_gt), -1)
        sp_gt, gen_gt, rat_gt = loo_gen_over_spread(Xgt, KS)
        gt_rows.append({"pose_scale": float(s), "spread": sp_gt, "ratio": rat_gt})

        # ---- endpoint: the FITTED parameters --------------------------------------------------
        rd = os.path.join(REPO, args.runs_root, f"{args.run_prefix}{float(s):.2f}")
        p = os.path.join(rd, "Stage_3_deform_fine.npz")
        if not os.path.exists(p):
            print(f"  !! missing fit for pose_scale {s}: {p}")
            continue
        d = np.load(p)
        betas_fit = np.asarray(d["betas"], dtype=np.float64)
        dv = np.asarray(d["deform_verts"], dtype=np.float64) if "deform_verts" in d else None
        verts_fit = np.asarray(d["verts"], dtype=np.float64) if "verts" in d else None
        Xf = rest_space_shaped(v_template, shapedirs, betas_fit, dv).reshape(len(betas_fit), -1)
        sp, gen, rat = loo_gen_over_spread(Xf, KS)

        # ---- check 5.4: fit quality, so a shape-space collapse is distinguishable from
        #      total fit failure. Symmetric chamfer against the GT mesh, same topology.
        cham = float("nan")
        if verts_fit is not None and verts_fit.shape == verts_gt.shape:
            cham = float(np.linalg.norm(verts_fit - verts_gt, axis=2).mean())
        rows.append({"pose_scale": float(s), "deg": deg, "spread": sp, "gen": gen,
                     "ratio": rat, "chamfer_to_gt": cham, "n": int(len(betas_fit))})
        print(f"pose_scale {s} ({deg:5.2f} deg/joint): gen@20/spread {rat[20]:.4f}  "
              f"gen@10 {rat[10]:.4f}  vert-err-to-GT {cham:.5f}", flush=True)

    # ------------------------------------------------------------------ check 5.1 verdict
    print("\n" + "=" * 94)
    print("[CHECK 5.1 -- VOIDING] instrument validation on GROUND-TRUTH geometry")
    print(f"data is exactly {n_betas}-dimensional by construction; gen/spread must reach <=0.10 "
          f"by k={n_betas}")
    print("=" * 94)
    print(f"{'pose_scale':<12}" + "".join(f"{'gen@'+str(k):>10}" for k in KS))
    for g in gt_rows:
        print(f"{g['pose_scale']:<12.2f}" + "".join(f"{g['ratio'][k]:>10.4f}" for k in KS))
    gt_ok = all(g["ratio"][n_betas if n_betas in KS else 25] <= 0.10 for g in gt_rows)
    print(f"  -> {'PASS' if gt_ok else 'FAIL -- VOID: the metric is not the right instrument'}")
    print("  (GT is pose-invariant by construction, so these rows should also be near-identical "
          "across the ladder -- a further sanity signal)")

    if not rows:
        print("\nno fits found; nothing to report")
        return

    # ------------------------------------------------------------------ endpoint
    print("\n" + "=" * 94)
    print("ENDPOINT -- gen@k / spread from the FITTED parameters, vs pose magnitude")
    print("anchors: ALL_ANTS_CLEAN 0.50 @k=20, worker registrations 0.98 @k=20")
    print("=" * 94)
    print(f"{'pose_scale':<12}{'deg/joint':>11}" + "".join(f"{'gen@'+str(k):>10}" for k in KS)
          + f"{'vert-err':>11}")
    for r in rows:
        print(f"{r['pose_scale']:<12.2f}{r['deg']:>11.2f}"
              + "".join(f"{r['ratio'][k]:>10.4f}" for k in KS)
              + f"{r['chamfer_to_gt']:>11.5f}")

    g20 = [r["ratio"][20] for r in rows]
    degs = [r["deg"] for r in rows]
    monotone = all(b >= a for a, b in zip(g20, g20[1:]))
    rho, rho_p = stats.spearmanr(degs, g20) if len(g20) > 2 else (float("nan"),) * 2
    spans = (g20[0] <= 0.60) and (g20[-1] >= 0.90)
    verdict = ("CONFIRMED" if (monotone and spans)
               else "PARTIAL" if (monotone and rho > 0)
               else "REFUTED")
    if not gt_ok:
        verdict = "VOID (instrument failed validation)"

    print("\n" + "=" * 94)
    print(f"monotone in pose magnitude : {monotone}")
    print(f"Spearman rho (deg vs gen@20/spread) : {rho:.3f}  (p={rho_p:.3g})")
    print(f"spans <=0.60 -> >=0.90     : {spans}   (lowest {g20[0]:.4f}, highest {g20[-1]:.4f})")
    print(f"\nVERDICT (pre-registered): {verdict}")
    print("=" * 94)

    payload = {"verdict": verdict, "n_betas": n_betas, "instrument_ok": bool(gt_ok),
               "gt_rows": gt_rows, "rows": rows, "monotone": bool(monotone),
               "spearman_rho": float(rho), "spearman_p": float(rho_p), "spans": bool(spans)}
    with open(os.path.join(out_dir, "y1_results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {out_dir}/y1_results.json")


if __name__ == "__main__":
    main()
