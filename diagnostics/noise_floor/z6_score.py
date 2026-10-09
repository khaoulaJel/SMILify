"""Z6 -- score against the bar in PREREGISTRATION_Z6_noise_floor.md."""
import json
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
sys.path.insert(0, os.path.join(REPO, "diagnostics", "pose_causality"))

from y1_shape_space_analysis import rest_space_shaped  # noqa: E402

RUN = "Z6_paired50"
BENCH50_SPREAD = 0.01273      # Z3/Z5 arm A, bench50_clean
BENCH50_CHAMFER = 0.00115


def loo_ratio(X, k=20, exclude=None):
    """gen@k / spread, leave-one-out. `exclude[i]` lists EXTRA rows to drop with row i.

    exclude=None reproduces y1_shape_space_analysis.loo_gen_over_spread exactly (verified in the
    voiding block). exclude=<partner index> is the leave-PAIR-out variant: the held-out specimen's
    conspecific is removed from the training set too, so the model cannot lean on a near-identical
    animal.
    """
    n = X.shape[0]
    e0, ek = [], []
    for i in range(n):
        drop = {i} | set(exclude[i] if exclude else [])
        keep = [j for j in range(n) if j not in drop]
        R = X[keep]
        mu = R.mean(0)
        d = X[i] - mu
        e0.append(np.sqrt((d ** 2).mean()))
        Rc = R - mu
        _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
        P = Vt[:min(k, Vt.shape[0])]
        recon = P.T @ (P @ d)
        ek.append(np.sqrt(((d - recon) ** 2).mean()))
    spread, gen = float(np.mean(e0)), float(np.mean(ek))
    return gen / spread, spread, gen


def main():
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    vt = np.asarray(dd["v_template"], dtype=np.float64)
    shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)
    sd_prior = np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64)))

    rd = os.path.join(REPO, "diagnostics/moonshot/runs", RUN)
    d = np.load(os.path.join(rd, "Stage_3_deform_fine.npz"))
    betas = np.asarray(d["betas"], dtype=np.float64)
    dv = np.asarray(d["deform_verts"], dtype=np.float64)
    names = [str(x) for x in np.asarray(d["labels"]).ravel()] if "labels" in d else None
    n = len(betas)

    pairs = json.load(open(os.path.join(HERE, "paired50/pairs.json")))

    print("=" * 100)
    print("[VOIDING 4.1] corpus is genuinely paired")
    ok_pair = (len(pairs) == 25 and len({k.split("_")[0] for k in pairs}) == 25
               and all(len(v) == 2 for v in pairs.values()) and n == 50)
    print(f"  {len(pairs)} species, {len({k.split('_')[0] for k in pairs})} genera, "
          f"{n} fitted specimens, all pairs size 2: {all(len(v)==2 for v in pairs.values())}")
    print(f"  -> {'PASS' if ok_pair else 'FAIL -- VOID'}")

    # map fitted row order -> partner row
    order = names if names else sorted(f for f in os.listdir(os.path.join(HERE, "paired50"))
                                       if f.endswith(".obj"))
    order = [os.path.basename(str(o)) for o in order]
    pos = {f: i for i, f in enumerate(order)}
    partner = [None] * n
    for sp, (a, b) in pairs.items():
        if a in pos and b in pos:
            partner[pos[a]], partner[pos[b]] = [pos[b]], [pos[a]]
    matched = sum(1 for p in partner if p is not None)
    print(f"  partner map resolved for {matched}/{n} specimens "
          f"{'PASS' if matched == n else 'FAIL -- VOID'}")
    ok_pair &= matched == n

    print("\n[VOIDING 4.2] shape space open")
    bsd = betas.std(axis=0)
    z = float((bsd / sd_prior[:len(bsd)]).mean())
    print(f"  betas sd {bsd.mean():.5f}  mean |z| {z:.3f}  -> {'PASS' if z > 0.5 else 'FAIL'}")

    print("\n[VOIDING 4.3] fit not collapsed")
    logs = sorted([f for f in os.listdir(os.path.join(HERE, "sbatch_logs")) if f.endswith(".log")])
    cham = float("nan")
    if logs:
        txt = open(os.path.join(HERE, "sbatch_logs", logs[-1])).read()
        m = re.findall(r"Stage_3_deform_fine\] it  999/1000.*?chamfer=([0-9.]+)", txt)
        cham = float(m[0]) if m else float("nan")
    rel = (cham - BENCH50_CHAMFER) / BENCH50_CHAMFER * 100
    ok_fit = bool(np.isfinite(rel) and abs(rel) <= 15)
    print(f"  chamfer {cham:.5f} vs bench50 {BENCH50_CHAMFER:.5f} ({rel:+.1f}%)  "
          f"-> {'PASS' if ok_fit else 'FAIL -- VOID'}")

    X = rest_space_shaped(vt, shapedirs, betas).reshape(n, -1) + dv.reshape(n, -1)
    P = rest_space_shaped(vt, shapedirs, betas).reshape(n, -1)

    loo, spread, _ = loo_ratio(X)
    print("\n[VOIDING 4.4] spread not degenerate")
    print(f"  spread {spread:.5f} vs bench50 {BENCH50_SPREAD:.5f} "
          f"(x{spread/BENCH50_SPREAD:.2f})  -> "
          f"{'PASS' if 0.5 <= spread/BENCH50_SPREAD <= 2.0 else 'FAIL -- VOID'}")
    ok_spread = 0.5 <= spread / BENCH50_SPREAD <= 2.0

    print("\n" + "=" * 100)
    print("ZERO-MODEL CONTROL -- replicate ratio: within-pair distance / population spread")
    print("null (no species signal) = sqrt(2) ~ 1.414 by construction; 0 = identical specimens")
    print("=" * 100)
    for nm, M in [("composed shape", X), ("betas only", P)]:
        wp = [np.sqrt(((M[i] - M[partner[i][0]]) ** 2).mean()) for i in range(n)]
        _, sp_, _ = loo_ratio(M)
        print(f"  {nm:<18} within-pair {np.mean(wp):.5f}   spread {sp_:.5f}   "
              f"ratio {np.mean(wp)/sp_:.4f}")

    print("\n" + "=" * 100)
    print("PRIMARY -- gen@20/spread with the CONSPECIFIC PRESENT (best case for the pipeline)")
    print("=" * 100)
    print(f"{'representation':<22}{'LOO (conspecific in)':>22}{'leave-PAIR-out':>18}{'gap':>9}")
    res = {}
    for nm, M in [("composed shape", X), ("betas only", P)]:
        a, _, _ = loo_ratio(M)
        b, _, _ = loo_ratio(M, exclude=partner)
        res[nm] = {"loo": a, "lpo": b}
        print(f"{nm:<22}{a:>22.4f}{b:>18.4f}{b-a:>+9.4f}")

    primary = res["composed shape"]["loo"]
    if primary <= 0.35:
        verdict = f"NOT A FLOOR (LOO {primary:.4f} <= 0.35) -- capacity project licensed"
    elif primary >= 0.55:
        verdict = f"FLOOR (LOO {primary:.4f} >= 0.55) -- per-specimen noise dominates"
    else:
        verdict = f"PARTIAL (LOO {primary:.4f}, between 0.35 and 0.55)"
    if not (ok_pair and z > 0.5 and ok_fit and ok_spread):
        verdict = "VOID (a voiding check failed)"

    print("\n" + "=" * 100)
    print(f"VERDICT (pre-registered): {verdict}")
    print("  bench50_clean reference (no conspecifics): gen@20/spread ~0.60")
    print("=" * 100)
    od = os.path.join(HERE, "out_Z6")
    os.makedirs(od, exist_ok=True)
    json.dump({"verdict": verdict, "results": res, "spread": spread, "chamfer": cham,
               "betas_z": z}, open(os.path.join(od, "z6_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
