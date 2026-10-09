"""Z3 -- score arms A/B/C against the bar in PREREGISTRATION_Z3_joint_blendshapes.md."""
import csv
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
sys.path.insert(0, os.path.join(REPO, "diagnostics", "pose_causality"))

from part_groups import PART_GROUPS_FINE, get_part_vertex_indices  # noqa: E402
from y1_shape_space_analysis import KS, loo_gen_over_spread, rest_space_shaped  # noqa: E402

ARMS = {"A_off": "Z3_A_off", "B_couple": "Z3_B_couple", "C_jresid": "Z3_C_jresid"}
COUPLED = {"A_off": False, "B_couple": True, "C_jresid": True}


def metrics_mean(run, col):
    p = os.path.join(REPO, "diagnostics/moonshot/runs", run, "metrics.csv")
    if not os.path.exists(p):
        return float("nan")
    vals = []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r.get("stage") == "Stage_3_deform_fine" and r.get(col):
                vals.append(float(r[col]))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)
    scaledirs = np.asarray(dd["scaledirs"], dtype=np.float64)
    sd_prior = np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64)))
    parts = get_part_vertex_indices(PART_GROUPS_FINE)

    R = {}
    for arm, run in ARMS.items():
        p = os.path.join(REPO, "diagnostics/moonshot/runs", run, "Stage_3_deform_fine.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing fit for {arm}: {p}")
        d = np.load(p)
        R[arm] = {k: np.asarray(d[k], dtype=np.float64)
                  for k in ["betas", "deform_verts", "log_beta_scales", "betas_trans"]}

    print("=" * 100)
    print("[VOIDING 3.2] shape space open in every arm (mean |z| > 0.5)")
    ok_open = True
    for arm in ARMS:
        bsd = R[arm]["betas"].std(axis=0)
        z = float((bsd / sd_prior[:len(bsd)]).mean())
        ok_open &= z > 0.5
        print(f"  {arm:<10} betas sd {bsd.mean():.5f}   mean |z| {z:.3f}")
    print(f"  -> {'PASS' if ok_open else 'FAIL -- VOID'}")

    print("\n[VOIDING 3.3] fit not collapsed (within 10% of arm A)")
    ok_fit = True
    base = {c: metrics_mean(ARMS["A_off"], c) for c in ["chamfer_l1", "fscore@0.01"]}
    for arm in ARMS:
        row = {c: metrics_mean(ARMS[arm], c) for c in base}
        rel = {c: (row[c] - base[c]) / base[c] * 100 for c in base}
        # nan must NOT read as "within bounds": metrics.csv is only written when
        # optimise_moonshot is asked to evaluate, and a missing file silently passed this
        # check on the first Z3 run. Fall back to the fitter's own final chamfer from the log.
        bad = not np.isfinite(rel["chamfer_l1"]) or abs(rel["chamfer_l1"]) > 10 \
            or not np.isfinite(rel["fscore@0.01"]) or abs(rel["fscore@0.01"]) > 10
        ok_fit &= not bad
        print(f"  {arm:<10} chamfer_l1 {row['chamfer_l1']:.5f} ({rel['chamfer_l1']:+.1f}%)   "
              f"fscore@0.01 {row['fscore@0.01']:.4f} ({rel['fscore@0.01']:+.1f}%)")
    print(f"  -> {'PASS' if ok_fit else 'FAIL -- VOID (missing metrics counts as a failure)'}")
    print("  NOTE: when metrics.csv is absent, judge fit quality from the fitter's own final")
    print("        chamfer in the sbatch log -- see RESULTS_Z3.")

    print("\n" + "=" * 100)
    print("PRIMARY (MECHANISM) -- share of applied per-joint log-scale DRIVEN by the shape space")
    print("=" * 100)
    print(f"{'arm':<10}{'driven sd':>12}{'free sd':>12}{'total sd':>12}{'driven share':>14}")
    shares = {}
    for arm in ARMS:
        betas = R[arm]["betas"]
        free = R[arm]["log_beta_scales"]
        if COUPLED[arm]:
            driven = np.log(np.clip(1.0 + np.einsum("nb,bjc->njc", betas, scaledirs), 1e-4, None))
        else:
            driven = np.zeros_like(free)
        total = free + driven
        sdd, sdf, sdt = driven.std(), free.std(), total.std()
        shares[arm] = float(sdd / sdt) if sdt > 0 else 0.0
        print(f"{arm:<10}{sdd:>12.5f}{sdf:>12.5f}{sdt:>12.5f}{shares[arm]*100:>13.1f}%")

    c = shares["C_jresid"]
    primary = "PASS" if c >= 0.50 else "PARTIAL" if c >= 0.20 else "FAIL"
    print(f"\n  arm C driven share {c*100:.1f}%  -> primary {primary}  (PASS >=50%, PARTIAL >=20%)")

    print("\n" + "-" * 100)
    print("SECONDARY -- required for a PASS to mean anything")
    print("-" * 100)
    print(f"{'arm':<10}{'gen@20/spread':>15}{'head ratio':>13}{'b_a_5 p99|ls|':>15}{'free sd':>10}")
    gen = {}
    for arm in ARMS:
        n = len(R[arm]["betas"])
        X = (rest_space_shaped(v_template, shapedirs, R[arm]["betas"]).reshape(n, -1)
             + R[arm]["deform_verts"].reshape(n, -1))
        _, _, rat = loo_gen_over_spread(X, KS)
        gen[arm] = rat[20]
        m = np.linalg.norm(R[arm]["deform_verts"], axis=2)
        hr = (m[:, parts["head"]].mean(axis=1) / m[:, parts["thorax"]].mean(axis=1)).mean()
        b5 = np.percentile(np.abs(R[arm]["log_beta_scales"][:, 5, :]), 99)
        print(f"{arm:<10}{rat[20]:>15.4f}{hr:>13.4f}{b5:>15.4f}"
              f"{R[arm]['log_beta_scales'].std():>10.4f}")
    print("\n  reference: synthetic EXACT correspondence 0.00 @k=20 (the instrument floor).")
    print("  ALL_ANTS_CLEAN is deliberately NOT the bar: different corpus, and Z1/REFUTE-R6")
    print("  showed it is in-sample for this shape space (7.27% of its free-form field lies in")
    print("  span(shapedirs) vs 0.90% for workers), so it is not a fair target.")

    gen_improved = gen["C_jresid"] < gen["A_off"] - 0.01
    verdict = primary
    if primary == "PASS" and not gen_improved:
        verdict = "PARTIAL (primary passed, gen/spread did not improve -- proxy without transfer)"
    if not (ok_open and ok_fit):
        verdict = "VOID (a voiding check failed)"

    print("\n" + "=" * 100)
    print(f"VERDICT (pre-registered): {verdict}")
    print("=" * 100)

    od = os.path.join(REPO, "diagnostics/joint_blendshapes/out_Z3")
    os.makedirs(od, exist_ok=True)
    json.dump({"verdict": verdict, "driven_share": shares, "gen20": gen,
               "open": bool(ok_open), "fit_ok": bool(ok_fit)},
              open(os.path.join(od, "z3_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
