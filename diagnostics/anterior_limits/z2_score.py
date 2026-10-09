"""Z2 step 3 -- score arms A/B against the bar in PREREGISTRATION_Z2_anterior_limits.md."""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
sys.path.insert(0, os.path.join(REPO, "diagnostics", "pose_causality"))

from part_groups import PART_GROUPS_FINE, get_part_vertex_indices  # noqa: E402
from y1_shape_space_analysis import KS, loo_gen_over_spread, rest_space_shaped  # noqa: E402

import pickle  # noqa: E402

BANDS = {1: 45.0, 2: 45.0, 3: 40.0, 4: 40.0, 5: 45.0, 46: 60.0,
         48: 90.0, 52: 90.0, 49: 90.0, 53: 90.0, 50: 75.0, 54: 75.0}
ARMS = {"A_stock": "diagnostics/moonshot/runs/Z2_A_stock",
        "B_limits": "diagnostics/moonshot/runs/Z2_B_limits"}


def load(rd):
    d = np.load(os.path.join(REPO, rd, "Stage_3_deform_fine.npz"))
    return {k: np.asarray(d[k], dtype=np.float64) for k in
            ["betas", "deform_verts", "joint_rot", "log_beta_scales"]}


def main():
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)
    sd_prior = np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64)))
    parts = get_part_vertex_indices(PART_GROUPS_FINE)

    R, out = {}, {}
    for arm, rd in ARMS.items():
        if not os.path.exists(os.path.join(REPO, rd, "Stage_3_deform_fine.npz")):
            raise SystemExit(f"missing fit for {arm}: {rd}")
        R[arm] = load(rd)

    print("=" * 96)
    print("[CHECK 4.1 -- VOIDING] do the authored bands actually bind?")
    print("=" * 96)
    print(f"{'j':>3}{'band':>8}" + f"{'A max|deg|':>13}{'B max|deg|':>13}{'B<=band':>10}{'A exceeds':>11}")
    binds_ok, a_exceeds_any = True, False
    for j, half in sorted(BANDS.items()):
        a = np.degrees(np.abs(R["A_stock"]["joint_rot"][:, j - 1, :])).max()
        b = np.degrees(np.abs(R["B_limits"]["joint_rot"][:, j - 1, :])).max()
        ok, exc = b <= half + 1.0, a > half
        binds_ok &= ok
        a_exceeds_any |= exc
        print(f"{j:>3}{half:>8.0f}{a:>13.1f}{b:>13.1f}{str(ok):>10}{str(exc):>11}")
    check1 = binds_ok and a_exceeds_any
    print(f"  -> {'PASS' if check1 else 'FAIL -- VOID'}  (B within bands: {binds_ok}; "
          f"A exceeds somewhere: {a_exceeds_any})")

    print("\n[CHECK 4.2 -- VOIDING] shape space open in both arms (mean |z| > 0.5)")
    check2 = True
    for arm in ARMS:
        bsd = R[arm]["betas"].std(axis=0)
        z = float((bsd / sd_prior[:len(bsd)]).mean())
        check2 &= z > 0.5
        print(f"  {arm:<10} betas sd {bsd.mean():.5f}  mean |z| {z:.3f}")
    print(f"  -> {'PASS' if check2 else 'FAIL -- VOID'}")

    print("\n[CHECK 4.3] thorax denominator")
    for arm in ARMS:
        m = np.linalg.norm(R[arm]["deform_verts"], axis=2)
        print(f"  {arm:<10} thorax |deform| {m[:, parts['thorax']].mean():.5f}  "
              f"(6.10 range 0.019-0.043)")

    print("\n" + "=" * 96)
    print("ENDPOINT (MECHANISM) -- §6.10 head-carried ratio, paired over 50 specimens")
    print("=" * 96)
    ratios = {}
    for arm in ARMS:
        m = np.linalg.norm(R[arm]["deform_verts"], axis=2)
        thx = m[:, parts["thorax"]].mean(axis=1)
        ratios[arm] = m[:, parts["head"]].mean(axis=1) / thx
    a, b = ratios["A_stock"], ratios["B_limits"]
    da, db = np.abs(a - 1.0), np.abs(b - 1.0)
    improvement = float(da.mean() - db.mean())
    n_better = int((db < da).sum())
    p = float(stats.binomtest(n_better, len(a), 0.5).pvalue)
    print(f"  A mean ratio {a.mean():.4f}   B mean ratio {b.mean():.4f}   delta {b.mean()-a.mean():+.4f}")
    print(f"  |A-1| {da.mean():.4f} -> |B-1| {db.mean():.4f}   improvement {improvement:+.4f} "
          f"(bar >= 0.05)")
    print(f"  specimens closer to 1.0 in B: {n_better}/{len(a)}   sign-test p = {p:.3g}")

    verdict = ("PASS" if (improvement >= 0.05 and p < 0.05 and db.mean() < da.mean())
               else "PARTIAL" if (db.mean() < da.mean() and p < 0.05) else "FAIL")
    if not (check1 and check2):
        verdict = "VOID (a voiding check failed)"

    print("\n" + "-" * 96)
    print("PROXY -- reported, NOT decisive (X1 rule: proxy without mechanism is a FAIL)")
    print("-" * 96)
    for arm in ARMS:
        X = (rest_space_shaped(v_template, shapedirs, R[arm]["betas"]).reshape(len(a), -1)
             + R[arm]["deform_verts"].reshape(len(a), -1))
        _, _, rat = loo_gen_over_spread(X, KS)
        hd = np.degrees(np.abs(R[arm]["joint_rot"][:, 45, :])).max()
        b5 = np.percentile(np.abs(R[arm]["log_beta_scales"][:, 5, :]), 99)
        print(f"  {arm:<10} gen@20/spread {rat[20]:.4f}   head max rot {hd:6.1f} deg   "
              f"b_a_5 p99 |log s| {b5:.4f}")
    print("\n" + "=" * 96)
    print(f"VERDICT (pre-registered): {verdict}")
    print("=" * 96)

    out = {"verdict": verdict, "improvement": improvement, "sign_p": p,
           "A_ratio": float(a.mean()), "B_ratio": float(b.mean()),
           "check_bind": bool(check1), "check_openshape": bool(check2)}
    od = os.path.join(REPO, "diagnostics/anterior_limits/out_Z2")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "z2_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
