"""Z5 -- score arms A/B/C against the bar in PREREGISTRATION_Z5_deform_symmetry.md."""
import json
import os
import pickle
import re
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
from z4_deform_signal_or_noise import involution  # noqa: E402

ARMS = {"A_off": "Z5_A_off", "B_dsym2": "Z5_B_dsym2", "C_dsym10": "Z5_C_dsym10"}
LOGDIR = os.path.join(HERE, "sbatch_logs")


def final_chamfer(arm_label):
    """Fitter's own final Stage_3 chamfer. metrics.csv is only written when optimise_moonshot is
    asked to evaluate; Z3's fit check passed vacuously on the resulting nan, so read the log."""
    logs = sorted([os.path.join(LOGDIR, f) for f in os.listdir(LOGDIR) if f.endswith(".log")],
                  key=os.path.getmtime)
    if not logs:
        return float("nan")
    txt = open(logs[-1]).read()
    seg = txt.split(arm_label)
    if len(seg) < 2:
        return float("nan")
    hits = re.findall(r"Stage_3_deform_fine\] it  999/1000.*?chamfer=([0-9.]+)", seg[1])
    return float(hits[0]) if hits else float("nan")


def main():
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    vt = np.asarray(dd["v_template"], dtype=np.float64)
    shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)
    sd_prior = np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64)))
    parts = get_part_vertex_indices(PART_GROUPS_FINE)
    idx, md, exact = involution(vt)
    sign = np.array([1.0, -1.0, 1.0])

    print("=" * 100)
    print(f"[VOIDING 3.1] involution: max match {md:.2e}, exact on {100*exact:.1f}%  "
          f"-> {'PASS' if md < 1e-6 and exact > 0.99 else 'FAIL -- VOID'}")

    R = {}
    for arm, run in ARMS.items():
        p = os.path.join(REPO, "diagnostics/moonshot/runs", run, "Stage_3_deform_fine.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing fit for {arm}: {p}")
        d = np.load(p)
        R[arm] = {k: np.asarray(d[k], dtype=np.float64) for k in ["betas", "deform_verts"]}

    print("\n[VOIDING 3.2] the term binds -- symmetric share must rise above A")
    shares = {}
    for arm in ARMS:
        D = R[arm]["deform_verts"]
        S, A = 0.5 * (D + D[:, idx, :] * sign), 0.5 * (D - D[:, idx, :] * sign)
        shares[arm] = float((S ** 2).sum() / ((S ** 2).sum() + (A ** 2).sum()))
        print(f"  {arm:<10} symmetric share {shares[arm]:.4f}")
    binds = shares["B_dsym2"] > shares["A_off"] and shares["C_dsym10"] > shares["A_off"]
    print(f"  -> {'PASS' if binds else 'FAIL -- VOID'}  (null 0.500; A was 0.6006 in Z4)")

    print("\n[VOIDING 3.3] shape space open (mean |z| > 0.5)")
    ok_open = True
    for arm in ARMS:
        bsd = R[arm]["betas"].std(axis=0)
        z = float((bsd / sd_prior[:len(bsd)]).mean())
        ok_open &= z > 0.5
        print(f"  {arm:<10} betas sd {bsd.mean():.5f}   |z| {z:.3f}")
    print(f"  -> {'PASS' if ok_open else 'FAIL -- VOID'}")

    print("\n[VOIDING 3.4] fit not collapsed (final chamfer within 10% of A; nan = FAIL)")
    cham = {arm: final_chamfer(lbl) for arm, lbl in
            zip(ARMS, ["arm A --", "arm B --", "arm C --"])}
    base = cham["A_off"]
    fit_ok = {}
    for arm in ARMS:
        rel = (cham[arm] - base) / base * 100 if np.isfinite(cham[arm]) and np.isfinite(base) \
            else float("nan")
        fit_ok[arm] = bool(np.isfinite(rel) and abs(rel) <= 10)
        print(f"  {arm:<10} chamfer {cham[arm]:.5f}  ({rel:+.1f}%)  "
              f"{'ok' if fit_ok[arm] else 'VOID'}")

    print("\n" + "=" * 100)
    print("PRIMARY -- gen@20/spread on WORKERS   (A ~0.6147; post-hoc projection bound 0.5292)")
    print("floor under exact correspondence 0.00 @k=20. ALL_ANTS_CLEAN is NOT the target.")
    print("=" * 100)
    print(f"{'arm':<10}{'gen@20/spread':>15}{'head ratio':>13}{'sym share':>12}{'chamfer':>11}")
    gen = {}
    for arm in ARMS:
        n = len(R[arm]["betas"])
        X = (rest_space_shaped(vt, shapedirs, R[arm]["betas"]).reshape(n, -1)
             + R[arm]["deform_verts"].reshape(n, -1))
        _, _, rat = loo_gen_over_spread(X, KS)
        gen[arm] = rat[20]
        m = np.linalg.norm(R[arm]["deform_verts"], axis=2)
        hr = (m[:, parts["head"]].mean(axis=1) / m[:, parts["thorax"]].mean(axis=1)).mean()
        print(f"{arm:<10}{rat[20]:>15.4f}{hr:>13.4f}{shares[arm]:>12.4f}{cham[arm]:>11.5f}")

    valid = [a for a in ("B_dsym2", "C_dsym10") if fit_ok[a]]
    best = min(valid, key=lambda a: gen[a]) if valid else None
    if best is None:
        verdict = "FAIL (every improving arm voided by fit collapse)"
    elif gen[best] <= 0.56:
        verdict = f"PASS ({best}, gen@20 {gen[best]:.4f})"
    elif gen["A_off"] - gen[best] >= 0.02:
        verdict = f"PARTIAL ({best}, gen@20 {gen[best]:.4f}, improved "
        verdict += f"{gen['A_off'] - gen[best]:+.4f} but did not reach 0.56)"
    else:
        verdict = "FAIL (no arm improved by >= 0.02)"
    if not (binds and ok_open and md < 1e-6):
        verdict = "VOID (a voiding check failed)"

    print("\n" + "=" * 100)
    print(f"VERDICT (pre-registered): {verdict}")
    print("=" * 100)
    od = os.path.join(REPO, "diagnostics/deform_nature/out_Z5")
    os.makedirs(od, exist_ok=True)
    json.dump({"verdict": verdict, "gen20": gen, "sym_share": shares, "chamfer": cham,
               "fit_ok": fit_ok}, open(os.path.join(od, "z5_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
