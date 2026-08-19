"""B7 analysis: 3-seed (0,1,2) pooled comparison, topofree vs gnc_medium, drop30/drop60.

Seed 0 = existing B6 single-seed runs. Seeds 1,2 = new B7 runs (this follow-up). Applies the
user's own pre-registered decision rule: if delta_R(topofree - gnc_medium) direction holds across
seeds within noise, promote damage-data status to match clean-data language. If it flips or
collapses into noise, gnc_medium stays the damage default.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ABS_SCALE = os.path.join(REPO, "diagnostics", "absolute_scale")
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, ABS_SCALE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from calib_features import block_of  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402

OUT = os.path.join(HERE, "out")
SEEDS = [0, 1, 2]


def load_gt_rows(corpus, M, bones, tpa):
    gt = np.load(os.path.join(MOON, corpus, "ground_truth.npz"))
    gtv_all, names = gt["verts"], [str(x) for x in gt["names"]]
    verts, labels = [], []
    for i, nm in enumerate(names):
        objp = os.path.join(MOON, corpus, f"{nm}.obj")
        if not os.path.isfile(objp):
            continue
        ov, _, _ = load_obj(objp, load_textures=False)
        ov = ov.numpy()
        c = ov.mean(0)
        verts.append((gtv_all[i] - c) / np.abs(ov - c).max())
        labels.append(nm)
    rows, _, _ = ms.measure_verts(np.stack(verts).astype(np.float64), labels, "synth", M, bones, tpa)
    return {r["label"]: r for r in rows}


def leg_distal_R(run, gt_by_label, M, bones, tpa, ld_bones):
    fit_rows, _, _ = ms.measure_run(run, "synth", M, bones, tpa)
    x, y = [], []
    for r in fit_rows:
        stem = r["label"][:-4] if r["label"].endswith(".obj") else r["label"]
        g = gt_by_label.get(stem)
        if g is None:
            continue
        for b in ld_bones:
            if b in r and b in g:
                x.append(r[b]); y.append(g[b])
    return (float(pearsonr(x, y)[0]) if len(x) >= 3 else None)


def mean_metric(run, col):
    p = os.path.join(MOON, "runs", run, "metrics.csv")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        vals = [float(row[col]) for row in csv.DictReader(fh) if row.get(col) not in (None, "")]
    return float(np.mean(vals)) if vals else None


def run_name(label, arm, seed):
    tag = {"gnc_medium": "gncmedium", "topofree": "gnclegonlytopofree"}[arm]
    if seed == 0:
        return f"SYN_clean_{label}_{tag}_w5"
    return f"SYN_clean_{label}_{tag}_w5_seed{seed}"


def main():
    M = ms.load_model(); bones = ms.bone_table(M); tpa = ms.template_part_axes(M)
    ld_bones = [b["name"] for b in bones if block_of(b["name"]) == "leg_distal"]
    report = {}

    for label, corpus in [("drop30", "synth_clean_drop30"), ("drop60", "synth_clean_drop60")]:
        gt_by_label = load_gt_rows(corpus, M, bones, tpa)
        baseline_run = f"SYN_clean_{label}_w5"
        baseline_R = leg_distal_R(baseline_run, gt_by_label, M, bones, tpa, ld_bones)

        cond_report = {"baseline_R": baseline_R, "per_arm": {}}
        print(f"\n=== {label} (baseline R={baseline_R:.3f}) ===")
        for arm in ["gnc_medium", "topofree"]:
            Rs, ans = [], []
            for seed in SEEDS:
                run = run_name(label, arm, seed)
                R = leg_distal_R(run, gt_by_label, M, bones, tpa, ld_bones)
                an = mean_metric(run, "part_antenna_dist_mean")
                Rs.append(R); ans.append(an)
                print(f"  {arm:<10} seed{seed}: R={R:.3f}  antenna={an:.5f}  ({run})")
            mean_R = float(np.mean(Rs)); std_R = float(np.std(Rs)); cv_R = float(std_R / abs(mean_R) * 100) if mean_R else None
            mean_an = float(np.mean(ans))
            cond_report["per_arm"][arm] = dict(seed_R=Rs, mean_R=mean_R, std_R=std_R, cv_R_pct=cv_R,
                                                seed_antenna=ans, mean_antenna=mean_an)
            print(f"  {arm:<10} mean R={mean_R:.3f} (std={std_R:.3f}, CV={cv_R:.1f}%)  mean antenna={mean_an:.5f}")

        gm = cond_report["per_arm"]["gnc_medium"]
        tf = cond_report["per_arm"]["topofree"]
        delta_R_per_seed = [tf["seed_R"][i] - gm["seed_R"][i] for i in range(len(SEEDS))]
        delta_R_mean = float(np.mean(delta_R_per_seed))
        delta_R_std = float(np.std(delta_R_per_seed))
        sign_consistent = all(d > 0 for d in delta_R_per_seed) or all(d < 0 for d in delta_R_per_seed)
        # noise floor: is |mean delta| smaller than the seed-to-seed std of either arm's own R?
        within_noise = abs(delta_R_mean) < max(gm["std_R"], tf["std_R"])
        print(f"  delta_R (topofree - gnc_medium) per seed: {[f'{d:+.3f}' for d in delta_R_per_seed]}")
        print(f"  mean delta_R={delta_R_mean:+.3f} (std={delta_R_std:.3f}), sign_consistent={sign_consistent}, "
              f"within_own_seed_noise_floor={within_noise}")
        cond_report["delta_R_per_seed"] = delta_R_per_seed
        cond_report["delta_R_mean"] = delta_R_mean
        cond_report["delta_R_std"] = delta_R_std
        cond_report["sign_consistent_across_seeds"] = sign_consistent
        cond_report["within_own_seed_noise_floor"] = within_noise
        report[label] = cond_report

    # overall verdict per the user's pre-registered rule
    both_hold = all(report[c]["sign_consistent_across_seeds"] and report[c]["delta_R_mean"] >= -0.02
                     for c in ["drop30", "drop60"])
    print(f"\n=== VERDICT ===")
    for c in ["drop30", "drop60"]:
        r = report[c]
        holds = r["sign_consistent_across_seeds"] and not r["within_own_seed_noise_floor"]
        print(f"{c}: sign_consistent={r['sign_consistent_across_seeds']}, "
              f"within_noise_floor={r['within_own_seed_noise_floor']} -> "
              f"{'HOLDS (real, distinguishable from noise)' if holds else 'DOES NOT CLEANLY HOLD'}")

    with open(os.path.join(OUT, "b7_seedrep_analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'b7_seedrep_analysis.json')}")


if __name__ == "__main__":
    main()
