"""B6 analysis: topology-robust gnc_legonly_topofree.

Part 1 (the real test): drop30/drop60 -- does topofree beat gnc_medium (the B4 substitute,
+0.176/+0.198 R) and/or the plain baseline, now that it can actually run there?
Part 2 (clean-data control): synth_clean_n50, 2 seeds -- does topofree regress vs B1's exact-label
gnc_legonly (the NN-based target label transfer is an approximation, not identical machinery)?
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
    return (float(pearsonr(x, y)[0]) if len(x) >= 3 else None), len(x)


def mean_metric(run, col):
    p = os.path.join(MOON, "runs", run, "metrics.csv")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        vals = [float(row[col]) for row in csv.DictReader(fh) if row.get(col) not in (None, "")]
    return float(np.mean(vals)) if vals else None


def f(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else "N/A"


def main():
    M = ms.load_model(); bones = ms.bone_table(M); tpa = ms.template_part_axes(M)
    ld_bones = [b["name"] for b in bones if block_of(b["name"]) == "leg_distal"]
    report = {}

    # --- Part 1: damage corpora ---
    print("=== Part 1: drop30/drop60 (the real test vs gnc_medium) ===")
    print(f"{'condition':<10}{'run':<40}{'leg_distal_R':>13}{'leg_distal_dist':>17}{'antenna_dist':>14}{'degen_tri':>11}{'folded':>9}")
    damage_known = {
        "drop30": {"baseline_R": None, "gnc_medium_delta_R": 0.176, "gnc_medium_antenna_pct": 10.2},
        "drop60": {"baseline_R": None, "gnc_medium_delta_R": 0.198, "gnc_medium_antenna_pct": 12.1},
    }
    for label, corpus in [("drop30", "synth_clean_drop30"), ("drop60", "synth_clean_drop60")]:
        gt_by_label = load_gt_rows(corpus, M, bones, tpa)
        report[label] = {}
        for run in [f"SYN_clean_{label}_w5", f"SYN_clean_{label}_gncmedium_w5", f"SYN_clean_{label}_gnclegonlytopofree_w5"]:
            R, n_obs = leg_distal_R(run, gt_by_label, M, bones, tpa, ld_bones)
            ld = mean_metric(run, "part_leg_distal_dist_mean")
            an = mean_metric(run, "part_antenna_dist_mean")
            deg = mean_metric(run, "degenerate_tri_frac")
            fold = mean_metric(run, "folded_face_frac")
            report[label][run] = dict(leg_distal_R=R, n_bone_obs=n_obs, part_leg_distal_dist_mean=ld,
                                       part_antenna_dist_mean=an, degenerate_tri_frac=deg,
                                       folded_face_frac=fold)
            print(f"{label:<10}{run:<40}{f(R):>13}{f(ld,5):>17}{f(an,5):>14}{f(deg,5):>11}{f(fold,5):>9}")

        base = report[label][f"SYN_clean_{label}_w5"]
        med = report[label][f"SYN_clean_{label}_gncmedium_w5"]
        topo = report[label][f"SYN_clean_{label}_gnclegonlytopofree_w5"]
        d_topo_vs_base = topo["leg_distal_R"] - base["leg_distal_R"]
        pct_an_topo = (topo["part_antenna_dist_mean"] - base["part_antenna_dist_mean"]) / base["part_antenna_dist_mean"] * 100
        d_topo_vs_med = topo["leg_distal_R"] - med["leg_distal_R"]
        beats_medium = d_topo_vs_med > 0
        print(f"  -> topofree delta_R vs baseline = {d_topo_vs_base:+.3f} (antenna {pct_an_topo:+.1f}%)")
        print(f"  -> topofree vs gnc_medium: delta_R = {d_topo_vs_med:+.3f}  {'BEATS' if beats_medium else 'DOES NOT BEAT'} gnc_medium")
        report[label]["delta_R_topofree_vs_baseline"] = d_topo_vs_base
        report[label]["pct_antenna_cost_topofree"] = pct_an_topo
        report[label]["delta_R_topofree_vs_gnc_medium"] = d_topo_vs_med
        report[label]["topofree_beats_gnc_medium"] = beats_medium

    # --- Part 2: clean N50 control ---
    print("\n=== Part 2: synth_clean_n50 control (topofree vs B1's exact-label gnc_legonly) ===")
    gt_n50 = load_gt_rows("synth_clean_n50", M, bones, tpa)
    n50_report = {}
    for arm_run_prefix in ["N50_baseline", "N50_gnc_legonly", "N50_gnc_legonly_topofree"]:
        Rs, ans = [], []
        for seed in [0, 1]:
            run = f"{arm_run_prefix}_seed{seed}"
            R, n_obs = leg_distal_R(run, gt_n50, M, bones, tpa, ld_bones)
            an = mean_metric(run, "part_antenna_dist_mean")
            Rs.append(R); ans.append(an)
            print(f"{run:<40}{f(R):>13}{'':>17}{f(an,5):>14}")
        mean_R = float(np.mean(Rs))
        mean_an = float(np.mean(ans))
        n50_report[arm_run_prefix] = dict(seed_R=Rs, mean_R=mean_R, seed_antenna=ans, mean_antenna=mean_an)

    base_R = n50_report["N50_baseline"]["mean_R"]
    legonly_R = n50_report["N50_gnc_legonly"]["mean_R"]
    topofree_R = n50_report["N50_gnc_legonly_topofree"]["mean_R"]
    print(f"\nN50 mean R: baseline={base_R:.3f}  gnc_legonly(exact)={legonly_R:.3f}  gnc_legonly_topofree={topofree_R:.3f}")
    print(f"topofree vs exact gnc_legonly: delta_R = {topofree_R - legonly_R:+.3f}")
    print(f"topofree vs baseline: delta_R = {topofree_R - base_R:+.3f}")
    n50_report["delta_R_topofree_vs_exact_legonly"] = topofree_R - legonly_R
    n50_report["delta_R_topofree_vs_baseline"] = topofree_R - base_R
    report["n50_control"] = n50_report

    with open(os.path.join(OUT, "b6_topofree_analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'b6_topofree_analysis.json')}")


if __name__ == "__main__":
    main()
