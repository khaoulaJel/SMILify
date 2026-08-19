"""B1 analysis: GNC replication at N=50, 2 seeds, 5 arms.

Per-arm, per-seed: leg_distal R (pooled Pearson vs ground truth), part_leg_distal_dist_mean,
part_antenna_dist_mean, mesh-quality sanity (degenerate_tri_frac, folded_face_frac). Then
per-arm cross-seed CV, and comparison against cycle 1's N=12 single-seed GNC numbers.
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

ARMS = ["baseline", "gnc_medium", "gnc_slow", "gnc_medium_splitdistal", "gnc_legonly"]
SEEDS = [0, 1]


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
    return float(pearsonr(x, y)[0]) if len(x) >= 3 else None, len(x)


def mean_metric(run, col):
    p = os.path.join(MOON, "runs", run, "metrics.csv")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        vals = [float(row[col]) for row in csv.DictReader(fh) if row.get(col) not in (None, "")]
    return float(np.mean(vals)) if vals else None


def main():
    M = ms.load_model(); bones = ms.bone_table(M); tpa = ms.template_part_axes(M)
    ld_bones = [b["name"] for b in bones if block_of(b["name"]) == "leg_distal"]
    gt_by_label = load_gt_rows("synth_clean_n50", M, bones, tpa)

    per_run = {}
    print(f"{'run':<36}{'leg_distal_R':>13}{'leg_distal_dist':>17}{'antenna_dist':>14}{'degen_tri':>11}{'folded_face':>13}")
    for arm in ARMS:
        for seed in SEEDS:
            run = f"N50_{arm}_seed{seed}"
            R, n_bones = leg_distal_R(run, gt_by_label, M, bones, tpa, ld_bones)
            ld = mean_metric(run, "part_leg_distal_dist_mean")
            an = mean_metric(run, "part_antenna_dist_mean")
            deg = mean_metric(run, "degenerate_tri_frac")
            fold = mean_metric(run, "folded_face_frac")
            per_run[run] = dict(arm=arm, seed=seed, leg_distal_R=R, n_bone_obs=n_bones,
                                 part_leg_distal_dist_mean=ld, part_antenna_dist_mean=an,
                                 degenerate_tri_frac=deg, folded_face_frac=fold)
            def f(v, nd=3):
                return f"{v:.{nd}f}" if v is not None else "N/A"
            print(f"{run:<36}{f(R):>13}{f(ld,5):>17}{f(an,5):>14}{f(deg,5):>11}{f(fold,5):>13}")

    print(f"\n{'arm':<26}{'mean R':>10}{'CV(R) %':>10}{'mean ld_dist':>14}{'mean antenna':>14}{'antenna CV %':>14}")
    per_arm = {}
    baseline_R = None
    baseline_an = None
    for arm in ARMS:
        Rs = [per_run[f"N50_{arm}_seed{s}"]["leg_distal_R"] for s in SEEDS]
        lds = [per_run[f"N50_{arm}_seed{s}"]["part_leg_distal_dist_mean"] for s in SEEDS]
        ans = [per_run[f"N50_{arm}_seed{s}"]["part_antenna_dist_mean"] for s in SEEDS]
        mean_R = float(np.mean(Rs)); cv_R = float(np.std(Rs) / abs(mean_R) * 100) if mean_R else None
        mean_ld = float(np.mean(lds))
        mean_an = float(np.mean(ans)); cv_an = float(np.std(ans) / abs(mean_an) * 100) if mean_an else None
        per_arm[arm] = dict(mean_leg_distal_R=mean_R, cv_R_pct=cv_R, mean_leg_distal_dist=mean_ld,
                             mean_antenna_dist=mean_an, cv_antenna_pct=cv_an,
                             seed_values_R=Rs, seed_values_antenna=ans)
        if arm == "baseline":
            baseline_R, baseline_an = mean_R, mean_an
        print(f"{arm:<26}{mean_R:>10.3f}{cv_R:>10.2f}{mean_ld:>14.5f}{mean_an:>14.5f}{cv_an:>14.2f}")

    print(f"\n{'arm':<26}{'delta_R_vs_baseline':>20}{'pct_antenna_cost':>18}")
    for arm in ARMS:
        if arm == "baseline":
            continue
        d = per_arm[arm]
        delta_R = d["mean_leg_distal_R"] - baseline_R
        pct_an = (d["mean_antenna_dist"] - baseline_an) / baseline_an * 100
        d["delta_R_vs_baseline"] = delta_R
        d["pct_antenna_cost_vs_baseline"] = pct_an
        print(f"{arm:<26}{delta_R:>+20.3f}{pct_an:>+18.1f}")

    report = dict(per_run=per_run, per_arm=per_arm)
    with open(os.path.join(OUT, "b1_gnc_n50_analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'b1_gnc_n50_analysis.json')}")


if __name__ == "__main__":
    main()
