"""Rank 5 analysis: distal_quota_protected (C5) vs the matching splitdistal-only baseline and
the plain baseline, at pose25 and pose0. Falsifier metric: part_antenna_dist_mean must NOT
regress for H3-vs-H7 to resolve in favour of 'sampler artifact' rather than 'fundamental'.
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
    from pytorch3d.io import load_obj
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
    return float(pearsonr(x, y)[0]) if len(x) >= 3 else None


def mean_metric(run, col):
    p = os.path.join(MOON, "runs", run, "metrics.csv")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        vals = [float(row[col]) for row in csv.DictReader(fh) if row.get(col)]
    return float(np.mean(vals)) if vals else None


def main():
    M = ms.load_model(); bones = ms.bone_table(M); tpa = ms.template_part_axes(M)
    ld_bones = [b["name"] for b in bones if block_of(b["name"]) == "leg_distal"]

    conditions = [
        ("pose25", "synth_clean", ["SYN_clean_w5", "SYN_clean_pose25_splitdistal_w5", "SYN_clean_pose25_c5protected_w5"]),
        ("pose0", "synth_clean_pose0", ["SYN_clean_pose0_w5", "SYN_clean_pose0_splitdistal_w5", "SYN_clean_pose0_c5protected_w5"]),
    ]

    report = {}
    print(f"{'condition':<10}{'run':<34}{'leg_distal_dist':>16}{'antenna_dist':>14}{'leg_distal_R':>14}")
    for label, corpus, runs in conditions:
        gt_by_label = load_gt_rows(corpus, M, bones, tpa)
        report[label] = {}
        for run in runs:
            ld = mean_metric(run, "part_leg_distal_dist_mean")
            an = mean_metric(run, "part_antenna_dist_mean")
            R = leg_distal_R(run, gt_by_label, M, bones, tpa, ld_bones)
            report[label][run] = dict(part_leg_distal_dist_mean=ld, part_antenna_dist_mean=an, leg_distal_R=R)
            ld_s = f"{ld:.5f}" if ld is not None else "  N/A"
            an_s = f"{an:.5f}" if an is not None else "  N/A"
            R_s = f"{R:.3f}" if R is not None else "  N/A"
            print(f"{label:<10}{run:<34}{ld_s:>16}{an_s:>14}{R_s:>14}")

    with open(os.path.join(OUT, "rank5_c5_analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'rank5_c5_analysis.json')}")


if __name__ == "__main__":
    main()
