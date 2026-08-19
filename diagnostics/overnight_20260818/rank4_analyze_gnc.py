"""Rank 4 analysis: GNC schedules (fast/medium/slow) vs the SYN_clean_w5 baseline (D1_low.yaml,
no robust kernel), on the SAME 12 pose25 specimens. Reports chamfer_l2, part_leg_distal_dist_mean,
part_antenna_dist_mean (collateral check), and true leg_distal ground-truth R.
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


def load_gt_verts(corpus_dir):
    gt = np.load(os.path.join(MOON, corpus_dir, "ground_truth.npz"))
    gtv_all, names = gt["verts"], [str(x) for x in gt["names"]]
    out = {}
    for i, nm in enumerate(names):
        objp = os.path.join(MOON, corpus_dir, f"{nm}.obj")
        if not os.path.isfile(objp):
            continue
        ov, _, _ = load_obj(objp, load_textures=False)
        ov = ov.numpy()
        c = ov.mean(0)
        out[nm] = (gtv_all[i] - c) / np.abs(ov - c).max()
    return out


def leg_distal_R(run, corpus, gt_by_label, M, bones, tpa, ld_bones):
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
    gt_by_label = {r["label"]: r for r in ms.measure_verts(
        np.stack(list(load_gt_verts("synth_clean").values())).astype(np.float64),
        list(load_gt_verts("synth_clean").keys()), "synth", M, bones, tpa)[0]}

    runs = ["SYN_clean_w5", "GNC_fast", "GNC_medium", "GNC_slow"]
    report = {}
    print(f"{'run':<16}{'chamfer_l2':>12}{'leg_distal_dist':>18}{'antenna_dist':>15}{'leg_distal_R':>14}")
    for run in runs:
        ch = mean_metric(run, "chamfer_l2")
        ld = mean_metric(run, "part_leg_distal_dist_mean")
        an = mean_metric(run, "part_antenna_dist_mean")
        R = leg_distal_R(run, "synth_clean", gt_by_label, M, bones, tpa, ld_bones)
        report[run] = dict(chamfer_l2=ch, part_leg_distal_dist_mean=ld, part_antenna_dist_mean=an, leg_distal_R=R)
        print(f"{run:<16}{ch:>12.5f}{ld:>18.5f}{an if an is not None else float('nan'):>15.5f}{R if R is not None else float('nan'):>14.3f}")

    with open(os.path.join(OUT, "rank4_gnc_analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'rank4_gnc_analysis.json')}")


if __name__ == "__main__":
    main()
