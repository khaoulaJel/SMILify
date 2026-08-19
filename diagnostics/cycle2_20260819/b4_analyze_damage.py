"""B4 analysis: gnc_medium vs baseline on drop30/drop60 (gnc_legonly could not run -- see
manifest.json's b4_gnc_legonly_damage FAILED entry for the topology-preserving root cause)."""
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
    return float(pearsonr(x, y)[0]) if len(x) >= 3 else None


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

    report = {}
    print(f"{'condition':<10}{'run':<32}{'leg_distal_R':>13}{'leg_distal_dist':>17}{'antenna_dist':>14}{'degen_tri':>11}{'folded':>9}")
    for label, corpus in [("drop30", "synth_clean_drop30"), ("drop60", "synth_clean_drop60")]:
        gt_by_label = load_gt_rows(corpus, M, bones, tpa)
        report[label] = {}
        for run in [f"SYN_clean_{label}_w5", f"SYN_clean_{label}_gncmedium_w5"]:
            R = leg_distal_R(run, gt_by_label, M, bones, tpa, ld_bones)
            ld = mean_metric(run, "part_leg_distal_dist_mean")
            an = mean_metric(run, "part_antenna_dist_mean")
            deg = mean_metric(run, "degenerate_tri_frac")
            fold = mean_metric(run, "folded_face_frac")
            report[label][run] = dict(leg_distal_R=R, part_leg_distal_dist_mean=ld,
                                       part_antenna_dist_mean=an, degenerate_tri_frac=deg,
                                       folded_face_frac=fold)
            def f(v, nd=3):
                return f"{v:.{nd}f}" if v is not None else "N/A"
            print(f"{label:<10}{run:<32}{f(R):>13}{f(ld,5):>17}{f(an,5):>14}{f(deg,5):>11}{f(fold,5):>9}")

        base = report[label][f"SYN_clean_{label}_w5"]
        gnc = report[label][f"SYN_clean_{label}_gncmedium_w5"]
        delta_R = gnc["leg_distal_R"] - base["leg_distal_R"]
        pct_an = (gnc["part_antenna_dist_mean"] - base["part_antenna_dist_mean"]) / base["part_antenna_dist_mean"] * 100
        print(f"  -> delta_R={delta_R:+.3f}  antenna_cost={pct_an:+.1f}%")
        report[label]["delta_R"] = delta_R
        report[label]["pct_antenna_cost"] = pct_an

    with open(os.path.join(OUT, "b4_damage_analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'b4_damage_analysis.json')}")


if __name__ == "__main__":
    main()
