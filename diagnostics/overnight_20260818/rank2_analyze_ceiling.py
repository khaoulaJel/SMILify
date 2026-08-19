"""Rank 2 analysis: turn CEILING_{pose25,pose35} npz outputs into the actual capacity-floor
number (per-specimen and cohort leg_distal relative error / R vs ground truth), and compare
against the SAME condition's normal zero-init baseline (SYN_clean_w5 / SYN_clean_pose35_w5) so
the ceiling is reported alongside what today's full pipeline achieves.
"""

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


def measure_npz(npz_path, corpus, M, bones, tpa):
    d = np.load(npz_path)
    rows, _, _ = ms.measure_verts(d["verts"].astype(np.float64), [str(x) for x in d["labels"]], "synth", M, bones, tpa)
    return rows


def relerr_and_R(fit_rows, gt_by_label, ld_bones):
    per_spec = {}
    x_all, y_all = [], []
    for r in fit_rows:
        stem = r["label"][:-4] if r["label"].endswith(".obj") else r["label"]
        g = gt_by_label.get(stem)
        if g is None:
            continue
        errs = []
        for b in ld_bones:
            if b in r and b in g and abs(g[b]) > 1e-9:
                errs.append(abs(r[b] - g[b]) / abs(g[b]))
                x_all.append(r[b])
                y_all.append(g[b])
        if errs:
            per_spec[stem] = float(np.mean(errs))
    R = float(pearsonr(x_all, y_all)[0]) if len(x_all) >= 3 else None
    return per_spec, R


def main():
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)
    ld_bones = [b["name"] for b in bones if block_of(b["name"]) == "leg_distal"]

    conditions = [
        ("pose25", "synth_clean", "diagnostics/overnight_20260818/runs/CEILING_pose25/Stage_0_ceiling.npz", "SYN_clean_w5"),
        ("pose35", "synth_clean_pose35", "diagnostics/overnight_20260818/runs/CEILING_pose35/Stage_0_ceiling.npz", "SYN_clean_pose35_w5"),
    ]

    report = {}
    print(f"{'condition':<10}{'arm':<12}{'cohort_R':>10}{'mean_relerr':>14}{'median_relerr':>16}")
    for label, corpus, ceiling_npz, baseline_run in conditions:
        gt_verts = load_gt_verts(corpus)
        labels = list(gt_verts.keys())
        gt_rows, _, _ = ms.measure_verts(np.stack([gt_verts[l] for l in labels]).astype(np.float64), labels, "synth", M, bones, tpa)
        gt_by_label = {r["label"]: r for r in gt_rows}

        ceiling_rows = measure_npz(os.path.join(REPO, ceiling_npz), corpus, M, bones, tpa)
        ceiling_relerr, ceiling_R = relerr_and_R(ceiling_rows, gt_by_label, ld_bones)

        baseline_rows, _, _ = ms.measure_run(baseline_run, "synth", M, bones, tpa)
        baseline_relerr, baseline_R = relerr_and_R(baseline_rows, gt_by_label, ld_bones)

        for arm, relerr, R in [("CEILING (GT pose + free bone scale)", ceiling_relerr, ceiling_R),
                                (f"baseline ({baseline_run}, zero-init, full pipeline)", baseline_relerr, baseline_R)]:
            vals = list(relerr.values())
            mean_e = float(np.mean(vals)) if vals else None
            med_e = float(np.median(vals)) if vals else None
            print(f"{label:<10}{arm:<40}{R if R is not None else float('nan'):>10.3f}{mean_e:>14.3f}{med_e:>16.3f}")

        report[label] = dict(
            corpus=corpus,
            ceiling=dict(R=ceiling_R, per_specimen_relerr=ceiling_relerr,
                         mean_relerr=float(np.mean(list(ceiling_relerr.values()))),
                         median_relerr=float(np.median(list(ceiling_relerr.values())))),
            baseline=dict(run=baseline_run, R=baseline_R, per_specimen_relerr=baseline_relerr,
                          mean_relerr=float(np.mean(list(baseline_relerr.values()))),
                          median_relerr=float(np.median(list(baseline_relerr.values())))),
        )

    with open(os.path.join(OUT, "rank2_ceiling_analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'rank2_ceiling_analysis.json')}")


if __name__ == "__main__":
    main()
