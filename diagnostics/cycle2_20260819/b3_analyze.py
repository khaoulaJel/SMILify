"""Cycle2 B3 analysis: complete the 2x2 capacity-vs-sampling table."""
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


def relerr_and_R(npz_path, gt_by_label, ld_bones, M, bones, tpa):
    d = np.load(npz_path)
    rows, _, _ = ms.measure_verts(d["verts"].astype(np.float64), [str(x) for x in d["labels"]], "synth", M, bones, tpa)
    x, y, errs = [], [], []
    for r in rows:
        stem = r["label"][:-4] if r["label"].endswith(".obj") else r["label"]
        g = gt_by_label.get(stem)
        if g is None:
            continue
        for b in ld_bones:
            if b in r and b in g and abs(g[b]) > 1e-9:
                x.append(r[b]); y.append(g[b])
                errs.append(abs(r[b] - g[b]) / abs(g[b]))
    R = float(pearsonr(x, y)[0]) if len(x) >= 3 else None
    return R, float(np.mean(errs)) if errs else None


def main():
    M = ms.load_model(); bones = ms.bone_table(M); tpa = ms.template_part_axes(M)
    ld_bones = [b["name"] for b in bones if block_of(b["name"]) == "leg_distal"]

    prev = json.load(open(os.path.join(REPO, "diagnostics/overnight_20260818/out/rank2_ceiling_analysis.json")))

    report = {}
    print(f"{'condition':<10}{'sampler':<22}{'leg_distal_R':>14}{'mean_relerr':>14}")
    for label, corpus in [("pose25", "synth_clean"), ("pose35", "synth_clean_pose35")]:
        gt_by_label = load_gt_rows(corpus, M, bones, tpa)
        npz = os.path.join(REPO, f"diagnostics/cycle2_20260819/runs/B3_CEILING_PROTECTED_{label}/B3_ceiling_protected.npz")
        R, relerr = relerr_and_R(npz, gt_by_label, ld_bones, M, bones, tpa)

        report[label] = {
            "normal_sampler_ceiling": {"R": prev[label]["ceiling"]["R"], "mean_relerr": prev[label]["ceiling"]["mean_relerr"]},
            "leg_protected_sampler_ceiling": {"R": R, "mean_relerr": relerr},
            "baseline_full_pipeline": {"R": prev[label]["baseline"]["R"], "mean_relerr": prev[label]["baseline"]["mean_relerr"]},
        }
        for samp, d in [("normal sampler (Rank 2)", report[label]["normal_sampler_ceiling"]),
                        ("leg-protected sampler (B3, new)", report[label]["leg_protected_sampler_ceiling"]),
                        ("baseline full pipeline", report[label]["baseline_full_pipeline"])]:
            print(f"{label:<10}{samp:<22}{d['R']:>14.3f}{d['mean_relerr']:>14.3f}")

    with open(os.path.join(OUT, "b3_ceiling_2x2.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'b3_ceiling_2x2.json')}")


if __name__ == "__main__":
    main()
