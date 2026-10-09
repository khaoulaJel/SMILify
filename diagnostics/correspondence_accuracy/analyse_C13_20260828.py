"""C13 readout, against the bar fixed in PREREGISTRATION_C13_tolerance_weighting_20260828.md.

PRIMARY endpoint: paired per-specimen `co` leg-level error, invtol/invtol2 vs uniform, n=48,
reported as sign + Wilcoxon + paired-t together.
  PASS    co leg error falls by >= 0.05 absolute AND sign p < 0.05
  PARTIAL falls, sign p < 0.05, but < 0.05 absolute
  FAIL    does not fall

GUARD-RAILS, also fixed in advance:
  - distal (ti/ta/pt) leg error must not rise by > 0.02 absolute, else report as a TRADE
  - overall leg_acc must not fall

C13_uniform is a regression check on the edited trainer: it should reproduce P48_cse_all, which
was fit through the UNEDITED code with the same correspondence file and recipe.

Run: python diagnostics/correspondence_accuracy/analyse_C13_20260828.py
"""

import json
import os
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
import labels as lb  # noqa: E402
import confusion as cf  # noqa: E402

ARMS = ["P48_cse_all", "C13_uniform", "C13_invtol", "C13_invtol2"]
CORPUS = "synth_power48"
N_SAMPLE = 8000


def run_arm(arm, face_lab, vlabels, segs):
    d = np.load(os.path.join(MOON, "runs", arm, "Stage_3_deform_fine.npz"))
    labs = [str(x) for x in d["labels"]]
    fitted = d["verts"].astype(np.float64)
    rng = np.random.default_rng(1)   # same seed for every arm -> identical sampled points
    per_spec = {}
    for i, lab in enumerate(labs):
        stem = lab[:-4] if lab.endswith(".obj") else lab
        ov, of, _ = load_obj(os.path.join(MOON, CORPUS, f"{stem}.obj"), load_textures=False)
        pts, plab = cf.sample_true_labeled_points(
            ov.numpy(), of.verts_idx.numpy(), face_lab, N_SAMPLE, rng
        )
        pts_t = torch.tensor(pts, dtype=torch.float32).unsqueeze(0)
        is_leg = plab["leg_id"] != None  # noqa: E711
        fi = torch.tensor(fitted[i], dtype=torch.float32).unsqueeze(0)
        _, leg_pred, _, true_leg = cf.leg_confusion(plab["leg_id"], is_leg, pts_t, fi, vlabels)
        wrong = leg_pred != true_leg
        seg_here = plab["leg_seg"][is_leg]
        rec = {"ALL": float(wrong.mean())}
        for s in segs:
            m = seg_here == s
            rec[s] = float(wrong[m].mean()) if m.any() else np.nan
        dm = np.isin(seg_here, ["ti", "ta", "pt"])
        rec["DISTAL"] = float(wrong[dm].mean()) if dm.any() else np.nan
        per_spec[stem] = rec
    return per_spec


def tests(a, b, res, key):
    ks = sorted(set(res[a]) & set(res[b]))
    d = np.array([res[b][k][key] - res[a][k][key] for k in ks])
    d = d[~np.isnan(d)]
    pos = int((d > 0).sum())
    sign = stats.binomtest(pos, len(d)).pvalue
    try:
        w = stats.wilcoxon(d).pvalue
    except Exception:
        w = float("nan")
    t = stats.ttest_rel(np.zeros(len(d)), d).pvalue
    return d.mean(), pos, len(d), sign, w, t


def main():
    M = ms.load_model()
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    face_lab = lb.face_labels(np.asarray(M["dd"]["f"]), vlabels)
    segs = list(lb.LEG_SEGMENTS)

    res = {}
    for a in ARMS:
        print(f"[run] {a} ...", flush=True)
        res[a] = run_arm(a, face_lab, vlabels, segs)

    print(f"\nmean leg-level ERROR rate by segment")
    hdr = "".join(f"{a:>14}" for a in ARMS)
    print(f"{'seg':<8}{hdr}")
    for s in segs + ["DISTAL", "ALL"]:
        cells = "".join(f"{np.nanmean([r[s] for r in res[a].values()]):>14.4f}" for a in ARMS)
        print(f"{s:<8}{cells}")

    print(f"\nregression check -- C13_uniform vs P48_cse_all (should be ~0)")
    dm, pos, n, sg, w, t = tests("P48_cse_all", "C13_uniform", res, "ALL")
    print(f"  ALL leg err  d={dm:+.4f}  {pos}/{n}  sign={sg:.3g} wilcox={w:.3g} t={t:.3g}")

    print(f"\nPRE-REGISTERED PRIMARY: co leg error vs C13_uniform")
    for arm in ["C13_invtol", "C13_invtol2"]:
        dm, pos, n, sg, w, t = tests("C13_uniform", arm, res, "co")
        verdict = ("PASS" if (dm <= -0.05 and sg < 0.05)
                   else "PARTIAL" if (dm < 0 and sg < 0.05) else "FAIL")
        print(f"  {arm:<12} d={dm:+.4f}  {pos}/{n} worse  sign={sg:.3g} wilcox={w:.3g} "
              f"t={t:.3g}   -> {verdict}")

    print(f"\nGUARD-RAILS vs C13_uniform")
    for arm in ["C13_invtol", "C13_invtol2"]:
        dd_, pos, n, sg, w, t = tests("C13_uniform", arm, res, "DISTAL")
        da, posa, na, sga, wa, ta = tests("C13_uniform", arm, res, "ALL")
        print(f"  {arm:<12} distal d={dd_:+.4f} (limit +0.02, {'OK' if dd_ <= 0.02 else 'TRADE'})"
              f"   overall leg_err d={da:+.4f} -> leg_acc {-da:+.4f} "
              f"({'OK' if da <= 0 else 'FAIL: leg_acc fell'})  sign={sga:.3g} wilcox={wa:.3g}")

    op = os.path.join(HERE, "out", "C13_per_specimen_20260828.json")
    with open(op, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
