"""Rank 3 (master prompt Section 8 row 9 / Section 9): intermediate pose_scale sweep analysis.

NOT a new fitting job. The fits already exist (diagnostics/registration_failure/runs/hier_SYN_
clean_pose{0,05,10,15,20,25,35}_w5, produced by Task 6) and the correspondence audit across all
seven pose levels already exists at diagnostics/registration_failure/out/d2b_correspondence_audit
.json (produced by probe_d2b_correspondence_audit.py, which already sweeps pose0..pose35 plus
drop30/drop60 -- this was not evident from the master prompt's own text, which asked for this
sweep as if it were new). This script only ADDS the one thing the existing audit does not
compute: per-condition leg_distal ground-truth R, joined against the existing cross-leg and
within-leg mismatch numbers, to make the H2-vs-H4 separation test the master prompt asks for:

  H2 (pose-INDEPENDENT within-leg segment ambiguity) predicts within_leg_mismatch ~ FLAT across
     pose_scale.
  H4 (pose-DEPENDENT cross-leg confusion) predicts cross_leg_confusion MONOTONIC in pose_scale.

Falsifier for the mechanism separation asserted in Section 6/10: if within_leg_mismatch moves
comparably to cross_leg_confusion across the sweep, they are not separable mechanisms after all.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ABS_SCALE = os.path.join(REPO, "diagnostics", "absolute_scale")
REG_FAIL = os.path.join(REPO, "diagnostics", "registration_failure")
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, ABS_SCALE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from calib_features import block_of  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# (run, corpus, pose_scale) -- pose_scale values per diagnostics/moonshot/make_synth_corpus.py
# naming convention (pose25 == the un-suffixed default corpus, confirmed in Rank 1/rank2 work)
CONDITIONS = [
    ("SYN_clean_pose0_w5", "synth_clean_pose0", 0.00),
    ("SYN_clean_pose05_w5", "synth_clean_pose05", 0.05),
    ("SYN_clean_pose10_w5", "synth_clean_pose10", 0.10),
    ("SYN_clean_pose15_w5", "synth_clean_pose15", 0.15),
    ("SYN_clean_pose20_w5", "synth_clean_pose20", 0.20),
    ("SYN_clean_w5", "synth_clean", 0.25),
    ("SYN_clean_pose35_w5", "synth_clean_pose35", 0.35),
]


def load_gt_verts(corpus_dir):
    from pytorch3d.io import load_obj

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


def cohort_leg_distal_R(run, corpus, M, bones, tpa):
    """Pooled Pearson R (fit vs GT) over all leg_distal bone-length columns x all specimens in
    this run -- same convention as this project's other cohort-level R reports."""
    fit_rows, _, _ = ms.measure_run(run, "synth", M, bones, tpa)
    if not fit_rows:
        return None, 0
    gt_verts = load_gt_verts(corpus)
    labels = list(gt_verts.keys())
    gt_rows, _, _ = ms.measure_verts(
        np.stack([gt_verts[l] for l in labels]).astype(np.float64), labels, "synth", M, bones, tpa
    )
    gt_by_label = {r["label"]: r for r in gt_rows}
    ld_bones = [b["name"] for b in bones if block_of(b["name"]) == "leg_distal"]

    x, y = [], []
    for r in fit_rows:
        stem = r["label"][:-4] if r["label"].endswith(".obj") else r["label"]
        g = gt_by_label.get(stem)
        if g is None:
            continue
        for b in ld_bones:
            if b in r and b in g:
                x.append(r[b])
                y.append(g[b])
    if len(x) < 3:
        return None, len(x)
    r, _ = pearsonr(x, y)
    return float(r), len(x)


def main():
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)

    d2b = json.load(open(os.path.join(REG_FAIL, "out", "d2b_correspondence_audit.json")))
    by_run = {}
    for r in d2b:
        by_run.setdefault(r["run"], []).append(r)

    rows = []
    print(f"{'pose_scale':>10}  {'leg_distal_R':>13}  {'cross_leg(ti/ta/pt)':>22}  {'within_mismatch(ti/ta/pt)':>28}")
    for run, corpus, pose_scale in CONDITIONS:
        R, n = cohort_leg_distal_R(run, corpus, M, bones, tpa)
        specs = by_run.get(run, [])
        cross = {seg: float(np.nanmean([s[f"cross_leg_{seg}"] for s in specs])) for seg in ("ti", "ta", "pt")} if specs else {}
        within = {seg: float(np.nanmean([s[f"within_leg_mismatch_{seg}"] for s in specs])) for seg in ("ti", "ta", "pt")} if specs else {}
        rows.append(dict(run=run, corpus=corpus, pose_scale=pose_scale, leg_distal_R=R, n_bone_obs=n,
                          cross_leg_confusion=cross, within_leg_mismatch=within,
                          n_specimens_in_audit=len(specs)))
        cross_s = "/".join(f"{cross.get(s, float('nan')):.3f}" for s in ("ti", "ta", "pt"))
        within_s = "/".join(f"{within.get(s, float('nan')):.3f}" for s in ("ti", "ta", "pt"))
        print(f"{pose_scale:>10.2f}  {R if R is not None else float('nan'):>13.3f}  {cross_s:>22}  {within_s:>28}")

    pose = np.array([r["pose_scale"] for r in rows])
    cross_pt = np.array([r["cross_leg_confusion"].get("pt", np.nan) for r in rows])
    within_pt = np.array([r["within_leg_mismatch"].get("pt", np.nan) for r in rows])
    R_vals = np.array([r["leg_distal_R"] if r["leg_distal_R"] is not None else np.nan for r in rows])

    def spear(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return None, None
        rho, p = spearmanr(a[m], b[m])
        return float(rho), float(p)

    rho_cross, p_cross = spear(pose, cross_pt)
    rho_within, p_within = spear(pose, within_pt)
    rho_R, p_R = spear(pose, R_vals)

    print(f"\nSpearman(pose_scale, cross_leg_confusion[pt])  rho={rho_cross}  p={p_cross}")
    print(f"Spearman(pose_scale, within_leg_mismatch[pt])  rho={rho_within}  p={p_within}")
    print(f"Spearman(pose_scale, leg_distal_R)              rho={rho_R}  p={p_R}")

    H4_supported = rho_cross is not None and rho_cross > 0.5 and p_cross is not None and p_cross < 0.10
    H2_supported = rho_within is not None and abs(rho_within) < 0.4  # flat = weak/no monotonic relationship

    interpretation = (
        f"H4 (pose-dependent cross-leg confusion): {'SUPPORTED' if H4_supported else 'NOT CLEARLY SUPPORTED'} "
        f"(rho={rho_cross}, p={p_cross}). "
        f"H2 (pose-independent within-leg mismatch): {'SUPPORTED' if H2_supported else 'NOT CLEARLY SUPPORTED'} "
        f"(rho={rho_within}, p={p_within}). "
        + (
            "Mechanism separation (Section 6/10) HOLDS: cross-leg confusion tracks pose severity while "
            "within-leg mismatch does not -- these are two distinct failure modes, not one."
            if H4_supported and H2_supported
            else "Mechanism separation is NOT cleanly confirmed by this sweep -- see per-condition numbers above; "
            "do not assert the H2/H4 split as settled on this evidence alone."
        )
    )
    print(f"\n{interpretation}")

    out = dict(conditions=rows,
               spearman_pose_vs_cross_leg_pt=dict(rho=rho_cross, p=p_cross),
               spearman_pose_vs_within_leg_pt=dict(rho=rho_within, p=p_within),
               spearman_pose_vs_leg_distal_R=dict(rho=rho_R, p=p_R),
               interpretation=interpretation)
    with open(os.path.join(OUT, "rank3_pose_sweep.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'rank3_pose_sweep.json')}")


if __name__ == "__main__":
    main()
