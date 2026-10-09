"""B3b -- direct measurement AGREEMENT against an independent physical reference (20 Atta).

WHY THIS EXISTS, beyond the allometry check already in mv_framework.py:
    "SMILify reproduces the expected biological scaling" is evidence of validity but it is NOT a
    measurement-accuracy test -- R^2 = 0.99 coexists happily with large systematic bias, and
    morphometric methodology treats biological signal and measurement error as separate things that
    must be established separately. This module measures AGREEMENT: bias, limits of agreement,
    proportional bias, and Lin's concordance -- not correlation.

CIRCULARITY, STATED UP FRONT BECAUSE IT DETERMINES WHICH ROWS MEAN ANYTHING:
    Model traits live in a per-specimen normalised frame, so physical scale is restored with
    scale_i = BL_mm_i / BL_model_i. That factor is DERIVED FROM body length, therefore:
      * BL agreement is exactly circular -- reproduces the reference by construction. Reported
        only as a arithmetic check that the conversion is applied correctly; it is not evidence.
      * HW-in-mm is PARTIALLY circular: its scale comes from BL, only its shape does not.
      * HW/BL as a dimensionless RATIO is fully scale-free -- the factor cancels. THIS is the
        genuinely independent agreement test, and it is the row to read.

Two head-width definitions are compared, because "head width" is not one thing:
    HW_ext      Type III max mediolateral extent of the head part (trait_extract's HW)
    HW_reporter b_h_l--b_h_r reporter-bone distance (the Atta replication's definition)
"""
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "groundtruth"))

import trait_extract as TX                                        # noqa: E402
from measure import load_model                                     # noqa: E402
from fitter_3d.trainer import SMAL3DFitter                         # noqa: E402
from fitter_3d.joint_limits import build_head_width_reporter_matrix  # noqa: E402

NPZ = os.path.join(REPO, "diagnostics/atta_reference/blender_bundle/ATTA20_ARM_A.npz")
REF = os.path.join(REPO, "diagnostics/atta_reference/measurements/"
                         "ant_body_lengths_and_estimated_head_widths.csv")


def agreement(model, ref, name, circular=""):
    """Bland-Altman agreement statistics. Deliberately NOT led by correlation."""
    model, ref = np.asarray(model, float), np.asarray(ref, float)
    diff = model - ref
    mean = (model + ref) / 2.0
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    n = len(diff)
    # proportional bias: does the error grow with the size of the thing measured?
    sl, ic, r, p, se = stats.linregress(mean, diff)
    # Lin's concordance correlation coefficient -- agreement, not association
    mx, my = model.mean(), ref.mean()
    vx, vy = model.var(ddof=0), ref.var(ddof=0)
    cov = ((model - mx) * (ref - my)).mean()
    ccc = float(2 * cov / (vx + vy + (mx - my) ** 2))
    rel = np.abs(diff) / ref * 100.0
    tcrit = stats.t.ppf(0.975, n - 1)
    return dict(
        trait=name, n=n, circularity=circular,
        bias=bias, bias_ci=[float(bias - tcrit * sd / np.sqrt(n)),
                            float(bias + tcrit * sd / np.sqrt(n))],
        bias_pct=float(bias / ref.mean() * 100),
        loa_lower=float(bias - 1.96 * sd), loa_upper=float(bias + 1.96 * sd),
        mae=float(np.abs(diff).mean()), mean_abs_rel_pct=float(rel.mean()),
        median_abs_rel_pct=float(np.median(rel)), max_abs_rel_pct=float(rel.max()),
        proportional_bias_slope=float(sl), proportional_bias_p=float(p),
        proportional_bias_present=bool(p < 0.05),
        pearson_r=float(stats.pearsonr(model, ref)[0]), ccc=ccc,
        bland_altman=dict(mean=[float(x) for x in mean], diff=[float(x) for x in diff]))


def main():
    M = load_model()
    jn = M["jnames"]
    z = np.load(NPZ, allow_pickle=True)
    n = z["betas"].shape[0]
    f = SMAL3DFitter(batch_size=n, device="cpu")
    t = lambda k: torch.tensor(z[k], dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        verts = f(betas=t("betas"), global_rot=t("global_rot"), joint_rot=t("joint_rot"),
                  trans=t("trans"), log_beta_scales=t("log_beta_scales"),
                  betas_trans=t("betas_trans"), deform_verts=t("deform_verts"))
    R = build_head_width_reporter_matrix(M["v_template"], dtype=torch.float32)
    hw_rep_model = torch.linalg.norm(
        (torch.einsum("jv,bvc->bjc", R, verts))[:, 0]
        - (torch.einsum("jv,bvc->bjc", R, verts))[:, 1], dim=-1).numpy().astype(float)
    V = verts.numpy().astype(np.float64)
    J = np.einsum("ij,njk->nik", M["Jr"], V)
    bl_model = np.linalg.norm(J[:, jn.index("b_t")] - J[:, jn.index("b_a_5")], axis=-1)
    tr = TX.traits(V, M=M)
    hw_ext_model = np.asarray(tr["HW"], float)

    ref = {}
    with open(REF) as fh:
        for row in csv.DictReader(fh):
            ref[int(row["Shape"])] = (float(row["b_t to b_a_5 [mm]"]),
                                      float(row["head width [mm]"]), float(row["mass [mg]"]))
    labels = [str(x).replace("_processed.obj", "").replace(".obj", "") for x in z["labels"]]
    idx = [i for i, l in enumerate(labels) if int(l.split("_")[0]) in ref]
    keys = [int(labels[i].split("_")[0]) for i in idx]
    bl_ref = np.array([ref[k][0] for k in keys])
    hw_ref = np.array([ref[k][1] for k in keys])

    scale = bl_ref / bl_model[idx]                      # mm per model unit, per specimen

    rows = [
        agreement(bl_model[idx] * scale, bl_ref, "BL (body length, mm)",
                  "EXACTLY CIRCULAR -- scale factor is defined from this; arithmetic check only"),
        agreement(hw_ext_model[idx] * scale, hw_ref, "HW_ext (Type III extent, mm)",
                  "PARTIALLY circular -- scale from BL, shape independent"),
        agreement(hw_rep_model[idx] * scale, hw_ref, "HW_reporter (b_h_l-b_h_r, mm)",
                  "PARTIALLY circular -- scale from BL, shape independent"),
        agreement(hw_ext_model[idx] / bl_model[idx], hw_ref / bl_ref,
                  "HW_ext / BL (dimensionless ratio)", "SCALE-FREE -- fully independent"),
        agreement(hw_rep_model[idx] / bl_model[idx], hw_ref / bl_ref,
                  "HW_reporter / BL (dimensionless ratio)", "SCALE-FREE -- fully independent"),
    ]

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "mv_agreement.json"), "w") as fh:
        json.dump({"n": len(idx), "rows": rows}, fh, indent=2)

    print(f"n = {len(idx)} Atta specimens\n")
    hdr = (f"{'measurement':36s}{'bias':>9s}{'bias%':>8s}{'LoA':>20s}"
           f"{'meanRel%':>10s}{'CCC':>7s}{'propBias':>10s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        loa = f"[{r['loa_lower']:+.4f},{r['loa_upper']:+.4f}]"
        pb = "YES p=%.3f" % r["proportional_bias_p"] if r["proportional_bias_present"] else "no"
        print(f"{r['trait']:36s}{r['bias']:>+9.4f}{r['bias_pct']:>+8.2f}{loa:>20s}"
              f"{r['mean_abs_rel_pct']:>10.2f}{r['ccc']:>7.3f}{pb:>10s}")
    print("\nCircularity (read this before the numbers):")
    for r in rows:
        print(f"  {r['trait']:36s} {r['circularity']}")
    print(f"\nwrote {os.path.join(OUT, 'mv_agreement.json')}")


if __name__ == "__main__":
    main()
