"""Phase 1a -- does the leg_distal length failure (R=0.136) come from bad JOINT positions?

WHAT THIS ANSWERS
`measure.py` derives every bone length from `J = Jr @ verts`, a weighted average over a joint's
dominantly-skinned vertices. If the fitted J for distal-leg joints (tibia/tarsus/pretarsus) is
itself badly localized relative to ground truth, then ANY length built from those joint
positions -- geodesic, chain-summed, ratio, whatever -- inherits that error, and no measurement
DEFINITION can fix it (the fault is upstream, in correspondence recovery, not in how length is
computed from positions). If instead individual joint positions are recovered well but bone
LENGTHS (differences of two joints) are not, the problem is specific to the two-point-difference
definition and a different definition could plausibly help. This probe distinguishes the two by
computing per-joint 3D position error directly, not just bone-length error.

METHOD
Reuses the exact ceiling-test pair this project already uses for R=0.136: the `SYN_clean_w5` fit
(`diagnostics/moonshot/runs/SYN_clean_w5/Stage_3_deform_fine.npz`) against
`diagnostics/moonshot/synth_clean/ground_truth.npz`. Ground truth is normalised EXACTLY as
`calib_features.py` does it (per-specimen recentre + divide by max|coord| of the ORIGINAL target
.obj, not the ground-truth array itself) so fitted and true joints are in the same frame -- this
is copied, not reimplemented, to avoid a second, possibly inconsistent, normalisation.

Per joint: error_i = || J_fitted[s,i] - J_true[s,i] ||, over specimens s. Reported both in raw
model units and normalised by that specimen's overall extent (from ground_truth.npz's `extent`
field), since raw units are only meaningful relative to specimen scale.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from calib_features import block_of  # noqa: E402  -- reuse the EXACT grouping that produced R=0.136
from pytorch3d.io import load_obj  # noqa: E402

RUN = os.environ.get("PROBE_RUN", "SYN_clean_w5")
CORPUS = os.environ.get("PROBE_CORPUS", "synth_clean")
OUT = os.path.join(HERE, "out")


def main():
    os.makedirs(OUT, exist_ok=True)
    M = ms.load_model()
    jn = M["jnames"]
    Jr = M["Jr"]
    counts = np.bincount(M["dominant"], minlength=len(jn))

    d = np.load(os.path.join(MOON, "runs", RUN, "Stage_3_deform_fine.npz"))
    labels = [str(x) for x in d["labels"]]
    fitted_verts = d["verts"].astype(np.float64)  # (n, V, 3)

    gt = np.load(os.path.join(MOON, CORPUS, "ground_truth.npz"))
    gtv_all, names, ext = gt["verts"], [str(x) for x in gt["names"]], gt["extent"]

    gt_verts = []
    for lab in labels:
        stem = lab[:-4] if lab.endswith(".obj") else lab
        ov, _, _ = load_obj(os.path.join(MOON, CORPUS, f"{stem}.obj"), load_textures=False)
        ov = ov.numpy()
        c = ov.mean(0)
        gt_verts.append((gtv_all[names.index(stem)] - c) / np.abs(ov - c).max())
    gt_verts = np.stack(gt_verts)  # same normalisation convention as calib_features.py:104-109

    J_fit = ms.joints(fitted_verts, Jr)  # (n, J, 3)
    J_gt = ms.joints(gt_verts, Jr)

    # No further per-specimen rescaling: both `fitted_verts` and `gt_verts` are ALREADY in the
    # same per-specimen normalised frame -- `gt_verts` was built above by matching each specimen's
    # own target .obj centre/max|coord| (calib_features.py's own convention), and `fitted_verts`
    # is that same target's fit output, so raw joint-position error is already comparable across
    # specimens without inventing a second normalisation.
    err = np.linalg.norm(J_fit - J_gt, axis=-1)  # (n, J)
    err_norm = err

    def group(name):
        if name.startswith("l_"):
            seg = name.split("_")[2]
            return "leg_distal" if seg in ("ti", "ta", "pt") else "leg_prox"
        if name in ("b_h", "b_t"):
            return "body_axis"
        if name.startswith("b_a_"):
            return "gaster"
        if name.startswith("an_"):
            return "antenna"
        if name.startswith("ma"):
            return "mandible"
        if name.startswith("w_"):
            return "wing"
        return "other"

    rows = []
    for j, name in enumerate(jn):
        g = group(name)
        if g == "wing":
            continue
        rows.append(
            dict(
                joint=name,
                group=g,
                support=int(counts[j]),
                median_err_raw=float(np.median(err[:, j])),
                median_err_norm=float(np.median(err_norm[:, j])),
                mean_err_norm=float(np.mean(err_norm[:, j])),
                std_err_norm=float(np.std(err_norm[:, j])),
            )
        )

    rows.sort(key=lambda r: (r["group"], -r["median_err_norm"]))

    print(f"{'joint':<12}{'group':<12}{'support':>8}{'median|err|(norm)':>20}{'mean':>10}{'std':>10}")
    for r in rows:
        print(
            f"{r['joint']:<12}{r['group']:<12}{r['support']:>8}"
            f"{r['median_err_norm']:>20.5f}{r['mean_err_norm']:>10.5f}{r['std_err_norm']:>10.5f}"
        )

    print("\n=== per-group summary (median across joints in group, of each joint's median error) ===")
    groups = sorted(set(r["group"] for r in rows))
    summary = {}
    for g in groups:
        m = [r for r in rows if r["group"] == g]
        med = float(np.median([r["median_err_norm"] for r in m]))
        supp = float(np.median([r["support"] for r in m]))
        summary[g] = dict(n_joints=len(m), median_of_median_err_norm=med, median_support=supp)
        print(f"  {g:<12} n={len(m):>3}  median support={supp:>7.1f}  median err_norm={med:.5f}")

    # correlation between support (vertex count) and error, across ALL non-wing joints --
    # direct test of the "low support -> high localisation error" hypothesis, not asserted.
    supports = np.array([r["support"] for r in rows], dtype=float)
    errs = np.array([r["median_err_norm"] for r in rows])
    valid = supports > 0
    rho = float(np.corrcoef(np.log(supports[valid]), np.log(errs[valid]))[0, 1])
    print(f"\ncorr(log support, log median joint-position error) across {valid.sum()} joints: r={rho:.3f}")

    # ---- does joint-position error explain BONE-LENGTH R, and is it outlier-driven? ----
    # Median distal-joint position error is only ~1.5x proximal (0.0255 vs 0.0175 above) -- far
    # too small a gap to explain R falling from ~0.75 to 0.14 on its own if errors were uniform
    # across specimens. But the STD/mean >> median pattern for distal joints (e.g. l_1_pt_l:
    # median 0.091, mean 0.306) signals a HEAVY TAIL -- a few catastrophically mislocalized
    # specimens, not uniformly-worse localization. Test this directly: for each leg_distal bone,
    # compute R with all 12 specimens vs R with the single worst-error specimen for that bone's
    # endpoint joints removed (leave-one-out on the outlier, not a general LOO).
    bones = ms.bone_table(M)
    B_fit = ms.bone_lengths(J_fit, bones)
    B_gt = ms.bone_lengths(J_gt, bones)
    print("\n=== bone-length R: full n=12 vs worst-outlier-removed (leg_distal bones) ===")
    print(f"{'bone':<14}{'R (n=12)':>10}{'R (n=11, outlier dropped)':>28}{'worst specimen err':>22}")
    outlier_rows = []
    for bi, b in enumerate(bones):
        if block_of(b["name"]) != "leg_distal":
            continue
        f, g = B_fit[:, bi], B_gt[:, bi]
        r_full = float(np.corrcoef(f, g)[0, 1]) if f.std() > 1e-12 else np.nan
        # the specimen with the largest position error on EITHER endpoint of this bone
        j_child, j_parent = b["child"], b["parent"]
        joint_err = np.maximum(err[:, j_child], err[:, j_parent])
        worst = int(np.argmax(joint_err))
        keep = np.arange(len(f)) != worst
        r_drop = float(np.corrcoef(f[keep], g[keep])[0, 1]) if f[keep].std() > 1e-12 else np.nan
        print(f"{b['name']:<14}{r_full:>10.3f}{r_drop:>28.3f}{joint_err[worst]:>22.4f}")
        outlier_rows.append(
            dict(bone=b["name"], r_full=r_full, r_outlier_dropped=r_drop, worst_specimen=labels[worst],
                 worst_specimen_endpoint_err=float(joint_err[worst]))
        )

    json.dump(
        dict(
            run=RUN,
            corpus=CORPUS,
            rows=rows,
            group_summary=summary,
            log_support_vs_err_r=rho,
            leg_distal_outlier_test=outlier_rows,
        ),
        open(os.path.join(OUT, f"probe_1a_joint_localization_{RUN}.json"), "w"),
        indent=1,
    )
    print(f"\nwrote {OUT}/probe_1a_joint_localization_{RUN}.json")


if __name__ == "__main__":
    main()
