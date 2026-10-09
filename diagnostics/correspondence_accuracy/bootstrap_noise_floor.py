"""Specimen-level bootstrap: how much can leg_acc / pt->ta / ti->ta shift by resampling noise
alone, on n=12 specimens?

WHY (pre-registered 2026-08-19, BEFORE HIER_split_segments' results were seen): the ~1.5-point
bar informally used to call GNC/split_distal/stratified sampling "null" in REPORT.md was never
checked against an actual noise floor. Pretarsus has only 7-8 dominant vertices and n=12
specimens is small -- a threshold picked by eye is exactly the kind of thing that's easy to
rationalize AFTER seeing a result ("well THIS shift feels real"). Computing the floor now, on
arms whose results are already fixed and written to disk, closes that door.

METHOD: a specimen-level (cluster/block) bootstrap, not a point-level one -- points sampled
from the SAME specimen are correlated (same fit, same local errors), so resampling points
directly would understate the true uncertainty. For B=1000 draws: resample the 12 specimens
WITH REPLACEMENT (same draw of specimen indices applied to every arm being compared, so
between-arm differences are PAIRED -- this is what actually answers "is a shift between two
arms bigger than noise", not just "how noisy is one arm alone"), re-pool each resampled
specimen's already-computed per-specimen confusion counts / geodesic distance lists, and
recompute the same summary statistics run_audit.py reports. The per-specimen quantities
themselves are computed ONCE with run_audit.py's exact functions (not reimplemented) so this
measures resampling noise only, not a different metric.
"""

import os
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj

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
import geodesic as geo  # noqa: E402

N_SAMPLE = 8000
DEVICE = "cpu"
N_BOOT = 1000

ARMS = [
    ("SYN_clean_w5", "synth_clean", None),
    ("SYN_clean_pose25_gnclegonly_w5", "synth_clean", None),
    ("SYN_clean_pose25_splitdistal_w5", "synth_clean", None),
    # gtinit was cited as the metric's OWN validation story ("wins every column") -- that claim
    # rested on the same point-estimate comparison that turned out to be noise for the other
    # three arms, so it needs the same paired-bootstrap check, not an exemption because its
    # margins looked large.
    ("SYN_clean_pose25_gtinit_w5", "synth_clean", None),
]


def per_specimen_stats(run, corpus, npz_path, template_faces, face_lab, vlabels, n_verts_template, geo_lookup, rng):
    """One entry per specimen: leg confusion matrix, within-leg-segment confusion matrix, and
    raw geodesic-distance lists for the pt->ta / ti->ta pairs -- everything a resample needs to
    re-pool without resampling points themselves.
    """
    d = np.load(npz_path)
    spec_labels = [str(x) for x in d["labels"]]
    fitted = d["verts"].astype(np.float64)
    assert fitted.shape[1] == n_verts_template

    out = []
    for i, spec_lab in enumerate(spec_labels):
        stem = spec_lab[:-4] if spec_lab.endswith(".obj") else spec_lab
        obj_path = os.path.join(MOON, corpus, f"{stem}.obj")
        ov, of, _ = load_obj(obj_path, load_textures=False)
        tgt_verts, tgt_faces = ov.numpy(), of.verts_idx.numpy()
        assert tgt_faces.shape == template_faces.shape

        pts, point_lab = cf.sample_true_labeled_points(tgt_verts, tgt_faces, face_lab, N_SAMPLE, rng)
        pts_t = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        true_is_leg = point_lab["leg_id"] != None  # noqa: E711
        fitted_i = torch.tensor(fitted[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        leg_acc_i, leg_pred, pts_leg, true_leg_sub = cf.leg_confusion(
            point_lab["leg_id"], true_is_leg, pts_t, fitted_i, vlabels
        )
        seg_acc_i = cf.within_leg_segment_confusion(
            point_lab["leg_seg"], true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_i, vlabels
        )
        true_vidx_here, matched_vidx, legs_here = cf.within_leg_segment_geodesic(
            point_lab["true_vertex_idx"], true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_i, vlabels
        )

        pt_ta, ti_ta = [], []
        if len(true_vidx_here):
            consistent = vlabels["leg_id"][true_vidx_here] == legs_here
            true_vidx_here, matched_vidx, legs_here = (
                true_vidx_here[consistent], matched_vidx[consistent], legs_here[consistent]
            )
        if len(true_vidx_here):
            dists = geo_lookup.dist(legs_here, true_vidx_here, matched_vidx)
            true_seg_of_point = vlabels["leg_seg"][true_vidx_here]
            matched_seg_of_point = vlabels["leg_seg"][matched_vidx]
            for ts, ms_, dd_ in zip(true_seg_of_point, matched_seg_of_point, dists):
                if np.isnan(dd_):
                    continue
                if ts == "pt" and ms_ == "ta":
                    pt_ta.append(dd_)
                elif ts == "ti" and ms_ == "ta":
                    ti_ta.append(dd_)

        out.append(dict(
            stem=stem,
            leg_M=leg_acc_i.M.copy(),
            seg_M=seg_acc_i.M.copy(),
            pt_ta=np.array(pt_ta, dtype=np.float64),
            ti_ta=np.array(ti_ta, dtype=np.float64),
        ))
    return out


def pooled_stat(specimen_subset):
    leg_M = sum(s["leg_M"] for s in specimen_subset)
    seg_M = sum(s["seg_M"] for s in specimen_subset)
    leg_acc = np.trace(leg_M) / max(leg_M.sum(), 1)
    seg_acc = np.trace(seg_M) / max(seg_M.sum(), 1)
    pt_ta = np.concatenate([s["pt_ta"] for s in specimen_subset])
    ti_ta = np.concatenate([s["ti_ta"] for s in specimen_subset])
    return dict(
        leg_acc=float(leg_acc),
        seg_acc=float(seg_acc),
        pt_ta_mean=float(pt_ta.mean()) if len(pt_ta) else float("nan"),
        ti_ta_mean=float(ti_ta.mean()) if len(ti_ta) else float("nan"),
    )


def bootstrap(all_arm_specimens, n_boot, seed):
    """all_arm_specimens: {arm_name: [per-specimen dict, ...] (same length, same specimen order
    across arms)}. Returns {arm_name: {stat: bootstrap array (n_boot,)}} using the SAME
    resampled specimen indices for every arm at each draw (paired bootstrap).
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(all_arm_specimens.values())))
    boot = {arm: {k: np.zeros(n_boot) for k in ("leg_acc", "seg_acc", "pt_ta_mean", "ti_ta_mean")} for arm in all_arm_specimens}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for arm, specimens in all_arm_specimens.items():
            subset = [specimens[i] for i in idx]
            stat = pooled_stat(subset)
            for k, v in stat.items():
                boot[arm][k][b] = v
    return boot


def summarize(boot_arr):
    return dict(
        mean=float(np.nanmean(boot_arr)),
        std=float(np.nanstd(boot_arr)),
        ci95=(float(np.nanpercentile(boot_arr, 2.5)), float(np.nanpercentile(boot_arr, 97.5))),
    )


def main():
    M = ms.load_model()
    template_faces = np.asarray(M["dd"]["f"])
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    face_lab = lb.face_labels(template_faces, vlabels)
    n_verts_template = len(M["dominant"])
    v_template = np.asarray(M["v_template"], dtype=np.float64)
    geo_tables = geo.load_or_build_leg_tables(v_template, template_faces, vlabels)
    geo_lookup = geo.LegGeodesicLookup(geo_tables)

    all_arm_specimens = {}
    for run, corpus, npz_path in ARMS:
        if npz_path is None:
            npz_path = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(npz_path):
            print(f"[skip] {run}: not found")
            continue
        rng = np.random.default_rng(1)  # SAME seed run_audit.py used, so point samples match
        print(f"[compute per-specimen] {run} ...")
        all_arm_specimens[run] = per_specimen_stats(
            run, corpus, npz_path, template_faces, face_lab, vlabels, n_verts_template, geo_lookup, rng
        )

    point_estimate = {arm: pooled_stat(specs) for arm, specs in all_arm_specimens.items()}
    print("\n=== point estimate (all 12 specimens, matches run_audit.py) ===")
    for arm, stat in point_estimate.items():
        print(f"{arm:<34}leg_acc={stat['leg_acc']:.3f}  seg_acc={stat['seg_acc']:.3f}  "
              f"pt_ta={stat['pt_ta_mean']:.4f}  ti_ta={stat['ti_ta_mean']:.4f}")

    print(f"\n=== bootstrap, B={N_BOOT}, specimen-level resample with replacement (n=12) ===")
    boot = bootstrap(all_arm_specimens, N_BOOT, seed=0)
    for arm in all_arm_specimens:
        print(f"\n-- {arm} --")
        for k in ("leg_acc", "seg_acc", "pt_ta_mean", "ti_ta_mean"):
            s = summarize(boot[arm][k])
            print(f"  {k:<14} mean={s['mean']:.4f}  std={s['std']:.4f}  95% CI=({s['ci95'][0]:.4f}, {s['ci95'][1]:.4f})  "
                  f"half-width={0.5 * (s['ci95'][1] - s['ci95'][0]):.4f}")

    # paired differences between the first arm (treated as baseline) and every other arm --
    # this is the number that actually answers "is arm X's shift bigger than resampling noise"
    arms = list(all_arm_specimens.keys())
    if len(arms) >= 2:
        base = arms[0]
        print(f"\n=== paired bootstrap differences vs {base} (same resample draw both arms) ===")
        for other in arms[1:]:
            for k in ("leg_acc", "seg_acc", "pt_ta_mean", "ti_ta_mean"):
                diff = boot[other][k] - boot[base][k]
                s = summarize(diff)
                point_diff = point_estimate[other][k] - point_estimate[base][k]
                print(f"  {other} - {base}  {k:<14} point_diff={point_diff:+.4f}  "
                      f"boot_mean={s['mean']:+.4f}  95% CI=({s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f})")

    OUT = os.path.join(HERE, "out")
    os.makedirs(OUT, exist_ok=True)
    import json
    summary = {
        "point_estimate": point_estimate,
        "bootstrap_summary": {arm: {k: summarize(boot[arm][k]) for k in boot[arm]} for arm in boot},
    }
    with open(os.path.join(OUT, "bootstrap_noise_floor.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {OUT}/bootstrap_noise_floor.json")


if __name__ == "__main__":
    main()
