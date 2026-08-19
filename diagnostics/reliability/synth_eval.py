"""Task 2, step 2 -- PRIMARY evaluation: does reliability weighting move fitted measurements
closer to known synthetic ground truth, and does it help low-reliability regions more?

Corpus: `diagnostics/moonshot/synth_{clean,noisy}` (12 + 12 specimens), each with its own
`ground_truth.npz`. Fitted outputs used: `diagnostics/moonshot/runs/SYN_{clean,noisy}_w5/
Stage_3_deform_fine.npz` -- `w5` is the folder name for the single w_offset=5.0/2.0 control arm
confirmed near-optimal in Task 3 (`diagnostics/offset_sweep_evidence/scorecard.md`); it is not
"w5 vs w2", it is one recipe per corpus.

Ground truth and fit go through the exact same `measure.measure_verts` code path (as
`calib_features.py` already established), and ground-truth verts are normalised identically to
how the fitter's own `load_meshes` would have normalised them (center + max-abs-coord scale),
using each specimen's own `.obj` -- copied verbatim from `calib_features.py` so this is not a
second, possibly-divergent implementation of the same normalisation.

The two corpora are POOLED into one n=24 evaluation set for computing composites/weights and for
the headline table (that is how this weighting would actually be used downstream: as one
reliability signal across whatever specimens are in a given run), with a per-corpus breakdown
also written to the output JSON for transparency.

Reporting standard (pre-registered, from the approved plan -- applied as-is): for every
scheme-vs-`equal` delta, per region, report the raw values, the delta, and a bootstrap CI
(2000 resamples of specimens with replacement). A fixed ~3-percentage-point-equivalent threshold
is a PRACTICAL-EFFECT SCREEN carried over from this project's established noise-floor convention,
not a significance test -- a delta can clear it and still have a CI crossing zero, and that is
reported as exactly that, not smoothed over.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ABS_SCALE = os.path.join(REPO, "diagnostics", "absolute_scale")
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, ABS_SCALE)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from calib_features import block_of  # noqa: E402
import weights as W  # noqa: E402

OUT = os.path.join(HERE, "out")
CORPORA = [("synth_clean", "SYN_clean_w5"), ("synth_noisy", "SYN_noisy_w5")]
N_BOOT = 2000
PRACTICAL_THRESHOLD = 0.03  # 3%-equivalent, this project's established noise-floor convention


def load_ground_truth(corpus_dir, labels, M, bones, tpa):
    """Ground-truth measurements for `labels`, normalised exactly as `load_meshes` would."""
    from pytorch3d.io import load_obj

    gt = np.load(os.path.join(MOON, corpus_dir, "ground_truth.npz"))
    gtv_all, names = gt["verts"], [str(x) for x in gt["names"]]
    gv = []
    for lab in labels:
        stem = lab[:-4] if lab.endswith(".obj") else lab
        ov, _, _ = load_obj(os.path.join(MOON, corpus_dir, f"{stem}.obj"), load_textures=False)
        ov = ov.numpy()
        c = ov.mean(0)
        gv.append((gtv_all[names.index(stem)] - c) / np.abs(ov - c).max())
    rows, cols, dev = ms.measure_verts(np.stack(gv), labels, "synth", M, bones, tpa)
    return rows, cols, dev


def collect():
    """NOTE: `synth_clean` and `synth_noisy` reuse the identical filenames `synth_000.obj`..
    `synth_011.obj` (same 12 synthetic identities under two noise conditions), so `label` alone is
    NOT a safe dict key once the two corpora are pooled -- `mesh_quality`/`part_quality` key their
    per-specimen output by `r["label"]`, and calling them on rows spanning both runs at once would
    silently let the second run's synth_000 overwrite the first's. Composites are therefore
    computed ONCE PER CORPUS (each call sees only that corpus's rows, so its internal label keys
    are unambiguous) and merged afterwards under a disambiguated `corpus::label` key ("uid")."""
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)

    fit_rows, gt_rows = [], []
    cols = dev = None
    global_composite, part_composite = {}, {r: {} for r in ms.PART_REGIONS}
    for corpus_dir, run in CORPORA:
        rf, c, d = ms.measure_run(run, "synth", M, bones, tpa)
        labels = [r["label"] for r in rf]
        rg, _, _ = load_ground_truth(corpus_dir, labels, M, bones, tpa)
        for r in rf:
            r["corpus"] = corpus_dir
            r["uid"] = f"{corpus_dir}::{r['label']}"
        for r in rg:
            r["corpus"] = corpus_dir
            r["uid"] = f"{corpus_dir}::{r['label']}"
        fit_rows += rf
        gt_rows += rg
        cols, dev = c, d
        print(f"{corpus_dir}: {len(rf)} specimens measured (fit + ground truth)")

        gcomp, _ = ms.quality_composite(rf, M)
        for r, z in zip(rf, gcomp):
            global_composite[r["uid"]] = float(z)
        pqc = ms.part_quality_composite(rf, M)
        for reg, d_ in pqc.items():
            for lab, z in d_.items():
                part_composite[reg][f"{corpus_dir}::{lab}"] = z

    print("global composite computed for", len(global_composite), "specimens")
    print("part composite regions:", {r: len(d) for r, d in part_composite.items()})
    return fit_rows, gt_rows, cols, dev, M, global_composite, part_composite


def region_columns(cols, dev):
    """{region: [columns]} using the SAME block_of() calib_features.py already uses."""
    out = {}
    for c in list(cols) + list(dev):
        out.setdefault(block_of(c), []).append(c)
    return out


def bootstrap_ci(fn, n, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot // 4:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def eval_scheme_region(labels, X_fit, X_gt, cols, region_cols, block_of_, scheme, global_composite, part_composite):
    """Weighted R / normalized-MAE / relerr for one scheme, one region, averaged over that
    region's columns (computed per-column on RAW values, never centered/standardized first).

    Bone lengths and part extents are strictly positive (see measure.py's `log_shape_ratios`
    docstring), so relative error `|x-y|/|y|` is well-defined per column without transformation.
    An earlier version of this script z-scored and mean-CENTERED values before computing relerr,
    which pushes many y values near zero and produces meaningless (>100%) relative errors -- fixed
    by keeping relerr on raw units. R is scale/location-invariant so is unaffected either way.
    MAE is reported NORMALIZED by that column's ground-truth SD (so columns of different natural
    scale, e.g. head_len vs mandible_wid, are comparable when averaged into one region figure).
    The region aggregate is the unweighted mean across that region's columns (each column's own
    R/MAE/relerr is already a `measurement_weights`-weighted statistic over specimens).
    """
    wmap = W.measurement_weights(labels, cols, block_of_, scheme, global_composite, part_composite)
    per_col = []
    for c in region_cols:
        ci = cols.index(c)
        x, y, w = X_fit[:, ci], X_gt[:, ci], wmap[c]
        gt_std = y.std()
        if gt_std < 1e-9:
            continue
        per_col.append((x, y, w, gt_std))
    if not per_col:
        return None
    n_cols = len(per_col)

    def stat(idx):
        Rs, MAEs, RELs = [], [], []
        for x, y, w, gt_std in per_col:
            xi, yi, wi = x[idx], y[idx], w[idx]
            Rs.append(W.weighted_pearson(xi, yi, wi))
            MAEs.append(W.weighted_mae(xi, yi, wi) / gt_std)
            RELs.append(W.weighted_relerr(xi, yi, wi))
        return float(np.nanmean(Rs)), float(np.nanmean(MAEs)), float(np.nanmean(RELs))

    r, mae, rel = stat(np.arange(len(labels)))

    r_ci = bootstrap_ci(lambda idx: stat(idx)[0], len(labels))
    mae_ci = bootstrap_ci(lambda idx: stat(idx)[1], len(labels))
    rel_ci = bootstrap_ci(lambda idx: stat(idx)[2], len(labels))
    point = dict(
        n_cols=n_cols,
        R=r,
        R_ci=r_ci,
        MAE=mae,
        MAE_ci=mae_ci,
        relerr=rel,
        relerr_ci=rel_ci,
        mean_weight=float(np.mean([w.mean() for _, _, w, _ in per_col])),
    )
    return point, stat


def main():
    os.makedirs(OUT, exist_ok=True)
    fit_rows, gt_rows, cols, dev, M, global_composite, part_composite = collect()
    all_cols = cols + dev
    labels = [r["uid"] for r in fit_rows]  # "uid" = corpus::label, disambiguated (see collect())
    gt_by_label = {r["uid"]: r for r in gt_rows}

    X_fit = np.array([[r[c] for c in all_cols] for r in fit_rows], dtype=np.float64)
    X_gt = np.array([[gt_by_label[lab][c] for c in all_cols] for lab in labels], dtype=np.float64)

    reg_cols = region_columns(cols, dev)
    print("\nregions:", {r: len(cs) for r, cs in reg_cols.items()})

    results = {}
    for reg, rcols in sorted(reg_cols.items()):
        results[reg] = {}
        for scheme in W.SCHEMES:
            out = eval_scheme_region(
                labels, X_fit, X_gt, all_cols, rcols, block_of, scheme, global_composite, part_composite
            )
            results[reg][scheme] = out  # (point_dict, stat_fn) or None

    # ---------------- report deltas vs `equal`, with a PAIRED bootstrap on the delta itself
    # (same resampled specimen indices fed to both the scheme and `equal`'s stat function on each
    # draw, so the CI is on scheme-minus-equal, not on the scheme's own R/MAE/relerr in isolation).
    print(
        f"\n{'region':<12}{'scheme':<20}{'R':>8}{'ΔR':>8}{'ΔR_CI':>16}"
        f"{'ΔMAE':>9}{'Δrel':>9}  clears {100 * PRACTICAL_THRESHOLD:.0f}pp?  ΔR_CI excl 0?"
    )
    summary = {}
    for reg, by_scheme in results.items():
        base = by_scheme.get("equal")
        if base is None:
            continue
        base_point, base_stat = base
        summary[reg] = {}
        for scheme, out in by_scheme.items():
            if out is None:
                continue
            res, stat_fn = out
            dR = res["R"] - base_point["R"]
            dMAE = res["MAE"] - base_point["MAE"]  # negative = improvement (lower error)
            drel = res["relerr"] - base_point["relerr"]

            def paired_delta(idx, stat_fn=stat_fn, base_stat=base_stat):
                sr, smae, srel = stat_fn(idx)
                br, bmae, brel = base_stat(idx)
                return sr - br

            dR_ci = bootstrap_ci(paired_delta, len(labels))
            ci_excl0 = np.isfinite(dR_ci[0]) and (dR_ci[0] > 0 or dR_ci[1] < 0)
            clears = abs(dR) >= PRACTICAL_THRESHOLD or abs(drel) >= PRACTICAL_THRESHOLD
            print(
                f"{reg:<12}{scheme:<20}{res['R']:>8.3f}{dR:>+8.3f}"
                f"   [{dR_ci[0]:+.3f},{dR_ci[1]:+.3f}]{dMAE:>+9.3f}{drel:>+9.3f}"
                f"  {'YES' if clears else 'no':<7}  {'YES' if ci_excl0 else 'no'}"
            )
            summary[reg][scheme] = dict(
                res,
                delta_R=dR,
                delta_R_ci=dR_ci,
                delta_R_ci_excludes_zero=bool(ci_excl0),
                delta_MAE=dMAE,
                delta_relerr=drel,
                clears_practical_threshold=bool(clears),
            )

    json.dump(summary, open(os.path.join(OUT, "synth_eval.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(OUT, 'synth_eval.json')}")


if __name__ == "__main__":
    main()
