"""Task 2 -- specimen- and part-aware registration reliability weights.

Four schemes, all built ONLY from `measure.mesh_quality`/`part_quality` (deform/edge/normal
composites) -- never from genus/species/subfamily, so the weight cannot leak taxonomy into the
measurements it is meant to protect:

  hard50            existing production behaviour: boolean, keep the best 50% by GLOBAL composite.
  equal             weight = 1 everywhere (no filtering) -- the baseline every scheme is compared to.
  continuous_global one weight per SPECIMEN, from its global composite.
  continuous_part   one weight per MEASUREMENT, from the composite of the anatomical region that
                     measurement is read off (via `calib_features.block_of`), falling back to the
                     specimen's global weight where a region lacks enough support to score.

The z-score -> weight transform is RANK-BASED (percentile rank, linear, monotonic) and has no
free parameters (no logistic k/c, no bandwidth) -- chosen specifically so the weighting function
is frozen before any fitted-vs-ground-truth comparison is run (see diagnostics/reliability/
REPORT.md and the approved plan). It is inspected only for numerical sanity (not degenerate/
all-tied) on the real composite distribution, never fit to maximise agreement with ground truth.
"""

import numpy as np

SCHEMES = ("hard50", "equal", "continuous_global", "continuous_part")


def rank_weight(values):
    """values: 1D array, HIGHER = WORSE (as `mesh_quality`/`part_quality` composites are).

    Returns weight in [0, 1], linear in rank: the best (lowest) value gets weight 1, the worst
    gets weight 0, ties broken by average rank. No free parameters.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= 1:
        return np.ones(n)
    order = values.argsort(kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # average rank for ties
    uniq, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        rank_sum = np.zeros(len(uniq))
        rank_cnt = np.zeros(len(uniq))
        np.add.at(rank_sum, inv, ranks)
        np.add.at(rank_cnt, inv, 1)
        ranks = (rank_sum / rank_cnt)[inv]
    return 1.0 - ranks / (n - 1)


def hard_filter_weight(values, top_pct=50):
    values = np.asarray(values, dtype=np.float64)
    thresh = np.percentile(values, top_pct)
    return (values <= thresh).astype(np.float64)


def measurement_weights(labels, cols, block_of, scheme, global_composite, part_composite=None, top_pct=50):
    """-> {col: weight_array (len(labels),)}

    labels           specimen labels, in the row order the caller's feature matrix uses.
    cols             measurement column names to produce weights for.
    block_of         column name -> region name (reused from calib_features.py, not reinvented).
    global_composite {label: z}  (from measure.quality_composite)
    part_composite   {region: {label: z}}  (from measure.part_quality_composite), required for
                      scheme='continuous_part'.
    """
    labels = list(labels)
    gvals = np.array([global_composite[lab] for lab in labels])
    if scheme == "equal":
        w = np.ones(len(labels))
        return {c: w.copy() for c in cols}
    if scheme == "hard50":
        w = hard_filter_weight(gvals, top_pct)
        return {c: w.copy() for c in cols}
    if scheme == "continuous_global":
        w = rank_weight(gvals)
        return {c: w.copy() for c in cols}
    if scheme == "continuous_part":
        assert part_composite is not None, "continuous_part needs part_composite"
        global_w = rank_weight(gvals)
        out = {}
        for c in cols:
            reg = block_of(c)
            pcomp = part_composite.get(reg)
            if pcomp is None:
                out[c] = global_w.copy()
                continue
            vals = np.array([pcomp.get(lab, np.nan) for lab in labels])
            ok = np.isfinite(vals)
            w = global_w.copy()  # fallback for specimens/regions with insufficient support
            if ok.sum() >= 3:
                w[ok] = rank_weight(vals[ok])
            out[c] = w
        return out
    raise ValueError(f"unknown scheme {scheme!r}")


def weighted_mean(x, w):
    return float((w * x).sum() / max(w.sum(), 1e-12))


def weighted_pearson(x, y, w):
    """Weighted Pearson R, as defined in the approved plan:
    R_w = sum(w(x-xbar_w)(y-ybar_w)) / sqrt(sum(w(x-xbar_w)^2) * sum(w(y-ybar_w)^2))
    """
    x, y, w = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(w, np.float64)
    if w.sum() <= 0:
        return np.nan
    xm, ym = weighted_mean(x, w), weighted_mean(y, w)
    cov = (w * (x - xm) * (y - ym)).sum()
    vx = (w * (x - xm) ** 2).sum()
    vy = (w * (y - ym) ** 2).sum()
    if vx <= 1e-12 or vy <= 1e-12:
        return np.nan
    return float(cov / np.sqrt(vx * vy))


def weighted_mae(x, y, w):
    x, y, w = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(w, np.float64)
    if w.sum() <= 0:
        return np.nan
    return float((w * np.abs(x - y)).sum() / w.sum())


def weighted_relerr(x, y, w):
    x, y, w = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(w, np.float64)
    if w.sum() <= 0:
        return np.nan
    return float((w * np.abs(x - y) / np.maximum(np.abs(y), 1e-12)).sum() / w.sum())
