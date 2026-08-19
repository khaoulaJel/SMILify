"""Task 5 -- does the subfamily -> genus taxonomy hierarchy improve genus retrieval, or does it
only look like it does because restricting candidates with the TRUE subfamily makes the problem
easier?

Reuses `analyse_source.py`'s exact machinery (`pcs`, `loo_1nn`, `perm_test`, `accession_lot`,
`features`, `select_features`) and its cached deliverable table (`out/morphometrics.csv` --
`morphometrics_source.csv` at repo root of this directory is that same table, already on disk
from a prior full-corpus run; no fit outputs are required to reproduce this).

FOUR THINGS ARE MEASURED, in increasing order of what they're allowed to claim:

1. FLAT BASELINE.  features -> genus, plain leave-one-lot-out 1-NN. Exactly
   `analyse_source.py`'s `signal_genus` protocol, recomputed here on the exact fold set used
   below (worker corpus, genus n>=min_n, subfamily known) so the comparison in (4) is apples to
   apples -- NOT the number in `out/analysis.json`, which is computed on a superset that includes
   specimens with no subfamily label.

2. SUBFAMILY PREDICTION.  features -> subfamily, plain leave-one-lot-out 1-NN. Same folds.

3. GENUS | TRUE SUBFAMILY (diagnostic, not a system).  1-NN restricted to training specimens that
   share the QUERY'S TRUE subfamily. This is expected to beat (1) almost by construction --
   shrinking the candidate pool with ground truth you would not have at inference time is not a
   result, it's a sanity check that the restriction logic works. Reported, never sold as "the
   hierarchy helps".

4. END-TO-END (the actual system).  1-NN restricted to training specimens that share the QUERY'S
   PREDICTED subfamily from (2) -- errors in (2) propagate into (4) exactly as they would at
   inference time. THIS is the only number allowed to be compared against (1).

DECISION RULE. Hierarchical only counts as an improvement if (4) beats (1) on the same folds, with
a paired-bootstrap-over-lots 95% CI on the difference that excludes zero -- the same statistical
bar this project's other diagnostics evidence uses (`reliability/REPORT.md`).

The null for (3) and (4) permutes genus labels WITHIN each specimen's true subfamily stratum, not
globally: a global permutation would let the null pipeline "predict" genus from subfamily
membership alone even when genus carries zero signal beyond subfamily, silently inflating the
lift of every hierarchical arm.  The null for (1) and (2) is the project's standard unconstrained
label permutation (`analyse_source.perm_test`).
"""

import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import analyse_source as asrc  # noqa: E402

CSV_PATH = os.path.join(HERE, "morphometrics_source.csv")
OUT = os.path.join(HERE, "out")

# same worker-naming convention as measure.parse_taxonomy, reimplemented here because the CSV
# stores only the final `label`, not the `specimen` code parse_taxonomy already extracted.
_CASENT = re.compile(r"(CASENT|OKENT|ANTWEB|UCDC|MCZ)[0-9A-Za-z]*", re.I)


def specimen_of(label):
    stem = label[:-4] if label.endswith(".obj") else label
    stem = stem.replace("_processed", "")
    m = _CASENT.search(stem)
    return m.group(0) if m else stem


# ------------------------------------------------------------------ restricted 1-NN
def loo_pred_restricted(F, y, groups, eligible=None):
    """Leave-one-GROUP-out 1-NN predictions (not just accuracy -- (4) needs (2)'s per-row output).

    `eligible`: optional (N, N) boolean matrix, eligible[i, j] = True if training point j may be
    used as a candidate for query i (on top of the group exclusion, which always applies).
    Returns (pred, ok): ok[i] is False iff query i had no eligible candidate left anywhere.
    """
    D = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        k = np.where(groups == g)[0]
        D[np.ix_(k, k)] = np.inf
    if eligible is not None:
        D = np.where(eligible, D, np.inf)
    ok = np.isfinite(D).any(1)
    pred = np.full(len(y), None, dtype=object)
    idx = np.where(ok)[0]
    nn = D[idx].argmin(1)
    pred[idx] = y[nn]
    return pred, ok


def loo_centroid_pred(F, y, groups):
    """Leave-one-LOT-out NEAREST-CENTROID (not nearest-neighbour) classifier.

    Exists to break a forced identity: if subfamily is routed by "the label of the nearest
    neighbour" using the SAME feature space and metric as the genus 1-NN stage, then the genus
    search restricted to that predicted subfamily can never find anything closer than the
    unrestricted flat 1-NN's own answer (the flat nearest neighbour is always a member of its own
    predicted subfamily, so it is always still eligible) -- the two-stage system is then
    PROVABLY identical to the flat baseline, not just empirically tied. A centroid-routed top
    stage is a genuinely different classifier and is not subject to that identity, so it is the
    only construction here that could actually show a hierarchical benefit.
    """
    groups = np.asarray(groups)
    uniq_y = np.unique(y)
    pred = np.full(len(y), None, dtype=object)
    ok = np.zeros(len(y), dtype=bool)
    for i in range(len(y)):
        train = groups != groups[i]
        cents, labels = [], []
        for c in uniq_y:
            k = train & (y == c)
            if k.sum() > 0:
                labels.append(c)
                cents.append(F[k].mean(0))
        if not cents:
            continue
        C = np.array(cents)
        d = ((F[i][None, :] - C) ** 2).sum(1)
        pred[i] = labels[int(d.argmin())]
        ok[i] = True
    return pred, ok


def stratified_permute(y, strata, rng):
    """Permute `y` within each `strata` group independently (preserves the strata->y coupling
    a hierarchical system is allowed to exploit, unlike a fully global permutation)."""
    y = np.asarray(y).copy()
    strata = np.asarray(strata)
    for s in np.unique(strata):
        k = np.where(strata == s)[0]
        y[k] = rng.permutation(y[k])
    return y


def paired_bootstrap_ci(diffs_fn, lots, n=2000, seed=0):
    """95% CI on a paired difference, resampling whole LOTS with replacement (the cluster unit
    this project's fold protocol is built around -- see accession_lot's docstring)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(lots)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(lots == p)[0] for p in pick])
        vals.append(diffs_fn(idx))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--min_n", type=int, default=3)
    ap.add_argument("--npc", type=int, default=10)
    ap.add_argument("--nperm", type=int, default=400)
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = list(csv.DictReader(open(CSV_PATH)))
    meta = ["label", "source", "genus", "species", "subfamily", "ecology", "asym_median", "log_size"]
    sc = [c for c in rows[0].keys() if c not in meta and not c.startswith("shapedev_")]
    sdev = [c for c in rows[0].keys() if c.startswith("shapedev_")]
    for r in rows:
        for c in sc + sdev:
            r[c] = float(r[c])
    F, Z, size, X = asrc.features(rows, sc, sdev)

    src = np.array([r["source"] for r in rows])
    gen = np.array([r["genus"] if r["genus"] else "?" for r in rows])
    sub = np.array([r["subfamily"] if r["subfamily"] else "?" for r in rows])
    lot = np.array([asrc.accession_lot(specimen_of(r["label"])) for r in rows])

    wmask = src == "worker"
    cnt = {g: int((gen[wmask] == g).sum()) for g in set(gen[wmask])}
    mask = wmask & np.array([cnt.get(g, 0) >= args.min_n and g != "?" for g in gen]) & (sub != "?")
    print(
        f"folds: {mask.sum()} worker specimens, genus n>={args.min_n}, subfamily known "
        f"-- {len(set(gen[mask]))} genera, {len(set(sub[mask]))} subfamilies, "
        f"{len(set(lot[mask]))} accession lots"
    )
    n_sf_singleton_genus = sum(
        1 for s in set(sub[mask]) if len(set(gen[mask][sub[mask] == s])) == 1
    )
    if n_sf_singleton_genus:
        print(
            f"  NOTE: {n_sf_singleton_genus} subfamilies in this fold set have only ONE genus "
            f"present -- true-subfamily-conditioned genus accuracy is 100% there by construction, "
            f"not signal. Watch for this inflating arm (3)."
        )

    Fk = asrc.pcs(F[mask], args.npc)
    gk, sk, lk = gen[mask], sub[mask], lot[mask]
    Fk = np.asarray(Fk)

    # ---------------- (1) flat baseline: features -> genus, plain lot-blind 1-NN
    pred1, ok1 = loo_pred_restricted(Fk, gk, lk)
    acc1 = float((pred1[ok1] == gk[ok1]).mean())
    a1, m1, s1, p1 = asrc.perm_test(Fk, gk, groups=lk, n=args.nperm)
    print(f"\n1. FLAT genus baseline           {100 * a1:6.1f}%   null {100 * m1:5.1f}+-{100 * s1:.1f}%  p={p1:.4f}  LIFT {a1 / max(m1, 1e-9):.2f}x")

    # ---------------- (2) subfamily prediction
    pred2, ok2 = loo_pred_restricted(Fk, sk, lk)
    acc2 = float((pred2[ok2] == sk[ok2]).mean())
    a2, m2, s2, p2 = asrc.perm_test(Fk, sk, groups=lk, n=args.nperm)
    print(f"2. SUBFAMILY prediction          {100 * a2:6.1f}%   null {100 * m2:5.1f}+-{100 * s2:.1f}%  p={p2:.4f}  LIFT {a2 / max(m2, 1e-9):.2f}x")

    # ---------------- (3) genus | TRUE subfamily -- diagnostic only
    elig_true = sk[:, None] == sk[None, :]
    pred3, ok3 = loo_pred_restricted(Fk, gk, lk, eligible=elig_true)
    acc3 = float((pred3[ok3] == gk[ok3]).mean())
    rng = np.random.default_rng(0)
    null3 = np.array(
        [
            (
                loo_pred_restricted(Fk, stratified_permute(gk, sk, rng), lk, eligible=elig_true)[0][ok3]
                == gk[ok3]
            ).mean()
            for _ in range(args.nperm)
        ]
    )
    p3 = float((null3 >= acc3).mean())
    m3, s3 = float(null3.mean()), float(null3.std())
    print(
        f"3. genus | TRUE subfamily (diag) {100 * acc3:6.1f}%   null {100 * m3:5.1f}+-{100 * s3:.1f}%  "
        f"p={p3:.4f}  LIFT {acc3 / max(m3, 1e-9):.2f}x   <- restricted with ground truth, expect inflated"
    )

    # ---------------- (4) end-to-end: genus | PREDICTED subfamily -- the actual system
    elig_pred = np.zeros_like(elig_true)
    for i in range(len(sk)):
        if ok2[i] and pred2[i] is not None:
            elig_pred[i] = sk == pred2[i]
        # if subfamily routing failed for this query, it has no candidates -> excluded (ok4=False)
    pred4, ok4 = loo_pred_restricted(Fk, gk, lk, eligible=elig_pred)
    acc4 = float((pred4[ok4] == gk[ok4]).mean())
    null4 = np.array(
        [
            (
                loo_pred_restricted(Fk, stratified_permute(gk, sk, rng), lk, eligible=elig_pred)[0][ok4]
                == gk[ok4]
            ).mean()
            for _ in range(args.nperm)
        ]
    )
    p4 = float((null4 >= acc4).mean())
    m4, s4 = float(null4.mean()), float(null4.std())
    print(
        f"4. END-TO-END genus (system)     {100 * acc4:6.1f}%   null {100 * m4:5.1f}+-{100 * s4:.1f}%  "
        f"p={p4:.4f}  LIFT {acc4 / max(m4, 1e-9):.2f}x   n_routed={int(ok4.sum())}/{len(sk)} "
        f"({int((~ok4).sum())} lost to failed subfamily routing)"
    )

    # ---------------- (2b)/(4b) CENTROID-routed variant -- (2)/(4) used NN-routing, which makes
    # (4) provably identical to (1) (see loo_centroid_pred's docstring). This is the one
    # construction that can actually test whether the hierarchy helps.
    pred2b, ok2b = loo_centroid_pred(Fk, sk, lk)
    acc2b = float((pred2b[ok2b] == sk[ok2b]).mean())
    rng_b = np.random.default_rng(1)
    null2b = np.array(
        [
            (loo_centroid_pred(Fk, rng_b.permutation(sk), lk)[0][ok2b] == sk[ok2b]).mean()
            for _ in range(args.nperm)
        ]
    )
    p2b = float((null2b >= acc2b).mean())
    m2b = float(null2b.mean())
    print(
        f"\n2b. SUBFAMILY via nearest CENTROID {100 * acc2b:5.1f}%   null {100 * m2b:5.1f}%          "
        f"p={p2b:.4f}  LIFT {acc2b / max(m2b, 1e-9):.2f}x"
    )

    elig_pred_b = np.zeros_like(elig_true)
    for i in range(len(sk)):
        if ok2b[i]:
            elig_pred_b[i] = sk == pred2b[i]
    pred4b, ok4b = loo_pred_restricted(Fk, gk, lk, eligible=elig_pred_b)
    acc4b = float((pred4b[ok4b] == gk[ok4b]).mean())
    null4b = np.array(
        [
            (
                loo_pred_restricted(Fk, stratified_permute(gk, sk, rng), lk, eligible=elig_pred_b)[0][ok4b]
                == gk[ok4b]
            ).mean()
            for _ in range(args.nperm)
        ]
    )
    p4b = float((null4b >= acc4b).mean())
    m4b = float(null4b.mean())
    print(
        f"4b. END-TO-END genus (centroid-routed) {100 * acc4b:5.1f}%   null {100 * m4b:5.1f}%          "
        f"p={p4b:.4f}  LIFT {acc4b / max(m4b, 1e-9):.2f}x   n_routed={int(ok4b.sum())}/{len(sk)}"
    )

    # ---------------- required comparison: (4) and (4b) vs (1), same folds, paired bootstrap over lots
    def make_diff_fn(predA, okA):
        both_ok = ok1 & okA

        def diff_fn(idx):
            i = idx[np.isin(idx, np.where(both_ok)[0])]
            if len(i) == 0:
                return 0.0
            return float((predA[i] == gk[i]).mean() - (pred1[i] == gk[i]).mean())

        return diff_fn, both_ok

    diff_fn4, both_ok4 = make_diff_fn(pred4, ok4)
    mean_diff, lo, hi = paired_bootstrap_ci(diff_fn4, lk, n=args.nboot)
    verdict = "IMPROVEMENT" if lo > 0 else ("REGRESSION" if hi < 0 else "NO DISTINGUISHABLE DIFFERENCE")
    print(
        f"\n=== (4) NN-routed END-TO-END vs (1) FLAT, same {both_ok4.sum()}-specimen folds:\n"
        f"    Δacc = {100 * mean_diff:+.1f} pp   95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}] pp   "
        f"-> {verdict} (expected: exactly 0, this arm is provably == flat, see docstring)"
    )

    diff_fn4b, both_ok4b = make_diff_fn(pred4b, ok4b)
    mean_diff_b, lo_b, hi_b = paired_bootstrap_ci(diff_fn4b, lk, n=args.nboot, seed=1)
    verdict_b = "IMPROVEMENT" if lo_b > 0 else ("REGRESSION" if hi_b < 0 else "NO DISTINGUISHABLE DIFFERENCE")
    print(
        f"\n=== (4b) CENTROID-routed END-TO-END vs (1) FLAT, same {both_ok4b.sum()}-specimen folds:\n"
        f"    Δacc = {100 * mean_diff_b:+.1f} pp   95% CI [{100 * lo_b:+.1f}, {100 * hi_b:+.1f}] pp   "
        f"-> {verdict_b}"
    )

    import json

    res = dict(
        n=int(mask.sum()),
        n_genus=len(set(gk)),
        n_subfamily=len(set(sk)),
        n_singleton_genus_subfamilies=n_sf_singleton_genus,
        flat_genus=dict(acc=a1, null=m1, p=p1, lift=a1 / max(m1, 1e-9)),
        subfamily=dict(acc=a2, null=m2, p=p2, lift=a2 / max(m2, 1e-9)),
        genus_given_true_subfamily=dict(acc=acc3, null=m3, p=p3, lift=acc3 / max(m3, 1e-9)),
        genus_end_to_end_nn_routed=dict(
            acc=acc4, null=m4, p=p4, lift=acc4 / max(m4, 1e-9), n_routed=int(ok4.sum())
        ),
        subfamily_centroid=dict(acc=acc2b, null=m2b, p=p2b, lift=acc2b / max(m2b, 1e-9)),
        genus_end_to_end_centroid_routed=dict(
            acc=acc4b, null=m4b, p=p4b, lift=acc4b / max(m4b, 1e-9), n_routed=int(ok4b.sum())
        ),
        end_to_end_nn_routed_vs_flat=dict(mean_diff=mean_diff, ci_lo=lo, ci_hi=hi, verdict=verdict),
        end_to_end_centroid_routed_vs_flat=dict(
            mean_diff=mean_diff_b, ci_lo=lo_b, ci_hi=hi_b, verdict=verdict_b
        ),
    )
    json.dump(res, open(os.path.join(OUT, "hier_taxonomy.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(OUT, 'hier_taxonomy.json')}")


if __name__ == "__main__":
    main()
