"""Task 4 -- does a better distance metric improve local taxonomic retrieval?

CONTEXT (from `absolute_scale/analyse_source.py`, `FINAL_REPORT.md`): 1-NN genus classification
on log-shape-ratio features shows a real, lot-blind-significant signal; PCA retains it; UMAP/
t-SNE/HDBSCAN find no global cluster structure (ARI <= 0.004, `FINAL_REPORT.md` sec 3.1). The
signal is therefore LOCAL (nearest-neighbour), not global cluster geometry -- so this task holds
the representation fixed (log-shape-ratios -> PCA, `analyse_source.pcs`, npc=10) and varies only
the DISTANCE on top of it.

FIXED BASELINE: log-shape-ratios -> PCA(10) -> squared Euclidean -> lot-blind LOO 1-NN. This is
`analyse_source.py`'s existing protocol, reproduced here (not reimplemented from scratch, but
copied rather than imported -- `analyse_source.py` imports `calib_features`, which imports
`pytorch3d.io` at module level; that is unavailable on this login node/shell, and none of the
functions actually needed here touch meshes, so they are copied verbatim instead of dragging in
a mesh-IO dependency for pure numpy/re logic).

DATA: `absolute_scale/morphometrics_source.csv`, the already-computed 838-specimen deliverable
table (757 worker + 81 ALL_ANTS_CLEAN, both corpora, `source` column distinguishes them) -- built
by `analyse_source.py` from actual fit outputs. No fit outputs need to be re-run for this task;
the log-shape-ratio inputs (raw length columns + already-scale-free shapedev_ columns) are read
straight from that CSV as `analyse_source.features()` would consume them.

WHAT IS AND ISN'T TESTED, and why:

  A. Euclidean baseline    -- current pipeline, reproduced for reference.
  B. Mahalanobis           -- covariance estimated per lot-blind fold from TRAINING specimens
                               only (the held-out lot's specimens never enter the fold's Sigma).
                               Shrunk toward the diagonal (Ledoit-Wolf-style convex blend) because
                               n_train >> npc but Sigma can still be ill-conditioned in the tails
                               of small-lot folds.
  C. Reliability-weighted  -- SKIPPED as a decisive test. Task 2 (`diagnostics/reliability/
                               REPORT.md`) returned a HOLD: no weighting scheme beat `equal` with
                               evidence clearing this project's own pre-registered bar at n=24
                               synthetic specimens, and the one statistically distinguishable
                               cell was a REGRESSION, not an improvement. Per this task's own
                               instruction ("if Task 2 establishes a defensible reliability
                               measure, test..."), that condition is not met, so this is not run
                               as a real arm. One exploratory, clearly-labelled addendum is
                               included instead (`continuous_global`, the frozen no-free-parameter
                               scheme from `reliability/weights.py`, reused rather than refit) so
                               the question isn't left completely unanswered, but it is NOT part
                               of the decision table.
  D. Metric learning       -- one lightweight, defensible version: per-fold diagonal reweighting
                               of the npc PCA axes by their Fisher ratio (between-genus /
                               within-genus variance), computed from TRAINING-fold genus labels
                               only (held-out lot's labels never touch the fold's Fisher scores).
                               This is the simplest thing that is legitimately "metric learning"
                               (a learned diagonal metric, cf. Relevant Component Analysis) without
                               requiring an unavailable package (no scikit-learn/metric-learn in
                               this environment) or overfitting a triplet net on a few hundred
                               points spread over dozens of classes.

EVALUATION: identical lot-blind leave-one-lot-out splits for every method (same `groups=lot`
array, same permutation seeds). Accuracy, permutation null (400 perms), lift, lot-cluster
bootstrap 95% CI (1000 resamples, matches `REPORT_ABSOLUTE_SCALE.md`'s protocol), accuracy at
several min_n thresholds (proxy for "by number of genera" -- more genera as the threshold
relaxes), and cross-corpus retrieval (ALL_ANTS_CLEAN specimen -> nearest WORKER genus centroid,
train covariance/Fisher scores from worker corpus only, clean labels never used to fit anything).
"""

import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ABS_SCALE = os.path.join(REPO, "diagnostics", "absolute_scale")
OUT = os.path.join(HERE, "out")
sys.path.insert(0, ABS_SCALE)

import measure as ms  # noqa: E402  (pure numpy/re, no pytorch3d at import time)

CSV_PATH = os.path.join(ABS_SCALE, "morphometrics_source.csv")
NPC = 10
MIN_N_DEFAULT = 3
N_PERM = 400
N_BOOT = 1000
SEED = 0

CORE_BLOCKS = ("head", "mandible", "antenna", "leg_prox", "mesosoma", "gaster")


# ------------------------------------------------------------------ copied, not reimplemented
# (verbatim logic from absolute_scale/analyse_source.py + calib_features.py; copied because
# analyse_source.py transitively imports pytorch3d.io via calib_features.py, unavailable here,
# for functionality this task never touches)
def block_of(c):
    if c.startswith("shapedev_"):
        c = c[len("shapedev_") :]
    if c.startswith("leaflen_ma") or c.startswith("mandible") or c.startswith("ma") or c == "attach_ma":
        return "mandible"
    if c.startswith("head"):
        return "head"
    if c.startswith("mesosoma"):
        return "mesosoma"
    if c.startswith("gaster") or "b_a" in c:
        return "gaster"
    if "an_" in c:
        return "antenna"
    if any(s in c for s in ("_ti", "_ta", "_pt")):
        return "leg_distal"
    if any(s in c for s in ("_co", "_tr", "_fe")):
        return "leg_prox"
    return "other"


def select_features(cols, which="core"):
    if which == "all":
        return list(cols)
    return [c for c in cols if block_of(c) in CORE_BLOCKS]


def pcs(A, k):
    A = np.nan_to_num(np.asarray(A, dtype=np.float64))
    A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-9)
    U, S, _ = np.linalg.svd(A - A.mean(0), full_matrices=False)
    return (U * S)[:, : min(k, A.shape[1])]


def accession_lot(specimen):
    m = re.match(r"([A-Za-z]+)0*(\d+)", specimen or "")
    return f"{m.group(1)}{int(m.group(2)) // 100}" if m else "?"


def perm_test_generic(acc_fn, y, n=N_PERM, seed=SEED):
    """acc_fn(labels) -> accuracy. Reused for every distance so the null is computed identically."""
    rng = np.random.default_rng(seed)
    a = acc_fn(y)
    null = np.array([acc_fn(rng.permutation(y)) for _ in range(n)])
    return a, float(null.mean()), float(null.std()), float((null >= a).mean())


def lot_bootstrap_ci(acc_fn, y, lots, n=N_BOOT, seed=SEED):
    """Lot-level cluster bootstrap, percentile 95% CI -- matches REPORT_ABSOLUTE_SCALE.md's protocol."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(lots)))
    accs = []
    for _ in range(n):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(lots == g)[0] for g in chosen])
        accs.append(acc_fn_subset(acc_fn, y, idx))
    accs = np.array([a for a in accs if not np.isnan(a)])
    if len(accs) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def acc_fn_subset(acc_fn, y, idx):
    # bootstrap resample changes membership only; acc_fn already handles arbitrary index sets
    return acc_fn(y, idx)


# ------------------------------------------------------------------ data loading
def load_table():
    with open(CSV_PATH) as fh:
        r = csv.DictReader(fh)
        rows = list(r)
    header = list(rows[0].keys())
    meta = ["label", "source", "genus", "species", "subfamily", "ecology", "asym_median", "log_size"]
    rest = [c for c in header if c not in meta]
    sdev = [c for c in rest if c.startswith("shapedev_")]
    sc = [c for c in rest if not c.startswith("shapedev_")]
    sc = select_features(sc, "core")
    sdev = select_features(sdev, "core")

    X = np.array([[float(r[c]) for c in sc] for r in rows], dtype=np.float64)
    D = np.array([[float(r[c]) for c in sdev] for r in rows], dtype=np.float64)
    Z, size = ms.log_shape_ratios(X)
    D = (D - np.nanmean(D, 0)) / np.maximum(np.nanstd(D, 0), 1e-9)
    F = np.hstack([Z, np.nan_to_num(D)])

    src = np.array([r["source"] for r in rows])
    gen = np.array([r["genus"] if r["genus"] else "?" for r in rows])
    label = np.array([r["label"] for r in rows])
    spec = np.array(
        [ms.parse_taxonomy(label[i], src[i])["specimen"] for i in range(len(rows))]
    )
    lot = np.array([accession_lot(s) for s in spec])
    return F, gen, src, lot, sc, sdev


# ------------------------------------------------------------------ distance-metric accuracy functions
def make_euclidean_acc(Fk, gk, lotk):
    D = ((Fk[:, None, :] - Fk[None, :, :]) ** 2).sum(-1)

    def acc(y, idx=None):
        if idx is None:
            idx = np.arange(len(y))
        Dsub = D[np.ix_(idx, idx)].copy()
        lsub = lotk[idx]
        for g in np.unique(lsub):
            k = np.where(lsub == g)[0]
            Dsub[np.ix_(k, k)] = np.inf
        ok = np.isfinite(Dsub).any(1)
        if not ok.any():
            return float("nan")
        ysub = y[idx]
        return float((ysub[Dsub[ok].argmin(1)] == ysub[ok]).mean())

    return acc


def shrunk_inv_cov(X, alpha=0.15, ridge=1e-6):
    """Ledoit-Wolf-style convex shrinkage toward the scaled identity, then invert."""
    Xc = X - X.mean(0)
    n = max(len(Xc), 2)
    S = (Xc.T @ Xc) / (n - 1)
    target = np.eye(S.shape[0]) * (np.trace(S) / S.shape[0])
    S = (1 - alpha) * S + alpha * target
    S += np.eye(S.shape[0]) * ridge
    return np.linalg.inv(S)


def make_mahalanobis_acc(Fk, gk, lotk, alpha=0.15):
    """Sigma re-estimated PER FOLD from training specimens only (held-out lot excluded)."""
    uniq_lots = np.unique(lotk)

    def acc(y, idx=None):
        if idx is None:
            idx = np.arange(len(y))
        idx_set = set(idx.tolist())
        correct, total = 0, 0
        for g in uniq_lots:
            test_i = np.array([i for i in np.where(lotk == g)[0] if i in idx_set])
            if len(test_i) == 0:
                continue
            train_i = np.array([i for i in idx if lotk[i] != g])
            if len(train_i) < Fk.shape[1] + 2:
                continue  # too few training points this fold to estimate a stable Sigma
            Sinv = shrunk_inv_cov(Fk[train_i], alpha=alpha)
            diff = Fk[test_i][:, None, :] - Fk[train_i][None, :, :]
            d2 = np.einsum("tik,kl,til->ti", diff, Sinv, diff)
            pred = y[train_i][d2.argmin(1)]
            correct += int((pred == y[test_i]).sum())
            total += len(test_i)
        return float(correct / total) if total else float("nan")

    return acc


def make_fisher_weighted_acc(Fk, gk, lotk):
    """Diagonal metric learning: per-fold Fisher ratio (between/within genus var) on TRAIN labels only."""
    uniq_lots = np.unique(lotk)

    def fisher_weights(train_i):
        y_tr = gk[train_i]
        Ftr = Fk[train_i]
        classes = [c for c in np.unique(y_tr) if (y_tr == c).sum() >= 2]
        if len(classes) < 2:
            return np.ones(Ftr.shape[1])
        grand = Ftr.mean(0)
        between = np.zeros(Ftr.shape[1])
        within = np.zeros(Ftr.shape[1])
        for c in classes:
            Fc = Ftr[y_tr == c]
            between += len(Fc) * (Fc.mean(0) - grand) ** 2
            within += ((Fc - Fc.mean(0)) ** 2).sum(0)
        within = np.maximum(within, 1e-9)
        ratio = between / within
        return np.sqrt(np.maximum(ratio, 0))

    def acc(y, idx=None):
        if idx is None:
            idx = np.arange(len(y))
        idx_set = set(idx.tolist())
        correct, total = 0, 0
        for g in uniq_lots:
            test_i = np.array([i for i in np.where(lotk == g)[0] if i in idx_set])
            if len(test_i) == 0:
                continue
            train_i = np.array([i for i in idx if lotk[i] != g])
            if len(train_i) < 3:
                continue
            w = fisher_weights(train_i)
            Ftr_w = Fk[train_i] * w
            Fte_w = Fk[test_i] * w
            d2 = ((Fte_w[:, None, :] - Ftr_w[None, :, :]) ** 2).sum(-1)
            pred = y[train_i][d2.argmin(1)]
            correct += int((pred == y[test_i]).sum())
            total += len(test_i)
        return float(correct / total) if total else float("nan")

    return acc


# ------------------------------------------------------------------ per-method evaluation
def evaluate(name, acc_fn, gk, lotk):
    a, m, s, p = perm_test_generic(acc_fn, gk)
    lo, hi = lot_bootstrap_ci(acc_fn, gk, lotk)
    lift = a / max(m, 1e-9)
    print(
        f"  {name:<28}{100*a:>6.1f}%   null {100*m:>5.1f}+-{100*s:.1f}%   p={p:.4f}   "
        f"lift {lift:.2f}x   CI [{100*lo:.1f}, {100*hi:.1f}]%"
    )
    return dict(acc=a, null_mean=m, null_std=s, p=p, lift=lift, ci95=[lo, hi])


def by_min_n(F, gen, src, lot, thresholds=(3, 5, 8, 12)):
    print("\n=== accuracy by min_n threshold (proxy for 'number of genera') ===")
    out = {}
    wmask = src == "worker"
    for min_n in thresholds:
        cnt = {g: int((gen[wmask] == g).sum()) for g in set(gen[wmask])}
        keep = wmask & np.array([cnt.get(g, 0) >= min_n and g != "?" for g in gen])
        Fk, gk, lotk = pcs(F[keep], NPC), gen[keep], lot[keep]
        n_genus = len(set(gk))
        if n_genus < 2 or keep.sum() < 10:
            print(f"  min_n={min_n:<3}  too few genera/specimens, skipped")
            continue
        acc_eu = make_euclidean_acc(Fk, gk, lotk)
        acc_mh = make_mahalanobis_acc(Fk, gk, lotk)
        a_eu, m_eu, _, _ = perm_test_generic(acc_eu, gk)
        a_mh, m_mh, _, _ = perm_test_generic(acc_mh, gk)
        print(
            f"  min_n={min_n:<3} n={int(keep.sum()):<4} genera={n_genus:<3}  "
            f"euclid {100*a_eu:5.1f}% (lift {a_eu/max(m_eu,1e-9):.2f}x)   "
            f"mahal {100*a_mh:5.1f}% (lift {a_mh/max(m_mh,1e-9):.2f}x)"
        )
        out[min_n] = dict(n=int(keep.sum()), n_genus=n_genus, euclid=a_eu, mahal=a_mh)
    return out


def cross_corpus(F, gen, src, lot):
    """ALL_ANTS_CLEAN specimen -> nearest WORKER genus centroid, three metrics.

    Corpus-safe by construction: Sigma / Fisher weights are estimated from the worker corpus
    alone, never touching a clean-corpus row; clean labels are used only to SCORE predictions.
    """
    print("\n=== 4. CROSS-CORPUS RETRIEVAL (worker -> centroid, clean specimens scored against it) ===")
    Fs = pcs(F, NPC)
    wmask = src == "worker"
    cm = (src == "clean") & (gen != "?")
    shared = sorted(set(gen[cm]) & set(gen[wmask]))
    if not shared:
        print("  no shared genera; skipped")
        return {}
    tm = cm & np.array([g in shared for g in gen])
    Xw, yw = Fs[wmask], gen[wmask]
    Xc, yc = Fs[tm], gen[tm]

    results = {}

    # A. Euclidean centroid
    C = np.array([Xw[yw == g].mean(0) for g in shared])
    d = ((Xc[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    pred = np.array(shared)[d.argmin(1)]
    hit = pred == yc
    results["euclidean"] = dict(top1=float(hit.mean()), n=int(tm.sum()), n_genera=len(shared))

    # B. Mahalanobis to centroid, Sigma from full worker corpus (never touches clean)
    Sinv = shrunk_inv_cov(Xw)
    diff = Xc[:, None, :] - C[None, :, :]
    d2 = np.einsum("tik,kl,til->ti", diff, Sinv, diff)
    pred_m = np.array(shared)[d2.argmin(1)]
    hit_m = pred_m == yc
    results["mahalanobis"] = dict(top1=float(hit_m.mean()), n=int(tm.sum()), n_genera=len(shared))

    # D. Fisher-weighted centroid, weights from worker genus labels only
    grand = Xw.mean(0)
    between = np.zeros(Xw.shape[1])
    within = np.zeros(Xw.shape[1])
    for g in shared:
        Fc = Xw[yw == g]
        if len(Fc) < 2:
            continue
        between += len(Fc) * (Fc.mean(0) - grand) ** 2
        within += ((Fc - Fc.mean(0)) ** 2).sum(0)
    w = np.sqrt(np.maximum(between / np.maximum(within, 1e-9), 0))
    Cw, Xcw = C * w, Xc * w
    dfw = ((Xcw[:, None, :] - Cw[None, :, :]) ** 2).sum(-1)
    pred_f = np.array(shared)[dfw.argmin(1)]
    hit_f = pred_f == yc
    results["fisher_weighted"] = dict(top1=float(hit_f.mean()), n=int(tm.sum()), n_genera=len(shared))

    chance = 1.0 / len(shared)
    hits = dict(euclidean=hit, mahalanobis=hit_m, fisher_weighted=hit_f)

    # per-method permutation null (shuffle the true labels among the tested specimens, keep
    # each method's own fixed predictions) and a specimen-level bootstrap 95% CI on top-1.
    rng = np.random.default_rng(SEED)
    n_test = len(yc)
    for name, pred_arr in dict(euclidean=pred, mahalanobis=pred_m, fisher_weighted=pred_f).items():
        null = np.array([(pred_arr == rng.permutation(yc)).mean() for _ in range(N_PERM)])
        p = float((null >= hits[name].mean()).mean())
        boot = np.array(
            [
                hits[name][rng.integers(0, n_test, size=n_test)].mean()
                for _ in range(N_BOOT)
            ]
        )
        results[name]["null_mean"] = float(null.mean())
        results[name]["p"] = p
        results[name]["ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    # paired bootstrap on the DIFFERENCE vs euclidean (same test specimens every draw) -- this is
    # the decisive comparison, not the marginal top-1 numbers, since a few-point top-1 gap on
    # n=46 specimens is otherwise indistinguishable from resampling noise.
    for name in ("mahalanobis", "fisher_weighted"):
        idx_boot = rng.integers(0, n_test, size=(N_BOOT, n_test))
        diff = hits[name][idx_boot].mean(1) - hits["euclidean"][idx_boot].mean(1)
        results[name]["paired_delta_vs_euclidean"] = float(hits[name].mean() - hits["euclidean"].mean())
        results[name]["paired_ci95_vs_euclidean"] = [
            float(np.percentile(diff, 2.5)),
            float(np.percentile(diff, 97.5)),
        ]

    for name, r in results.items():
        extra = ""
        if "paired_delta_vs_euclidean" in r:
            lo, hi = r["paired_ci95_vs_euclidean"]
            extra = f"   delta-vs-euclid {100*r['paired_delta_vs_euclidean']:+.1f}pp CI [{100*lo:+.1f}, {100*hi:+.1f}]pp"
        print(
            f"  {name:<18} top1 {100*r['top1']:.1f}%   chance {100*chance:.1f}%   "
            f"null {100*r.get('null_mean', float('nan')):.1f}%   p={r.get('p', float('nan')):.4f}   "
            f"n={r['n']} genera={r['n_genera']}{extra}"
        )
    results["chance"] = chance
    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    F, gen, src, lot, sc, sdev = load_table()
    print(f"loaded {len(gen)} specimens ({(src=='worker').sum()} worker, {(src=='clean').sum()} clean), "
          f"{len(sc)} length + {len(sdev)} shape-dev core features")

    wmask = src == "worker"
    cnt = {g: int((gen[wmask] == g).sum()) for g in set(gen[wmask])}
    keep = wmask & np.array([cnt.get(g, 0) >= MIN_N_DEFAULT and g != "?" for g in gen])
    Fk, gk, lotk = pcs(F[keep], NPC), gen[keep], lot[keep]
    n_genus, n_lot = len(set(gk)), len(set(lotk))
    print(f"\n=== FIXED BASELINE COMPARISON  (n={int(keep.sum())}, {n_genus} genera, {n_lot} lots, "
          f"min_n>={MIN_N_DEFAULT}, npc={NPC}) ===")

    res = {}
    res["A_euclidean"] = evaluate("A. Euclidean (baseline)", make_euclidean_acc(Fk, gk, lotk), gk, lotk)
    res["B_mahalanobis"] = evaluate("B. Mahalanobis (fold-safe Sigma)", make_mahalanobis_acc(Fk, gk, lotk), gk, lotk)
    res["D_fisher_weighted"] = evaluate("D. Fisher-weighted (fold-safe)", make_fisher_weighted_acc(Fk, gk, lotk), gk, lotk)

    res["by_min_n"] = by_min_n(F, gen, src, lot)
    res["cross_corpus"] = cross_corpus(F, gen, src, lot)

    res["n_specimens"] = int(keep.sum())
    res["n_genus"] = n_genus
    res["n_lot"] = n_lot
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    print(f"\nwrote {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    main()
