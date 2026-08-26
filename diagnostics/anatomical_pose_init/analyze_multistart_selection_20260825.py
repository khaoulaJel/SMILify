#!/usr/bin/env python3
"""Multi-start coherent candidate pool analysis (2026-08-25), answering the user's three
pre-registered checks BEFORE treating any single leg_acc number as an answer:

  1. Does cross-estimator direction spread (cross_estimator_spread.csv) predict per-LEG outcome?
  2. Does the 5-candidate oracle ceiling (zero, IK_tip_waypoint, PCA/Cluster/Tipdir-coherent)
     meaningfully beat the previous 3-candidate ceiling?
  3. Does the GT-free (fscore@0.02) selector's per-specimen correctness rate hold up, or degrade,
     when given 5 candidates instead of 3?

Requires: run_multistart_coherent_20260825.sh finished (Cluster_coherent/Tipdir_coherent D1 runs
exist), and diagnostics/correspondence_accuracy/run_audit.py re-run so
correspondence_confusion.json includes them.
"""
import csv
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS_DIR = os.path.join(REPO, "diagnostics", "moonshot", "runs")
CONF_JSON = os.path.join(REPO, "diagnostics", "correspondence_accuracy", "out", "correspondence_confusion.json")
SPREAD_CSV = os.path.join(HERE, "out_ik_20260825", "cross_estimator_spread.csv")

CANDIDATES = {
    "zero": "SYN_clean_zero_wsl",
    "ik": "IK_tip_waypoint",
    "pca": "PCA_coherent",
    "cluster": "Cluster_coherent",
    "tipdir": "Tipdir_coherent",
}
COHERENT_FAMILY = ["pca", "cluster", "tipdir"]


def load_fscore(run):
    path = os.path.join(RUNS_DIR, run, "metrics.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            stem = row["mesh"].removesuffix(".obj")
            out[stem] = float(row["fscore@0.02"])
    return out


def load_confusion():
    with open(CONF_JSON) as f:
        results = json.load(f)
    by_run = {r["run"]: r for r in results}
    return by_run


def main():
    by_run = load_confusion()
    missing = [r for r in CANDIDATES.values() if r not in by_run]
    if missing:
        raise SystemExit(f"correspondence_confusion.json missing runs: {missing} -- re-run run_audit.py first")

    fscore = {k: load_fscore(run) for k, run in CANDIDATES.items()}
    leg_acc_scalar = {}  # cand -> {specimen: leg_acc}
    leg_acc_per_leg = {}  # cand -> {specimen: {leg: acc}}
    for k, run in CANDIDATES.items():
        r = by_run[run]
        leg_acc_scalar[k] = dict(r["per_specimen_leg_acc"])
        leg_acc_per_leg[k] = dict(r["per_specimen_per_leg_acc"])

    specimens = sorted(leg_acc_scalar["zero"].keys())
    assert len(specimens) == 12, f"expected 12 synth_clean specimens, got {len(specimens)}"

    # ---------------------------------------------------------------
    # Check 2: 5-candidate oracle ceiling vs zero-init baseline
    # ---------------------------------------------------------------
    print("=" * 78)
    print("CHECK 2: 5-candidate oracle ceiling")
    print("=" * 78)
    zero_mean = np.mean([leg_acc_scalar["zero"][s] for s in specimens])
    per_cand_mean = {k: np.mean([leg_acc_scalar[k][s] for s in specimens]) for k in CANDIDATES}
    oracle_best = {s: max(leg_acc_scalar[k][s] for k in CANDIDATES) for s in specimens}
    oracle_which = {s: max(CANDIDATES, key=lambda k: leg_acc_scalar[k][s]) for s in specimens}
    oracle_mean_5 = np.mean(list(oracle_best.values()))

    # 3-candidate ceiling for direct comparison (zero, ik, pca -- the pool before this session)
    oracle_best_3 = {s: max(leg_acc_scalar[k][s] for k in ("zero", "ik", "pca")) for s in specimens}
    oracle_mean_3 = np.mean(list(oracle_best_3.values()))

    for k in CANDIDATES:
        print(f"  {k:8s} ({CANDIDATES[k]:20s}) mean leg_acc = {per_cand_mean[k]:.3f}")
    print()
    print(f"  3-candidate (zero/ik/pca) oracle ceiling: {oracle_mean_3:.3f}")
    print(f"  5-candidate (+cluster/tipdir) oracle ceiling: {oracle_mean_5:.3f}")
    print(f"  delta: {oracle_mean_5 - oracle_mean_3:+.3f}")
    print()
    print("  per-specimen oracle winner (5-candidate pool):")
    for s in specimens:
        flag = " <-- Q1-flagged" if s in ("synth_006", "synth_011") else ""
        print(f"    {s:12s} best={oracle_which[s]:8s} leg_acc={oracle_best[s]:.3f}{flag}")

    # ---------------------------------------------------------------
    # Check 3: GT-free (fscore@0.02) selector, 3-candidate vs 5-candidate
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("CHECK 3: GT-free selector correctness, 3-candidate vs 5-candidate pool")
    print("=" * 78)

    def run_selector(pool):
        picks, margins, correct_vs_oracle, beats_zero = {}, {}, {}, {}
        for s in specimens:
            scores = sorted(((fscore[k][s], k) for k in pool), reverse=True)
            best_score, best_k = scores[0]
            second_score = scores[1][0]
            picks[s] = best_k
            margins[s] = best_score - second_score
            oracle_best_pool = max(pool, key=lambda k: leg_acc_scalar[k][s])
            correct_vs_oracle[s] = (best_k == oracle_best_pool)
            beats_zero[s] = leg_acc_scalar[best_k][s] >= leg_acc_scalar["zero"][s]
        return picks, margins, correct_vs_oracle, beats_zero

    for label, pool in [("3-candidate", ("zero", "ik", "pca")), ("5-candidate", tuple(CANDIDATES))]:
        picks, margins, correct, beats_zero = run_selector(pool)
        sel_leg_acc = np.mean([leg_acc_scalar[picks[s]][s] for s in specimens])
        n_correct = sum(correct.values())
        n_beats_zero = sum(beats_zero.values())
        print(f"\n  [{label}] pool={pool}")
        print(f"    mean SELECTED leg_acc: {sel_leg_acc:.3f}  (zero baseline: {zero_mean:.3f})")
        print(f"    selector picks the TRUE oracle-best candidate: {n_correct}/12")
        print(f"    selected candidate's leg_acc >= zero-init: {n_beats_zero}/12")
        for s in specimens:
            flag = " <-- Q1-flagged" if s in ("synth_006", "synth_011") else ""
            ok = "OK " if correct[s] else "MISS"
            print(f"      {s:12s} picked={picks[s]:8s} margin={margins[s]:+.4f} oracle_match={ok}{flag}")

    # margin-vs-correctness correlation, 5-candidate pool
    picks5, margins5, correct5, _ = run_selector(tuple(CANDIDATES))
    m = np.array([margins5[s] for s in specimens])
    c = np.array([1.0 if correct5[s] else 0.0 for s in specimens])
    if len(set(c.tolist())) > 1:
        r, p = stats.pointbiserialr(c, m)
        print(f"\n  margin vs correctness (5-candidate), point-biserial r={r:.3f} p={p:.3f} (n=12, caveat: small n)")
    else:
        print(f"\n  margin vs correctness (5-candidate): all correct={bool(c[0])}, correlation undefined (no variance)")

    # ---------------------------------------------------------------
    # Check 1: cross-estimator spread vs per-LEG outcome
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("CHECK 1: cross-estimator spread vs per-leg outcome (coherent family: pca/cluster/tipdir)")
    print("=" * 78)

    spread_rows = []
    with open(SPREAD_CSV) as f:
        for row in csv.DictReader(f):
            if row["n_valid_estimators"] != "3":
                continue
            vals = [float(row["pca_vs_cluster"]), float(row["pca_vs_tipdir"]), float(row["cluster_vs_tipdir"])]
            spread_rows.append((row["specimen"], row["leg"], float(np.mean(vals)), float(np.max(vals))))

    mean_spreads, max_spreads, coherent_acc_at_leg = [], [], []
    for specimen, leg, mean_spread, max_spread in spread_rows:
        accs = [leg_acc_per_leg[k][specimen][leg] for k in COHERENT_FAMILY if leg in leg_acc_per_leg[k][specimen]]
        if not accs:
            continue
        mean_spreads.append(mean_spread)
        max_spreads.append(max_spread)
        coherent_acc_at_leg.append(float(np.mean(accs)))  # mean per-leg acc ACROSS the 3 coherent candidates

    mean_spreads = np.array(mean_spreads)
    max_spreads = np.array(max_spreads)
    coherent_acc_at_leg = np.array(coherent_acc_at_leg)
    print(f"  n legs with usable data: {len(mean_spreads)}/{len(spread_rows)}")

    for label, x in [("mean pairwise spread", mean_spreads), ("max pairwise spread", max_spreads)]:
        r, p = stats.pearsonr(x, coherent_acc_at_leg)
        rs, ps = stats.spearmanr(x, coherent_acc_at_leg)
        print(f"  {label:22s} vs mean-of-3-coherent-candidates per-leg accuracy: "
              f"Pearson r={r:.3f} p={p:.3f}  Spearman rho={rs:.3f} p={ps:.3f}")

    # high-spread (p95-range, >=47deg by the earlier real-scan measurement) vs low-spread split
    threshold = 47.0
    hi = coherent_acc_at_leg[max_spreads >= threshold]
    lo = coherent_acc_at_leg[max_spreads < threshold]
    print(f"\n  legs with max-pairwise-spread >= {threshold} deg (n={len(hi)}): "
          f"mean coherent-family per-leg acc = {hi.mean() if len(hi) else float('nan'):.3f}")
    print(f"  legs with max-pairwise-spread <  {threshold} deg (n={len(lo)}): "
          f"mean coherent-family per-leg acc = {lo.mean() if len(lo) else float('nan'):.3f}")


if __name__ == "__main__":
    main()
