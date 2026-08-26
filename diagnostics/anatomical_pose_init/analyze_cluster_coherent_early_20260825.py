#!/usr/bin/env python3
"""Early check (does not need Tipdir_coherent): does Cluster_coherent ALSO lose to zero-init in
aggregate, like PCA_coherent did -- and does cross-estimator spread (pca_vs_cluster only, the one
pair available before tipdir lands) predict per-leg outcome. Run again with the 3rd estimator
once Tipdir_coherent's D1 run finishes (see analyze_multistart_selection_20260825.py)."""
import csv
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CONF_JSON = os.path.join(REPO, "diagnostics", "correspondence_accuracy", "out", "correspondence_confusion.json")
SPREAD_CSV = os.path.join(HERE, "out_ik_20260825", "cross_estimator_spread.csv")

CANDIDATES = {
    "zero": "SYN_clean_zero_wsl",
    "ik_tip_waypoint": "IK_tip_waypoint",
    "pca_coherent": "PCA_coherent",
    "cluster_coherent": "Cluster_coherent",
}


def main():
    with open(CONF_JSON) as f:
        results = json.load(f)
    by_run = {r["run"]: r for r in results}

    leg_acc_scalar = {k: dict(by_run[run]["per_specimen_leg_acc"]) for k, run in CANDIDATES.items()}
    leg_acc_per_leg = {k: dict(by_run[run]["per_specimen_per_leg_acc"]) for k, run in CANDIDATES.items()}
    specimens = sorted(leg_acc_scalar["zero"].keys())

    print("=" * 78)
    print("ITEM 1: does Cluster_coherent also lose to zero-init in aggregate, like PCA did?")
    print("=" * 78)
    means = {k: float(np.mean([leg_acc_scalar[k][s] for s in specimens])) for k in CANDIDATES}
    for k in CANDIDATES:
        print(f"  {k:18s} mean leg_acc = {means[k]:.3f}")
    print(f"\n  PCA_coherent vs zero:     {means['pca_coherent'] - means['zero']:+.3f}")
    print(f"  Cluster_coherent vs zero: {means['cluster_coherent'] - means['zero']:+.3f}")
    print()
    print("  per-specimen (zero / ik / pca / cluster):")
    for s in specimens:
        flag = " <-- Q1-flagged" if s in ("synth_006", "synth_011") else ""
        print(f"    {s:12s} " + "  ".join(f"{k}={leg_acc_scalar[k][s]:.3f}" for k in CANDIDATES) + flag)

    print()
    print("=" * 78)
    print("ITEM 2: does pca_vs_cluster spread predict per-leg outcome (tipdir not available yet)")
    print("=" * 78)
    spread_rows = []
    with open(SPREAD_CSV) as f:
        for row in csv.DictReader(f):
            spread_rows.append((row["specimen"], row["leg"], float(row["pca_vs_cluster"])))

    spreads, pca_acc, cluster_acc, mean_acc = [], [], [], []
    for specimen, leg, spread in spread_rows:
        pa = leg_acc_per_leg["pca_coherent"][specimen].get(leg)
        ca = leg_acc_per_leg["cluster_coherent"][specimen].get(leg)
        if pa is None or ca is None:
            continue
        spreads.append(spread)
        pca_acc.append(pa)
        cluster_acc.append(ca)
        mean_acc.append((pa + ca) / 2.0)

    spreads = np.array(spreads)
    pca_acc = np.array(pca_acc)
    cluster_acc = np.array(cluster_acc)
    mean_acc = np.array(mean_acc)
    print(f"  n legs with usable per-leg data: {len(spreads)}/{len(spread_rows)}")

    for label, y in [("PCA per-leg acc", pca_acc), ("Cluster per-leg acc", cluster_acc), ("mean(PCA,Cluster) per-leg acc", mean_acc)]:
        r, p = stats.pearsonr(spreads, y)
        rs, ps = stats.spearmanr(spreads, y)
        print(f"  spread vs {label:30s}: Pearson r={r:.3f} p={p:.3f}  Spearman rho={rs:.3f} p={ps:.3f}")

    # crude per-specimen proxy fallback, in case per-leg granularity turns out uninformative
    print()
    print("  Fallback per-specimen check (mean spread across 6 legs vs specimen's own leg_acc):")
    per_specimen_mean_spread = {}
    for specimen, leg, spread in spread_rows:
        per_specimen_mean_spread.setdefault(specimen, []).append(spread)
    x = np.array([np.mean(per_specimen_mean_spread[s]) for s in specimens])
    y_pca = np.array([leg_acc_scalar["pca_coherent"][s] for s in specimens])
    y_cluster = np.array([leg_acc_scalar["cluster_coherent"][s] for s in specimens])
    for label, y in [("PCA specimen leg_acc", y_pca), ("Cluster specimen leg_acc", y_cluster)]:
        r, p = stats.pearsonr(x, y)
        print(f"    mean-spread vs {label:28s}: Pearson r={r:.3f} p={p:.3f} (n=12, caveat: small n)")


if __name__ == "__main__":
    main()
