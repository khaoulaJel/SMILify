"""TASK 3 step 1: independently recompute the baseline-vs-gentle GWN
disagreement-rate summary directly from the raw per-specimen CSVs
(fit3d_results_all_{baseline,gentle}/Stage_3_deform_fine_gwn_disagreement.csv),
not from the two SLURM logs' printed summaries.

Cross-checks against the table already in TASK3_HANDOFF.md (which was read off
the logs) and flags any mismatch.
"""
import csv
import os
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CSVS = {
    "baseline": os.path.join(REPO, "fit3d_results_all_baseline", "Stage_3_deform_fine_gwn_disagreement.csv"),
    "gentle": os.path.join(REPO, "fit3d_results_all_gentle", "Stage_3_deform_fine_gwn_disagreement.csv"),
}

# handoff table (from the log's own printed summary), for cross-check only
HANDOFF = {
    ("mandible", "antenna", "A_into_B"): (0.0479, 0.1221),
    ("mandible", "antenna", "B_into_A"): (0.0925, 0.0492),
    ("gaster", "legs", "A_into_B"): (0.0195, 0.0844),
    ("head", "legs", "A_into_B"): (0.0036, 0.0377),
    ("antenna", "legs", "A_into_B"): (0.0097, 0.0250),
    ("mandible", "legs", "A_into_B"): (0.0020, 0.0220),
    ("antenna", "thorax", "B_into_A"): (0.0017, 0.0148),
    ("gaster", "legs", "B_into_A"): (0.0131, 0.0127),
    ("mandible", "thorax", "B_into_A"): (0.0010, 0.0100),
    ("antenna", "legs", "B_into_A"): (0.0019, 0.0091),
    ("antenna", "thorax", "A_into_B"): (0.0031, 0.0056),
    ("head", "legs", "B_into_A"): (0.0016, 0.0036),
    ("mandible", "thorax", "A_into_B"): (0.0033, 0.0034),
    ("mandible", "legs", "B_into_A"): (0.0014, 0.0031),
    ("antenna", "gaster", "B_into_A"): (0.0000, 0.0007),
}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["disagreement_rate"] = float(r["disagreement_rate"]) if r["disagreement_rate"] != "" else None
            r["trustworthy_frac"] = float(r["trustworthy_frac"])
            rows.append(r)
    return rows


def per_specimen_means(rows):
    """mean disagreement_rate per specimen, over all pair/directions, matching
    the eval script's own per-specimen printed line."""
    by_spec = defaultdict(list)
    for r in rows:
        if r["disagreement_rate"] is not None:
            by_spec[r["specimen"]].append(r["disagreement_rate"])
    return {k: sum(v) / len(v) for k, v in by_spec.items()}


def population_by_pair_direction(rows):
    by_pd = defaultdict(list)
    for r in rows:
        if r["disagreement_rate"] is not None:
            by_pd[(r["part_a"], r["part_b"], r["direction"])].append(r["disagreement_rate"])
    return {k: sum(v) / len(v) for k, v in by_pd.items()}


def main():
    data = {arm: load(path) for arm, path in CSVS.items()}
    for arm, rows in data.items():
        specs = {r["specimen"] for r in rows}
        pairs = {(r["part_a"], r["part_b"], r["direction"]) for r in rows}
        n_none = sum(1 for r in rows if r["disagreement_rate"] is None)
        print(f"{arm}: {len(rows)} rows, {len(specs)} specimens, {len(pairs)} pair/directions, "
              f"{n_none} rows with disagreement_rate=None")
        assert len(specs) == 50, f"expected 50 specimens, got {len(specs)}"

    pd_base = population_by_pair_direction(data["baseline"])
    pd_gentle = population_by_pair_direction(data["gentle"])
    all_keys = sorted(set(pd_base) | set(pd_gentle), key=lambda k: -abs(pd_gentle.get(k, 0) - pd_base.get(k, 0)))

    print("\n=== recomputed directly from per-specimen CSVs (mean over 50 specimens/pair) ===")
    print(f"{'pair':<10} {'dir':<10} {'other':<10} {'baseline':>10} {'gentle':>10} {'delta':>10}")
    worse, better, tie = 0, 0, 0
    for a, b, d in all_keys:
        base_v = pd_base.get((a, b, d))
        gen_v = pd_gentle.get((a, b, d))
        if base_v is None or gen_v is None:
            print(f"{a:<10} {d:<10} {b:<10} {'MISSING':>10}")
            continue
        delta = gen_v - base_v
        if abs(delta) < 1e-4:
            tie += 1
        elif delta > 0:
            worse += 1
        else:
            better += 1
        print(f"{a:<10} {d:<10} {b:<10} {base_v:>10.4f} {gen_v:>10.4f} {delta:>+10.4f}")

    print(f"\n{worse} worse (gentle disagrees more), {better} better, {tie} tie "
          f"(threshold |delta|<1e-4), out of {len(all_keys)} pair/directions")

    print("\n=== cross-check vs TASK3_HANDOFF.md table (log-derived) ===")
    max_abs_diff = 0.0
    for key, (h_base, h_gentle) in HANDOFF.items():
        r_base = pd_base.get(key)
        r_gentle = pd_gentle.get(key)
        if r_base is None or r_gentle is None:
            print(f"  {key}: NOT FOUND in recompute")
            continue
        d_base = abs(r_base - h_base)
        d_gentle = abs(r_gentle - h_gentle)
        max_abs_diff = max(max_abs_diff, d_base, d_gentle)
        flag = "  <-- MISMATCH" if (d_base > 0.0001 or d_gentle > 0.0001) else ""
        print(f"  {key}: handoff=({h_base:.4f},{h_gentle:.4f}) recompute=({r_base:.4f},{r_gentle:.4f}){flag}")
    print(f"\nmax abs diff handoff-vs-recompute across all checked rows: {max_abs_diff:.6f}")

    # per-specimen paired comparison (independent of population-level averaging)
    ps_base = per_specimen_means(data["baseline"])
    ps_gentle = per_specimen_means(data["gentle"])
    common = sorted(set(ps_base) & set(ps_gentle))
    assert len(common) == 50
    n_worse_spec = sum(1 for s in common if ps_gentle[s] > ps_base[s])
    n_better_spec = sum(1 for s in common if ps_gentle[s] < ps_base[s])
    n_tie_spec = len(common) - n_worse_spec - n_better_spec
    print(f"\n=== per-specimen (mean disagreement over all 20 pair/directions), n=50 ===")
    print(f"specimens with higher mean disagreement under gentle: {n_worse_spec}")
    print(f"specimens with lower mean disagreement under gentle: {n_better_spec}")
    print(f"tied: {n_tie_spec}")
    mean_base = sum(ps_base.values()) / len(ps_base)
    mean_gentle = sum(ps_gentle.values()) / len(ps_gentle)
    print(f"grand mean disagreement_rate: baseline={mean_base:.4f} gentle={mean_gentle:.4f} "
          f"delta={mean_gentle - mean_base:+.4f}")

    # sign test style summary restricted to the gaster-legs pair specifically
    # (the only pair any penetration term targets, per the Hard Rule)
    print("\n=== gaster-legs only (the targeted pair) ===")
    for d in ("A_into_B", "B_into_A"):
        key = ("gaster", "legs", d)
        print(f"  {d}: baseline={pd_base[key]:.4f} gentle={pd_gentle[key]:.4f} delta={pd_gentle[key]-pd_base[key]:+.4f}")


if __name__ == "__main__":
    main()
