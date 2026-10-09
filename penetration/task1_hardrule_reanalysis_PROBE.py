"""TASK 1 re-analysis under the project's Hard Rule metric panel.

Re-scores the ALREADY-COMPLETED bench50_clean, scheme:'all', gentle-vs-baseline
penetration ablation (scripts/penetration_joint_study/FINDINGS.md step 1 + the
reseeded rerun) using the four Hard Rule instruments:

  1. per-pair count, gaster-legs ONLY (the only pair any penetration term targets)
  2. F-score@0.01 (surface-fit sanity metric)
  3. deform_mag, edge_logratio (integrity)
  4. outlier flags for visual-render follow-up

No new GPU training. Reads:
  - fit3d_results_all_baseline/, fit3d_results_all_gentle/           (unseeded arm)
  - fit3d_results_seeded/{baseline,gentle}_seed{0,1,2}/               (reseeded arm)
  - Stage_3_deform_fine_per_pair_penetration.csv  (already on disk, per run)
  - Stage_3_deform_fine_eval_metrics.csv          (already on disk, per run; f_score@0.01)
  - diagnostics/khaoula_review/task1_metrics_*.csv (this session; deform_mag/edge_logratio,
    produced by diagnostics/moonshot/eval_run.py off the same saved Stage_3 verts)
"""

import csv
import os
import statistics as stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUNS = {
    "unseeded_baseline": "fit3d_results_all_baseline",
    "unseeded_gentle": "fit3d_results_all_gentle",
    "seed0_baseline": "fit3d_results_seeded/baseline_seed0",
    "seed0_gentle": "fit3d_results_seeded/gentle_seed0",
    "seed1_baseline": "fit3d_results_seeded/baseline_seed1",
    "seed1_gentle": "fit3d_results_seeded/gentle_seed1",
    "seed2_baseline": "fit3d_results_seeded/baseline_seed2",
    "seed2_gentle": "fit3d_results_seeded/gentle_seed2",
}

INTEGRITY_CSV = {
    "unseeded_baseline": "diagnostics/khaoula_review/task1_metrics_all_baseline.csv",
    "unseeded_gentle": "diagnostics/khaoula_review/task1_metrics_fit3d_results_all_gentle.csv",
    "seed0_baseline": "diagnostics/khaoula_review/task1_metrics_fit3d_results_seeded_baseline_seed0.csv",
    "seed0_gentle": "diagnostics/khaoula_review/task1_metrics_fit3d_results_seeded_gentle_seed0.csv",
    "seed1_baseline": "diagnostics/khaoula_review/task1_metrics_fit3d_results_seeded_baseline_seed1.csv",
    "seed1_gentle": "diagnostics/khaoula_review/task1_metrics_fit3d_results_seeded_gentle_seed1.csv",
    "seed2_baseline": "diagnostics/khaoula_review/task1_metrics_fit3d_results_seeded_baseline_seed2.csv",
    "seed2_gentle": "diagnostics/khaoula_review/task1_metrics_fit3d_results_seeded_gentle_seed2.csv",
}


def gaster_legs_count(run_dir):
    """Per-specimen gaster<->legs penetrating-vertex count (both directions summed)."""
    path = os.path.join(REPO, run_dir, "Stage_3_deform_fine_per_pair_penetration.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["part_a"] == "gaster" and row["part_b"] == "legs":
                spec = row["specimen"]
                out.setdefault(spec, {"A_into_B": None, "B_into_A": None})
                out[spec][row["direction"]] = float(row["num_penetrating"])
    totals = {}
    for spec, d in out.items():
        assert d["A_into_B"] is not None and d["B_into_A"] is not None, spec
        totals[spec] = {
            "gl_total": d["A_into_B"] + d["B_into_A"],
            "gl_A_into_B": d["A_into_B"],  # gaster verts penetrating into legs
            "gl_B_into_A": d["B_into_A"],  # legs verts penetrating into gaster
        }
    return totals


def fscore01(run_dir):
    path = os.path.join(REPO, run_dir, "Stage_3_deform_fine_eval_metrics.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["specimen"]] = float(row["f_score@0.01"])
    return out


def integrity(csv_path):
    path = os.path.join(REPO, csv_path)
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["mesh"]] = {
                "deform_mag": float(row["deform_mag_mean"]),
                "edge_logratio": float(row["edge_logratio_absmean"]),
            }
    return out


def load_arm(key):
    run_dir = RUNS[key]
    gl = gaster_legs_count(run_dir)
    fs = fscore01(run_dir)
    it = integrity(INTEGRITY_CSV[key])
    specs = sorted(set(gl) & set(fs) & set(it))
    assert len(specs) == 50, f"{key}: expected 50 specimens, got {len(specs)}"
    rows = {}
    for s in specs:
        rows[s] = {**gl[s], "fscore01": fs[s], **it[s]}
    return rows


def summarize(pairs, label):
    """pairs: list of (baseline_row, gentle_row) dicts, one per specimen."""
    n = len(pairs)
    d_gl = [g["gl_total"] - b["gl_total"] for b, g in pairs]
    d_fs = [g["fscore01"] - b["fscore01"] for b, g in pairs]
    d_dm = [g["deform_mag"] - b["deform_mag"] for b, g in pairs]
    d_el = [g["edge_logratio"] - b["edge_logratio"] for b, g in pairs]
    mean_b_gl = stats.mean(b["gl_total"] for b, g in pairs)
    mean_g_gl = stats.mean(g["gl_total"] for b, g in pairs)
    print(f"\n=== {label} (n={n} specimens) ===")
    print(f"  gaster-legs count      baseline mean {mean_b_gl:.2f}  gentle mean {mean_g_gl:.2f}  "
          f"delta {mean_g_gl - mean_b_gl:+.2f} ({100*(mean_g_gl - mean_b_gl)/mean_b_gl:+.1f}%)")
    print(f"  gaster-legs count      improved(gentle<baseline): {sum(1 for x in d_gl if x < 0)}/{n}  "
          f"worse: {sum(1 for x in d_gl if x > 0)}/{n}  flat: {sum(1 for x in d_gl if x == 0)}/{n}")
    print(f"  fscore@0.01 delta      mean {stats.mean(d_fs):+.4f}  median {stats.median(d_fs):+.4f}  "
          f"min {min(d_fs):+.4f}  max {max(d_fs):+.4f}")
    print(f"  deform_mag delta       mean {stats.mean(d_dm):+.5f} ({100*stats.mean(d_dm)/stats.mean(b['deform_mag'] for b,g in pairs):+.2f}%)")
    print(f"  edge_logratio delta    mean {stats.mean(d_el):+.5f} ({100*stats.mean(d_el)/stats.mean(b['edge_logratio'] for b,g in pairs):+.2f}%)")
    return d_gl, d_fs


def main():
    arms = {k: load_arm(k) for k in RUNS}

    # --- unseeded arm ---
    specs = sorted(arms["unseeded_baseline"])
    pairs = [(arms["unseeded_baseline"][s], arms["unseeded_gentle"][s]) for s in specs]
    d_gl_uns, d_fs_uns = summarize(pairs, "UNSEEDED (single run, matches FINDINGS.md step 1)")

    # --- 3-seed reruns, pooled ---
    all_pairs = []
    per_seed = {}
    for si in range(3):
        b = arms[f"seed{si}_baseline"]
        g = arms[f"seed{si}_gentle"]
        sp = sorted(set(b) & set(g))
        pr = [(b[s], g[s]) for s in sp]
        per_seed[si] = pr
        all_pairs += pr
    d_gl_seeded, d_fs_seeded = summarize(all_pairs, "3-SEED RERUN, POOLED (150 specimen-seeds)")

    # per-seed breakdown for gaster-legs count only
    print("\n--- gaster-legs count, per seed ---")
    for si in range(3):
        pr = per_seed[si]
        mb = stats.mean(b["gl_total"] for b, g in pr)
        mg = stats.mean(g["gl_total"] for b, g in pr)
        print(f"  seed {si}: baseline mean {mb:.2f}  gentle mean {mg:.2f}  delta {100*(mg-mb)/mb:+.2f}%")

    # majority-vote across seeds, per specimen (Fabian's method B, restricted to gaster-legs)
    specs50 = sorted(arms["seed0_baseline"])
    flips = 0
    improved_majority = 0
    worse_majority = 0
    for s in specs50:
        deltas = [arms[f"seed{si}_gentle"][s]["gl_total"] - arms[f"seed{si}_baseline"][s]["gl_total"] for si in range(3)]
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
        if len(set(s for s in signs if s != 0)) > 1:
            flips += 1
        elif sum(signs) < 0:
            improved_majority += 1
        elif sum(signs) > 0:
            worse_majority += 1
    print(f"\n  per-specimen seed-agreement (gaster-legs count only): "
          f"{improved_majority}/50 unanimous-improved, {worse_majority}/50 unanimous-worse, "
          f"{flips}/50 flip sign across seeds")

    # --- outlier flagging for render follow-up ---
    print("\n=== OUTLIER FLAGS (largest |delta gaster-legs count| and |delta fscore@0.01|, unseeded arm) ===")
    rows = []
    for s in specs:
        b, g = arms["unseeded_baseline"][s], arms["unseeded_gentle"][s]
        rows.append((s, g["gl_total"] - b["gl_total"], g["fscore01"] - b["fscore01"]))
    by_count = sorted(rows, key=lambda r: -abs(r[1]))[:5]
    by_fscore = sorted(rows, key=lambda r: -abs(r[2]))[:5]
    print("  top 5 by |delta gaster-legs count|:")
    for s, dc, df in by_count:
        print(f"    {s:55s} d_count={dc:+8.1f}  d_fscore01={df:+.4f}")
    print("  top 5 by |delta fscore@0.01|:")
    for s, dc, df in by_fscore:
        print(f"    {s:55s} d_count={dc:+8.1f}  d_fscore01={df:+.4f}")


if __name__ == "__main__":
    main()
