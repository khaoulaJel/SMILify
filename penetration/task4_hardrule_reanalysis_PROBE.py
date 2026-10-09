"""TASK 4 (pair-scoped gentle+offset) Hard Rule scoring: restricts the
w_penetration LOSS to gaster-legs only (fitter_3d/ants_cfg_all_offset_gentle_pairscoped.yaml,
via the new `penetration_train_pairs` Stage kwarg in fitter_3d/trainer.py), keeping
scheme:'all', gentle weights 0.02/0.05, w_offset 5.0/2.0, w_limit in all stages
-- everything else identical to TASK 2. Compares against TASK 2's already-run
baseline+offset arm (unaffected by this change, w_penetration=0) and TASK 2's
own all-10-pairs gentle+offset arm for context.

Primary comparison (exact, seed-matched): unseeded pairscoped-gentle vs unseeded
offset-baseline. Seed check: seed1 pairscoped-gentle compared against the SAME
unseeded offset-baseline (no seed-matched baseline was run -- flagged explicitly,
this is an approximation) plus a same-arm cross-seed stability check against the
unseeded pairscoped-gentle numbers.
"""

import csv
import os
import statistics as stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUNS = {
    "baseline": "fit3d_results_all_offset_baseline",
    "gentle_all10": "fit3d_results_all_offset_gentle",
    "gentle_pairscoped": "fit3d_results_all_offset_gentle_pairscoped",
    "gentle_pairscoped_seed1": "fit3d_results_seeded/offset_gentle_pairscoped_seed1",
}
INTEGRITY_CSV = {
    "baseline": "diagnostics/khaoula_review/task2_metrics_offset_baseline.csv",
    "gentle_all10": "diagnostics/khaoula_review/task2_metrics_offset_gentle.csv",
    "gentle_pairscoped": "diagnostics/khaoula_review/task4_metrics_offset_gentle_pairscoped.csv",
    "gentle_pairscoped_seed1": "diagnostics/khaoula_review/task4_metrics_offset_gentle_pairscoped_seed1.csv",
}


def gaster_legs_count(run_dir):
    path = os.path.join(REPO, run_dir, "Stage_3_deform_fine_per_pair_penetration.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["part_a"] == "gaster" and row["part_b"] == "legs":
                spec = row["specimen"]
                out.setdefault(spec, {})[row["direction"]] = float(row["num_penetrating"])
    return {s: {"gl_total": d["A_into_B"] + d["B_into_A"]} for s, d in out.items()}


def all_pairs_count(run_dir):
    path = os.path.join(REPO, run_dir, "Stage_3_deform_fine_per_pair_penetration.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out.setdefault(row["specimen"], 0.0)
            out[row["specimen"]] += float(row["num_penetrating"])
    return out


def per_pair_table(run_dir):
    path = os.path.join(REPO, run_dir, "Stage_3_deform_fine_per_pair_penetration.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["part_a"], row["part_b"])
            out.setdefault(key, []).append(float(row["num_penetrating"]))
    return {k: stats.mean(v) for k, v in out.items()}


def fscore01(run_dir):
    path = os.path.join(REPO, run_dir, "Stage_3_deform_fine_eval_metrics.csv")
    return {row["specimen"]: float(row["f_score@0.01"]) for row in csv.DictReader(open(path))}


def integrity(csv_path):
    path = os.path.join(REPO, csv_path)
    return {row["mesh"]: {"deform_mag": float(row["deform_mag_mean"]),
                           "edge_logratio": float(row["edge_logratio_absmean"])}
            for row in csv.DictReader(open(path))}


def load_arm(key, expect_n=50):
    run_dir = RUNS[key]
    gl = gaster_legs_count(run_dir)
    tot = all_pairs_count(run_dir)
    fs = fscore01(run_dir)
    it = integrity(INTEGRITY_CSV[key])
    specs = sorted(set(gl) & set(fs) & set(it))
    assert len(specs) == expect_n, f"{key}: {len(specs)} specimens, expected {expect_n}"
    return {s: {**gl[s], "total": tot[s], "fscore01": fs[s], **it[s]} for s in specs}


def report_pair(label, b, g, task_ref=None):
    specs = sorted(set(b) & set(g))
    n = len(specs)
    mb_gl = stats.mean(b[s]["gl_total"] for s in specs)
    mg_gl = stats.mean(g[s]["gl_total"] for s in specs)
    mb_tot = stats.mean(b[s]["total"] for s in specs)
    mg_tot = stats.mean(g[s]["total"] for s in specs)
    d_fs = [g[s]["fscore01"] - b[s]["fscore01"] for s in specs]
    d_dm = [g[s]["deform_mag"] - b[s]["deform_mag"] for s in specs]
    d_el = [g[s]["edge_logratio"] - b[s]["edge_logratio"] for s in specs]

    print(f"=== {label} (n={n}) ===")
    print(f"gaster-legs count   baseline {mb_gl:.2f}  gentle {mg_gl:.2f}  "
          f"delta {mg_gl-mb_gl:+.2f} ({100*(mg_gl-mb_gl)/mb_gl:+.1f}%)")
    print(f"all-pairs count     baseline {mb_tot:.2f}  gentle {mg_tot:.2f}  "
          f"delta {mg_tot-mb_tot:+.2f} ({100*(mg_tot-mb_tot)/mb_tot:+.1f}%)")
    print(f"fscore@0.01 delta   mean {stats.mean(d_fs):+.4f}  median {stats.median(d_fs):+.4f}  "
          f"min {min(d_fs):+.4f}  max {max(d_fs):+.4f}")
    print(f"deform_mag delta    mean {stats.mean(d_dm):+.5f} "
          f"({100*stats.mean(d_dm)/stats.mean(b[s]['deform_mag'] for s in specs):+.2f}%)")
    print(f"edge_logratio delta mean {stats.mean(d_el):+.5f} "
          f"({100*stats.mean(d_el)/stats.mean(b[s]['edge_logratio'] for s in specs):+.2f}%)")

    if task_ref:
        print(f"\n--- pre-registered success criteria check ({task_ref}) ---")
        gl_pct = 100 * (mg_gl - mb_gl) / mb_gl
        print(f"  gaster-legs count <= TASK2 level (~-20%)?      {gl_pct:+.1f}%  "
              f"{'PASS' if gl_pct <= -18 else 'CHECK'}")
        tot_pct = 100 * (mg_tot - mb_tot) / mb_tot
        print(f"  all-pairs aggregate <= 0% (or < TASK2's +5.4%)? {tot_pct:+.1f}%  "
              f"{'PASS' if tot_pct <= 5.4 else 'FAIL'}")
        mean_fs = stats.mean(d_fs)
        worst_fs = min(d_fs)
        print(f"  mean fscore delta >= -0.01?                    {mean_fs:+.4f}  "
              f"{'PASS' if mean_fs >= -0.01 else 'FAIL'}")
        print(f"  worst specimen fscore delta >= -0.08?           {worst_fs:+.4f}  "
              f"{'PASS' if worst_fs >= -0.08 else 'FAIL'}")

    print("\n--- top 5 outliers by |delta gaster-legs count| and |delta fscore@0.01| ---")
    rows = [(s, g[s]["gl_total"] - b[s]["gl_total"], g[s]["fscore01"] - b[s]["fscore01"]) for s in specs]
    for lbl, key in [("count", 1), ("fscore", 2)]:
        top = sorted(rows, key=lambda r: -abs(r[key]))[:5]
        print(f"  top 5 by |delta {lbl}|:")
        for s, dc, df in top:
            print(f"    {s:55s} d_count={dc:+8.1f}  d_fscore01={df:+.4f}")
    print()
    return specs


def main():
    b = load_arm("baseline")
    g10 = load_arm("gentle_all10")
    gps = load_arm("gentle_pairscoped")

    report_pair("TASK4 primary: baseline vs pairscoped-gentle (unseeded)", b, gps,
                task_ref="vs TASK2's +5.4% aggregate / -20.2% gaster-legs")

    print("=== context: TASK2 all-10-pairs gentle (for reference, same baseline) ===")
    report_pair("baseline vs gentle_all10 (TASK2, unseeded)", b, g10)

    print("=== full per-pair table, unseeded (why restricting scope changes the aggregate) ===")
    pb = per_pair_table(RUNS["baseline"])
    pg10 = per_pair_table(RUNS["gentle_all10"])
    pgps = per_pair_table(RUNS["gentle_pairscoped"])
    all_pairs = sorted(set(pb) | set(pg10) | set(pgps))
    print(f"{'pair':<20} {'baseline':>10} {'gentle_all10':>14} {'gentle_pairscoped':>18}")
    for p in all_pairs:
        print(f"{str(p):<20} {pb.get(p,0):>10.2f} {pg10.get(p,0):>14.2f} {pgps.get(p,0):>18.2f}")
    print()

    try:
        gps1 = load_arm("gentle_pairscoped_seed1")
    except (FileNotFoundError, AssertionError) as e:
        print(f"seed1 run not available yet: {e}")
        return

    print("=== SEED CHECK: seed1 pairscoped-gentle vs unseeded offset-baseline ===")
    print("CAVEAT: no seed-matched baseline was run for this cheap check -- this delta")
    print("mixes a seed1-treatment arm against an unseeded baseline, so treat as approximate.")
    report_pair("baseline (unseeded) vs pairscoped-gentle (seed1)", b, gps1)

    print("=== same-arm cross-seed stability: unseeded vs seed1 pairscoped-gentle (no baseline diff) ===")
    specs = sorted(set(gps) & set(gps1))
    d_gl = [gps1[s]["gl_total"] - gps[s]["gl_total"] for s in specs]
    d_tot = [gps1[s]["total"] - gps[s]["total"] for s in specs]
    print(f"n={len(specs)}")
    print(f"gaster-legs count: unseeded mean {stats.mean(gps[s]['gl_total'] for s in specs):.2f}  "
          f"seed1 mean {stats.mean(gps1[s]['gl_total'] for s in specs):.2f}  "
          f"mean |delta| across specimens {stats.mean(abs(x) for x in d_gl):.2f}")
    print(f"all-pairs count:   unseeded mean {stats.mean(gps[s]['total'] for s in specs):.2f}  "
          f"seed1 mean {stats.mean(gps1[s]['total'] for s in specs):.2f}  "
          f"mean |delta| across specimens {stats.mean(abs(x) for x in d_tot):.2f}")


if __name__ == "__main__":
    main()
