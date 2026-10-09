"""TASK 5 (pair-scoped gentle+offset+scalecap) Hard Rule scoring: TASK4's pair-scoped
gaster-legs-only gentle+offset arm, plus the newly-wired w_scale=0.052 (validated
elsewhere, T0.4) / w_trans=1.0 (uncalibrated) barriers on log_beta_scales/betas_trans
(fitter_3d/joint_limits.py, wired into fitter_3d/trainer.py for the first time this run).
One bench50 arm, unseeded, compared against TASK2's offset-baseline (unaffected by this
change) and TASK4's pairscoped-gentle-no-scalecap arm for context.
"""

import csv
import os
import statistics as stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUNS = {
    "baseline": "fit3d_results_all_offset_baseline",
    "gentle_pairscoped": "fit3d_results_all_offset_gentle_pairscoped",
    "gentle_pairscoped_scalecap": "fit3d_results_all_offset_gentle_pairscoped_scalecap",
}
INTEGRITY_CSV = {
    "baseline": "diagnostics/khaoula_review/task2_metrics_offset_baseline.csv",
    "gentle_pairscoped": "diagnostics/khaoula_review/task4_metrics_offset_gentle_pairscoped.csv",
    "gentle_pairscoped_scalecap": "diagnostics/khaoula_review/task5_metrics_offset_gentle_pairscoped_scalecap.csv",
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
        print(f"  gaster-legs count <= -15%?                      {gl_pct:+.1f}%  "
              f"{'PASS' if gl_pct <= -15 else 'FAIL'}")
        tot_pct = 100 * (mg_tot - mb_tot) / mb_tot
        print(f"  all-pairs aggregate <= 0%?                      {tot_pct:+.1f}%  "
              f"{'PASS' if tot_pct <= 0 else 'FAIL'}")
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
    gps = load_arm("gentle_pairscoped")
    gps_sc = load_arm("gentle_pairscoped_scalecap")

    report_pair("TASK5 primary: baseline vs pairscoped-gentle+scalecap (unseeded)", b, gps_sc,
                task_ref="pre-registered TASK5 panel")

    print("=== context: TASK4 pairscoped-gentle, no scalecap (for reference, same baseline) ===")
    report_pair("baseline vs pairscoped-gentle (TASK4, unseeded)", b, gps)

    print("=== does scalecap change TASK4's own result? pairscoped-gentle vs pairscoped-gentle+scalecap ===")
    specs = sorted(set(gps) & set(gps_sc))
    d_gl = stats.mean(gps_sc[s]["gl_total"] - gps[s]["gl_total"] for s in specs)
    d_tot = stats.mean(gps_sc[s]["total"] - gps[s]["total"] for s in specs)
    d_fs = stats.mean(gps_sc[s]["fscore01"] - gps[s]["fscore01"] for s in specs)
    d_dm = stats.mean(gps_sc[s]["deform_mag"] - gps[s]["deform_mag"] for s in specs)
    d_el = stats.mean(gps_sc[s]["edge_logratio"] - gps[s]["edge_logratio"] for s in specs)
    print(f"gaster-legs count delta (scalecap - no scalecap): {d_gl:+.2f}")
    print(f"all-pairs count delta:                            {d_tot:+.2f}")
    print(f"fscore@0.01 delta:                                {d_fs:+.4f}")
    print(f"deform_mag delta:                                 {d_dm:+.5f}")
    print(f"edge_logratio delta:                              {d_el:+.5f}")

    print("\n=== full per-pair table, unseeded ===")
    pb = per_pair_table(RUNS["baseline"])
    pgps = per_pair_table(RUNS["gentle_pairscoped"])
    pgps_sc = per_pair_table(RUNS["gentle_pairscoped_scalecap"])
    all_pairs = sorted(set(pb) | set(pgps) | set(pgps_sc))
    print(f"{'pair':<20} {'baseline':>10} {'pairscoped':>14} {'pairscoped_scalecap':>20}")
    for p in all_pairs:
        print(f"{str(p):<20} {pb.get(p,0):>10.2f} {pgps.get(p,0):>14.2f} {pgps_sc.get(p,0):>20.2f}")


if __name__ == "__main__":
    main()
