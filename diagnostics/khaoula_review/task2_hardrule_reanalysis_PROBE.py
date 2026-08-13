"""TASK 2 re-analysis under the Hard Rule: w_offset (5.0/2.0, D1 setting) added on
top of TASK 1's scheme:'all' + gentle-penetration base config, baseline vs gentle,
bench50_clean, unseeded. Compares against TASK 1's unseeded numbers.
"""

import csv
import os
import statistics as stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUNS = {
    "baseline": "fit3d_results_all_offset_baseline",
    "gentle": "fit3d_results_all_offset_gentle",
}
INTEGRITY_CSV = {
    "baseline": "diagnostics/khaoula_review/task2_metrics_offset_baseline.csv",
    "gentle": "diagnostics/khaoula_review/task2_metrics_offset_gentle.csv",
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


def fscore01(run_dir):
    path = os.path.join(REPO, run_dir, "Stage_3_deform_fine_eval_metrics.csv")
    return {row["specimen"]: float(row["f_score@0.01"]) for row in csv.DictReader(open(path))}


def integrity(csv_path):
    path = os.path.join(REPO, csv_path)
    return {row["mesh"]: {"deform_mag": float(row["deform_mag_mean"]),
                           "edge_logratio": float(row["edge_logratio_absmean"])}
            for row in csv.DictReader(open(path))}


def load_arm(key):
    run_dir = RUNS[key]
    gl = gaster_legs_count(run_dir)
    tot = all_pairs_count(run_dir)
    fs = fscore01(run_dir)
    it = integrity(INTEGRITY_CSV[key])
    specs = sorted(set(gl) & set(fs) & set(it))
    assert len(specs) == 50, f"{key}: {len(specs)} specimens"
    return {s: {**gl[s], "total": tot[s], "fscore01": fs[s], **it[s]} for s in specs}


def main():
    b = load_arm("baseline")
    g = load_arm("gentle")
    specs = sorted(b)
    pairs = [(b[s], g[s]) for s in specs]

    mb_gl = stats.mean(x["gl_total"] for x, _ in pairs)
    mg_gl = stats.mean(y["gl_total"] for _, y in pairs)
    mb_tot = stats.mean(x["total"] for x, _ in pairs)
    mg_tot = stats.mean(y["total"] for _, y in pairs)
    d_fs = [y["fscore01"] - x["fscore01"] for x, y in pairs]
    d_dm = [y["deform_mag"] - x["deform_mag"] for x, y in pairs]
    d_el = [y["edge_logratio"] - x["edge_logratio"] for x, y in pairs]

    print(f"=== TASK 2 (w_offset 5.0/2.0 added, unseeded, n={len(specs)}) ===")
    print(f"gaster-legs count   baseline {mb_gl:.2f}  gentle {mg_gl:.2f}  delta {mg_gl-mb_gl:+.2f} ({100*(mg_gl-mb_gl)/mb_gl:+.1f}%)")
    print(f"all-pairs count     baseline {mb_tot:.2f}  gentle {mg_tot:.2f}  delta {mg_tot-mb_tot:+.2f} ({100*(mg_tot-mb_tot)/mb_tot:+.1f}%)")
    print(f"fscore@0.01 delta   mean {stats.mean(d_fs):+.4f}  median {stats.median(d_fs):+.4f}  min {min(d_fs):+.4f}  max {max(d_fs):+.4f}")
    print(f"deform_mag delta    mean {stats.mean(d_dm):+.5f} ({100*stats.mean(d_dm)/stats.mean(x['deform_mag'] for x,_ in pairs):+.2f}%)")
    print(f"edge_logratio delta mean {stats.mean(d_el):+.5f} ({100*stats.mean(d_el)/stats.mean(x['edge_logratio'] for x,_ in pairs):+.2f}%)")

    print("\n--- vs TASK 1 (no offset penalty), unseeded ---")
    print("TASK1: gaster-legs -20.8%, all-pairs +16.8%, fscore mean -0.0048, deform_mag +4.13%, edge_logratio +5.63%")

    print("\n=== absolute integrity level (baseline arm only): does w_offset itself cost integrity vs TASK1's baseline? ===")
    print(f"TASK2 offset-baseline deform_mag mean: {stats.mean(x['deform_mag'] for x,_ in pairs):.5f}")
    print(f"TASK2 offset-baseline edge_logratio mean: {stats.mean(x['edge_logratio'] for x,_ in pairs):.5f}")

    print("\n=== OUTLIER FLAGS (top 5 by |delta gaster-legs count| and |delta fscore@0.01|) ===")
    rows = [(s, g[s]["gl_total"] - b[s]["gl_total"], g[s]["fscore01"] - b[s]["fscore01"]) for s in specs]
    for label, key in [("count", 1), ("fscore", 2)]:
        top = sorted(rows, key=lambda r: -abs(r[key]))[:5]
        print(f"  top 5 by |delta {label}|:")
        for s, dc, df in top:
            print(f"    {s:55s} d_count={dc:+8.1f}  d_fscore01={df:+.4f}")


if __name__ == "__main__":
    main()
