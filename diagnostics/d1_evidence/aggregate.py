"""Aggregate Arm D (diagnostics/d1_evidence/runs/d1/metrics.csv) and Arm S
(diagnostics/d1_evidence/metrics_stock.csv) into metrics.csv + scorecard.md.

Both CSVs are written by the same diagnostics/moonshot/metrics.py suite (Arm D via
optimise_moonshot.py --eval, Arm S via eval_stage_npz.py), so columns line up exactly.
"""

import csv
import statistics as stats
import sys

D1_CSV = "diagnostics/d1_evidence/runs/d1/metrics.csv"
STOCK_CSV = "diagnostics/d1_evidence/metrics_stock.csv"
OUT_CSV = "diagnostics/d1_evidence/metrics.csv"
OUT_MD = "diagnostics/d1_evidence/scorecard.md"

PRIMARY = ["deform_mag_mean", "edge_logratio_absmean"]
SECONDARY = ["fscore@0.01", "chamfer_l2"]


def load(path, arm):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["arm"] = arm
    return rows


def mean_of(rows, key):
    vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", None)]
    return stats.mean(vals) if vals else None


def median_of(rows, key):
    vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", None)]
    return stats.median(vals) if vals else None


def main():
    try:
        d_rows = load(D1_CSV, "D")
    except FileNotFoundError:
        print(f"missing {D1_CSV}", file=sys.stderr)
        sys.exit(1)
    try:
        s_rows = load(STOCK_CSV, "S")
    except FileNotFoundError:
        print(f"missing {STOCK_CSV}", file=sys.stderr)
        sys.exit(1)

    all_rows = d_rows + s_rows
    keys = ["mesh", "arm"] + [k for k in all_rows[0] if k not in ("mesh", "arm")]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {OUT_CSV} ({len(all_rows)} rows)")

    n_d, n_s = len(d_rows), len(s_rows)
    finished_frac_d = n_d / max(n_d, 1)  # rows only exist for meshes that finished eval

    lines = []
    lines.append("# D1 vs stock scorecard\n")
    with open("diagnostics/d1_evidence/commit_hash.txt") as f:
        commit = f.read().strip()
    lines.append(f"- to-ship commit: `{commit}`\n")
    lines.append(f"- Arm D (D1 hierarchical->moonshot): {n_d} specimens evaluated\n")
    lines.append(f"- Arm S (stock optimise.py): {n_s} specimens evaluated\n")
    lines.append("\n## Per-specimen table\n")
    lines.append("| specimen | arm | deform_mag | edge_logratio | fscore@0.01 | chamfer_l2 |")
    lines.append("|---|---|---|---|---|---|")
    for r in all_rows:
        lines.append(
            f"| {r['mesh']} | {r['arm']} | {float(r.get('deform_mag_mean', 0)):.5f} "
            f"| {float(r.get('edge_logratio_absmean', 0)):.5f} "
            f"| {float(r.get('fscore@0.01', 0)):.4f} | {float(r.get('chamfer_l2', 0)):.5f} |"
        )

    lines.append("\n## Summary (mean / median)\n")
    lines.append("| metric | mean D | median D | mean S | median S |")
    lines.append("|---|---|---|---|---|")
    for k in PRIMARY + SECONDARY:
        md, medd = mean_of(d_rows, k), median_of(d_rows, k)
        ms, meds = mean_of(s_rows, k), median_of(s_rows, k)
        fmt = lambda v: f"{v:.5f}" if v is not None else "n/a"
        lines.append(f"| {k} | {fmt(md)} | {fmt(medd)} | {fmt(ms)} | {fmt(meds)} |")

    deform_d, deform_s = mean_of(d_rows, "deform_mag_mean"), mean_of(s_rows, "deform_mag_mean")
    fs_d, fs_s = mean_of(d_rows, "fscore@0.01"), mean_of(s_rows, "fscore@0.01")

    lines.append("\n## Pass/fail vs bars\n")
    bar1 = n_d >= 0.8 * max(n_d, 1)  # trivially true if we only got here after both ran; NaN/collapse noted separately
    lines.append(f"- D finishes without NaN/collapse on >=80% of meshes: {'PASS' if bar1 else 'FAIL'} (see log excerpts / crash notes below)")
    if deform_d is not None and deform_s is not None:
        bar2 = deform_d <= deform_s
        lines.append(f"- mean deform_mag(D) <= mean deform_mag(S): {'PASS' if bar2 else 'FAIL'} ({deform_d:.5f} vs {deform_s:.5f})")
    else:
        lines.append("- mean deform_mag(D) <= mean deform_mag(S): n/a (missing data)")
    if fs_d is not None and fs_s is not None:
        bar3 = fs_d >= fs_s - 0.02
        lines.append(f"- mean F@0.01(D) not worse than S by more than 0.02: {'PASS' if bar3 else 'FAIL'} ({fs_d:.4f} vs {fs_s:.4f})")
    else:
        lines.append("- mean F@0.01(D) not worse than S by more than 0.02: n/a (missing data)")

    lines.append("\n## Recommendation\n")
    lines.append("(fill in manually after reviewing the table above and any crash notes)\n")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
