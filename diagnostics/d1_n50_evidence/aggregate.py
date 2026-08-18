"""Aggregate 3-seed x N=50 D1 evidence into metrics.csv + scorecard.md, evaluated against the
pre-registered promotion criteria in diagnostics/d1_n50_evidence/PROMOTION_CRITERIA.md.
"""

import csv
import statistics as stats

SEEDS = [0, 1, 2]
CSV_TMPL = "diagnostics/d1_n50_evidence/runs/seed{s}/d1/metrics.csv"
OUT_CSV = "diagnostics/d1_n50_evidence/metrics.csv"
OUT_MD = "diagnostics/d1_n50_evidence/scorecard.md"

PRIMARY = ["edge_logratio_absmean", "deform_mag_mean"]
INTEGRITY = ["edge_logratio_absmean", "deform_mag_mean", "deform_mag_p95", "deform_mag_max",
             "folded_face_frac", "dihedral_p99", "tri_quality_mean"]
SECONDARY = ["fscore@0.01", "chamfer_l2"]

# Task 1's N=10 D1 numbers (diagnostics/d1_evidence/scorecard.md), reference for bar 3.
TASK1_N10 = {
    "edge_logratio_absmean": 0.14856,
    "deform_mag_mean": 0.00401,
}


def load(seed):
    with open(CSV_TMPL.format(s=seed)) as f:
        rows = list(csv.DictReader(f))
    return {r["mesh"]: r for r in rows}


def main():
    by_seed = {s: load(s) for s in SEEDS}
    meshes = sorted(set.intersection(*(set(d) for d in by_seed.values())))
    assert len(meshes) == 50, f"expected 50 specimens common to all 3 seeds, got {len(meshes)}"

    all_rows = []
    for s in SEEDS:
        for m in meshes:
            r = dict(by_seed[s][m])
            r["seed"] = s
            all_rows.append(r)
    keys = ["mesh", "seed"] + [k for k in all_rows[0] if k not in ("mesh", "seed")]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    # ---- Bar 1: catastrophic per-specimen failures ----
    catastrophic = []
    per_specimen_deform_max = {m: [] for m in meshes}
    for s in SEEDS:
        for m in meshes:
            r = by_seed[s][m]
            vals = {k: float(r[k]) for k in INTEGRITY + SECONDARY if r.get(k) not in ("", None)}
            for k, v in vals.items():
                if v != v:  # NaN check
                    catastrophic.append((s, m, f"NaN in {k}"))
            ff = float(r["folded_face_frac"])
            if ff > 0.10:
                catastrophic.append((s, m, f"folded_face_frac={ff:.4f} > 0.10"))
            per_specimen_deform_max[m].append(float(r["deform_mag_max"]))
    for m in meshes:
        vals = per_specimen_deform_max[m]
        med = stats.median(vals)
        for s, v in zip(SEEDS, vals):
            if med > 0 and v > 10 * med:
                catastrophic.append((s, m, f"deform_mag_max={v:.5f} > 10x cross-seed median {med:.5f}"))
    bar1_pass = len(catastrophic) == 0

    # ---- Bar 2: cross-seed variance (CV) on primary metrics ----
    cv_by_metric = {}
    for k in PRIMARY:
        cvs = []
        for m in meshes:
            vals = [float(by_seed[s][m][k]) for s in SEEDS]
            mean = stats.mean(vals)
            if mean == 0:
                continue
            sd = stats.stdev(vals) if len(vals) > 1 else 0.0
            cvs.append(abs(sd / mean))
        cv_by_metric[k] = stats.mean(cvs) if cvs else None
    bar2_pass = all(v is not None and v < 0.15 for v in cv_by_metric.values())

    # ---- Bar 3: N=50 vs Task 1's N=10 numbers, per seed ----
    n50_means = {s: {} for s in SEEDS}
    for s in SEEDS:
        for k in PRIMARY:
            vals = [float(by_seed[s][m][k]) for m in meshes]
            n50_means[s][k] = stats.mean(vals)
    bar3_fail_details = []
    for s in SEEDS:
        for k in PRIMARY:
            ref = TASK1_N10[k]
            rel = (n50_means[s][k] - ref) / ref
            if rel > 0.15:
                bar3_fail_details.append(f"seed{s} {k}: {n50_means[s][k]:.5f} vs N=10 ref {ref:.5f} ({rel*100:+.1f}%)")
    bar3_pass = len(bar3_fail_details) == 0

    # ---- Bar 4: fscore/chamfer don't collapse ----
    fs_means = {s: stats.mean(float(by_seed[s][m]["fscore@0.01"]) for m in meshes) for s in SEEDS}
    ch_means = {s: stats.mean(float(by_seed[s][m]["chamfer_l2"]) for m in meshes) for s in SEEDS}
    task1_fs, task1_ch = 0.62511, 0.00038  # Task 2's Arm A (shrink control) N=10 mean, same recipe family
    bar4_fail_details = []
    for s in SEEDS:
        if fs_means[s] < task1_fs - 0.05:
            bar4_fail_details.append(f"seed{s} fscore@0.01={fs_means[s]:.4f} worse than ref {task1_fs:.4f} by >0.05")
        if ch_means[s] > task1_ch * 1.5:
            bar4_fail_details.append(f"seed{s} chamfer_l2={ch_means[s]:.5f} worse than ref {task1_ch:.5f} by >50%")
    bar4_pass = len(bar4_fail_details) == 0

    all_pass = bar1_pass and bar2_pass and bar3_pass and bar4_pass
    verdict = "PROMOTE D1 to shipped default" if all_pass else "HOLD (stay experimental-mainline candidate)"

    lines = []
    lines.append("# D1 N=50 x 3-seed validation scorecard\n")
    lines.append(f"## VERDICT: {verdict}\n")
    lines.append("- Recipe: `diagnostics/moonshot/cfg/D1_low_scalecap.yaml` (D1_low.yaml + w_scale: 0.052, "
                  "see PROMOTION_CRITERIA.md addendum for why this recipe was used instead of D1_low.yaml).")
    lines.append("- N=50 (full `bench50_clean`), seeds {0,1,2}, full budget (1000+1000 its), 1x H100 each "
                  "(Slurm array job 3034796, ~17 min/seed).")
    lines.append("- Promotion bar and design pre-registered in `PROMOTION_CRITERIA.md` before this job "
                  "was submitted.\n")

    lines.append("## Bar 1: no catastrophic per-specimen failures\n")
    lines.append(f"**{'PASS' if bar1_pass else 'FAIL'}** -- {len(catastrophic)} catastrophic events out of "
                  f"{len(SEEDS) * len(meshes)} (seed, specimen) runs.")
    for s, m, why in catastrophic[:30]:
        lines.append(f"- seed{s} {m}: {why}")

    lines.append("\n## Bar 2: cross-seed variance (CV = std/mean across 3 seeds, averaged over 50 specimens)\n")
    lines.append(f"**{'PASS' if bar2_pass else 'FAIL'}** (threshold: CV < 15%)")
    lines.append("| metric | mean CV across specimens |")
    lines.append("|---|---|")
    for k, v in cv_by_metric.items():
        lines.append(f"| {k} | {'n/a' if v is None else f'{100*v:.2f}%'} |")

    lines.append("\n## Bar 3: N=50 vs Task 1's N=10 D1 numbers (per seed)\n")
    lines.append(f"**{'PASS' if bar3_pass else 'FAIL'}** (threshold: <15% relative regression)")
    lines.append("| seed | edge_logratio_absmean | deform_mag_mean |")
    lines.append("|---|---|---|")
    for s in SEEDS:
        lines.append(f"| {s} | {n50_means[s]['edge_logratio_absmean']:.5f} | {n50_means[s]['deform_mag_mean']:.5f} |")
    lines.append(f"| N=10 reference (Task 1) | {TASK1_N10['edge_logratio_absmean']:.5f} | {TASK1_N10['deform_mag_mean']:.5f} |")
    for d in bar3_fail_details:
        lines.append(f"- FAIL detail: {d}")

    lines.append("\n## Bar 4: fscore/chamfer don't collapse\n")
    lines.append(f"**{'PASS' if bar4_pass else 'FAIL'}**")
    lines.append("| seed | fscore@0.01 mean | chamfer_l2 mean |")
    lines.append("|---|---|---|")
    for s in SEEDS:
        lines.append(f"| {s} | {fs_means[s]:.4f} | {ch_means[s]:.5f} |")
    for d in bar4_fail_details:
        lines.append(f"- FAIL detail: {d}")

    lines.append("\n## Full integrity metric means (mean across 50 specimens, per seed)\n")
    lines.append("| metric | seed0 | seed1 | seed2 |")
    lines.append("|---|---|---|---|")
    for k in INTEGRITY + SECONDARY:
        row = [f"{stats.mean(float(by_seed[s][m][k]) for m in meshes):.5f}" for s in SEEDS]
        lines.append(f"| {k} | " + " | ".join(row) + " |")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
