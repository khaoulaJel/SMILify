#!/usr/bin/env python3
"""Score `Network_correspondence` (C2) against the B3 pre-registered success bar
(out_B2_correspondence_net_20260825/PREREGISTRATION_B3_success_bar_20260825.md), written BEFORE
this network's leg_acc/seg_acc were seen. Reads `run_audit.py`'s
`per_specimen_leg_acc`/`per_specimen_seg_acc` (already computed per-specimen, no new metric code)
for `Network_correspondence` and the `SYN_clean_zero_wsl` baseline, joined by specimen name.

Reports, per the project's standing discipline (never violated in this investigation without
being caught and corrected -- see finding #9): paired-t, sign test, AND Wilcoxon signed-rank on
the full 12 specimens, plus the outlier-excluded delta (drop the 2 largest movers) alongside the
full-sample delta -- regardless of which result looks better, and regardless of whether the
outcome is positive, negative, or inconclusive.
"""
import json
import os
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CONF_PATH = os.path.join(REPO, "diagnostics", "correspondence_accuracy", "out", "correspondence_confusion.json")

BASELINE = "SYN_clean_zero_wsl"
CANDIDATE = "Network_correspondence"
ORACLE_LEG = "Oracle_GT_partition"
ORACLE_SEG = "Dense_GT_oracle"

# PREREGISTRATION_B3_success_bar_20260825.md hardcoded these from a JSON snapshot taken when it
# was written. CAUGHT 2026-08-26 (before trusting any C2 verdict): re-checking against the
# CURRENTLY committed correspondence_confusion.json shows leg_acc is stable (0.8678/0.9397, exact
# match) but seg_acc's baseline drifted 0.8201 -> 0.8322 between then and now (oracle ~stable,
# 0.8745->0.8741) -- shrinking the true gap from 0.0544 to 0.0419, enough that the frozen "25%
# floor" (0.8337) sits only 0.0015 above the CURRENT live baseline. Root cause not tracked down
# (SYN_clean_zero_wsl's fit or the seg_acc metric code likely changed sometime in the WSL
# continuation window) -- rather than trust either frozen number, the gap is now computed FRESH
# from the same JSON this script itself reads results from, every time it runs, so a future silent
# drift can't corrupt the verdict again the same way.
def load_metric_value(results, run_name, metric_key):
    for r in results:
        if r["run"] == run_name:
            return r[metric_key]["accuracy"]
    raise KeyError(f"{run_name} not found in {CONF_PATH} -- did the fit+audit finish?")


def load_per_specimen(results, run_name, key):
    for r in results:
        if r["run"] == run_name:
            return dict(r[key])
    raise KeyError(f"{run_name} not found in {CONF_PATH} -- did the fit+audit finish?")


def paired_tests(base_vals, cand_vals, names):
    base = [base_vals[n] for n in names]
    cand = [cand_vals[n] for n in names]
    deltas = [c - b for b, c in zip(base, cand)]

    t_stat, t_p = stats.ttest_rel(cand, base)
    try:
        sign_res = stats.binomtest(sum(d > 0 for d in deltas), len(deltas), 0.5)
        sign_p = sign_res.pvalue
    except AttributeError:  # older scipy
        sign_p = stats.binom_test(sum(d > 0 for d in deltas), len(deltas), 0.5)
    try:
        w_stat, w_p = stats.wilcoxon(cand, base)
    except ValueError:
        w_p = float("nan")

    order = sorted(range(len(deltas)), key=lambda i: -abs(deltas[i]))
    excl = set(order[:2])
    deltas_excl = [d for i, d in enumerate(deltas) if i not in excl]
    excl_names = [names[i] for i in order[:2]]

    return dict(
        n=len(deltas), mean_delta=sum(deltas) / len(deltas),
        wins=sum(d > 0 for d in deltas), losses=sum(d < 0 for d in deltas),
        t_p=t_p, sign_p=sign_p, wilcoxon_p=w_p,
        mean_delta_excl_outliers=sum(deltas_excl) / len(deltas_excl),
        outliers_excluded=excl_names,
    )


def report_metric(name, base_vals, cand_vals, base_ref, gap):
    names = sorted(set(base_vals) & set(cand_vals))
    assert len(names) == 12, f"{name}: expected 12 synth_clean specimens, got {len(names)}: {names}"
    res = paired_tests(base_vals, cand_vals, names)
    cand_mean = sum(cand_vals[n] for n in names) / len(names)
    frac_gap = (cand_mean - base_ref) / gap

    print(f"\n=== {name} ===")
    print(f"baseline mean={base_ref:.4f}  candidate mean={cand_mean:.4f}  "
          f"delta(full)={res['mean_delta']:+.4f}  delta(outlier-excl)={res['mean_delta_excl_outliers']:+.4f}")
    print(f"wins/losses={res['wins']}/{res['losses']} of {res['n']}  "
          f"paired-t p={res['t_p']:.4f}  sign p={res['sign_p']:.4f}  wilcoxon p={res['wilcoxon_p']:.4f}")
    print(f"fraction of oracle gap recovered: {frac_gap:.1%} (gap={gap:.4f})")
    print(f"outliers excluded from the outlier-excl delta: {res['outliers_excluded']}")

    significant = res["sign_p"] < 0.05 and res["wilcoxon_p"] < 0.05
    real_floor = frac_gap >= 0.25 and significant
    strong_tier = frac_gap >= 0.50 and significant
    verdict = "STRONG EFFECT" if strong_tier else ("REAL-EFFECT FLOOR MET" if real_floor else "DOES NOT CLEAR THE BAR")
    print(f"B3 verdict: {verdict}"
          + ("" if significant else "  (sign+Wilcoxon significance requirement NOT met, regardless of mean)"))
    return dict(name=name, frac_gap=frac_gap, significant=significant, verdict=verdict, **res)


def main():
    with open(CONF_PATH) as f:
        results = json.load(f)

    base_leg_acc = load_metric_value(results, BASELINE, "leg_confusion")
    oracle_leg_acc = load_metric_value(results, ORACLE_LEG, "leg_confusion")
    base_seg_acc = load_metric_value(results, BASELINE, "within_leg_segment_confusion")
    oracle_seg_acc = load_metric_value(results, ORACLE_SEG, "within_leg_segment_confusion")
    leg_gap = oracle_leg_acc - base_leg_acc
    seg_gap = oracle_seg_acc - base_seg_acc

    print(f"[live gap check] leg_acc: baseline={base_leg_acc:.4f} oracle={oracle_leg_acc:.4f} gap={leg_gap:.4f} "
          f"(frozen doc: 0.8678/0.9397/0.0719)")
    print(f"[live gap check] seg_acc: baseline={base_seg_acc:.4f} oracle={oracle_seg_acc:.4f} gap={seg_gap:.4f} "
          f"(frozen doc: 0.8201/0.8745/0.0544)")
    if abs(base_seg_acc - 0.8201) > 0.002 or abs(base_leg_acc - 0.8678) > 0.002:
        print("[WARNING] live baseline differs from the frozen PREREGISTRATION doc by >0.002 -- "
              "using the LIVE numbers above (same JSON this script reads results from), not the frozen ones.")

    base_leg = load_per_specimen(results, BASELINE, "per_specimen_leg_acc")
    cand_leg = load_per_specimen(results, CANDIDATE, "per_specimen_leg_acc")
    base_seg = load_per_specimen(results, BASELINE, "per_specimen_seg_acc")
    cand_seg = load_per_specimen(results, CANDIDATE, "per_specimen_seg_acc")

    leg_result = report_metric("leg_acc", base_leg, cand_leg, base_leg_acc, leg_gap)
    seg_result = report_metric("seg_acc", base_seg, cand_seg, base_seg_acc, seg_gap)

    print("\n=== Overall B3 outcome (per pre-registration: BOTH metrics must clear the same tier) ===")
    tiers = {"DOES NOT CLEAR THE BAR": 0, "REAL-EFFECT FLOOR MET": 1, "STRONG EFFECT": 2}
    overall_tier = min(tiers[leg_result["verdict"]], tiers[seg_result["verdict"]])
    overall = [k for k, v in tiers.items() if v == overall_tier][0]
    print(f"leg_acc: {leg_result['verdict']}   seg_acc: {seg_result['verdict']}   -> overall: {overall}")


if __name__ == "__main__":
    main()
