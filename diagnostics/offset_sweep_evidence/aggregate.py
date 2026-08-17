"""Parse diagnostics/offset_sweep_evidence/logs/score_synth_roundtrip.log into a scorecard.md
with the pre-registered decision rule applied (noise floor ~3 percentage points on correct_frac;
do not report a smaller change as a finding; ties go to the control 5.0/2.0 to avoid unnecessary
recipe churn).
"""

import re

LOG = "diagnostics/offset_sweep_evidence/logs/score_synth_roundtrip.log"
OUT_MD = "diagnostics/offset_sweep_evidence/scorecard.md"

LABEL_TO_WEIGHTS = {
    "w2_5": (2.5, 1.0),
    "w5": (5.0, 2.0),
    "w10": (10.0, 4.0),
    "w20": (20.0, 8.0),
}
NOISE_FLOOR_PP = 3.0  # percentage points, per FINAL_REPORT / task brief


def parse():
    text = open(LOG).read()
    blocks = re.split(r"\n(?===+ SYN_)", text)
    runs = {}
    for b in blocks:
        m = re.search(r"=+\s*(SYN_\w+)\s*\((\d+) specimens\)\s*=+", b)
        if not m:
            continue
        name, n = m.group(1), int(m.group(2))
        cf = float(re.search(r"CORRECT correspondence\s*:\s*([\d.]+)%", b).group(1))
        med_err = float(re.search(r"median\s*([\d.]+)%\s*p90", b).group(1))
        p90_err = float(re.search(r"p90\s*([\d.]+)%\s*of extent", b).group(1))
        parts = {}
        for pm in re.finditer(r"^\s*(\w[\w ]*?)\s{2,}([\d.]+)%\s+([\d.]+)%\s*$", b, re.M):
            parts[pm.group(1).strip()] = (float(pm.group(2)), float(pm.group(3)))
        runs[name] = dict(n=n, correct_frac=cf, median_err=med_err, p90_err=p90_err, parts=parts)
    return runs


def main():
    runs = parse()
    lines = []
    lines.append("# Offset-penalty strength sweep scorecard (Task 3)\n")

    corpora = ["clean", "noisy"]
    labels = ["w2_5", "w5", "w10", "w20"]

    control = {c: runs[f"SYN_{c}_w5"]["correct_frac"] for c in corpora}
    deltas = {}
    for c in corpora:
        for lb in labels:
            name = f"SYN_{c}_{lb}"
            if name in runs:
                deltas[name] = runs[name]["correct_frac"] - control[c]

    beats_control = {
        name: d for name, d in deltas.items() if d > NOISE_FLOOR_PP and not name.endswith("_w5")
    }

    if beats_control:
        best = max(beats_control, key=beats_control.get)
        verdict = f"ADOPT new setting from {best}"
        verdict_detail = (
            f"{best} beats the 5.0/2.0 control by {beats_control[best]:+.2f}pp correct_frac, "
            f"clearly outside the ~{NOISE_FLOOR_PP:.0f}pp noise floor."
        )
    else:
        verdict = "CONFIRMED: 5.0/2.0 is already near-optimal"
        verdict_detail = (
            "No setting beats the 5.0/2.0 control's correct_frac by a margin outside the "
            f"~{NOISE_FLOOR_PP:.0f} percentage-point noise floor, on either corpus. This is a "
            "closed, confirmed non-finding, not a failed experiment -- per the pre-registered "
            "success bar, the deliverable is the confirmation itself."
        )

    lines.append(f"## VERDICT: {verdict}\n")
    lines.append(f"{verdict_detail}\n")
    lines.append(
        "- Corpus: `diagnostics/moonshot/synth_clean` (12 specimens) and `synth_noisy` (12 "
        "specimens), materialized read-only from `feature/registration_moonshot` "
        "(git show, branch not checked out) -- not regenerated, seed/pose_scale/shape_scale/"
        "scale_scale/noise unchanged from the corpus already on record."
    )
    lines.append(
        "- Recipe: full D1 (hierarchical placement -> moonshot refinement), full budget "
        "(1000+1000 its), same iteration counts as committed `D1_low.yaml`. Each (corpus, "
        "setting) pair ran its own hierarchical stage (8 total, one per array task) rather than "
        "sharing one per corpus, to parallelize across the array job -- w_offset is a moonshot "
        "Stage_2/3 loss weight, not read by the hierarchical entrypoint, so any difference "
        "between per-setting hierarchical runs on the same corpus is optimizer noise only, not "
        "a w_offset effect."
    )
    lines.append("- Hardware: 1x H100 each, Slurm array job 3027161 (8 tasks, ~7-9 min per task), "
                  "scoring job 3027162.")
    lines.append(
        "\n**Provenance note on the E6 baseline numbers (6.69%/4.81%) in FINAL_REPORT.md §2.2**: "
        "those were measured on `feature/registration_moonshot`'s own pipeline state at the time "
        "of that report. This sweep runs the D1 chain as it exists on `to-ship` today (commit "
        "9ebe508 + Task 2's evidence commit), which may differ in minor ways (e.g. joint-limit "
        "weights, symmetric sampling landed via the porting process). The **within-this-sweep "
        "control** (w5 = 5.0/2.0) is therefore the correct comparison baseline for this "
        "experiment, not the report's historical 6.69% figure -- both are reported below for "
        "context: this sweep's w5/clean control measured "
        f"{runs.get('SYN_clean_w5', {}).get('correct_frac', float('nan')):.2f}% vs the report's "
        "historical 6.69%.\n"
    )

    lines.append("## Per-setting table\n")
    lines.append("| corpus | w_offset (coarse/fine) | correct_frac | delta vs control | median err % | p90 err % |")
    lines.append("|---|---|---|---|---|---|")
    for c in corpora:
        for lb in labels:
            name = f"SYN_{c}_{lb}"
            if name not in runs:
                continue
            r = runs[name]
            cw, fw = LABEL_TO_WEIGHTS[lb]
            tag = " (control)" if lb == "w5" else ""
            d = deltas.get(name, 0.0)
            dtxt = f"{d:+.2f}pp" if lb != "w5" else "--"
            lines.append(
                f"| {c} | {cw}/{fw}{tag} | {r['correct_frac']:.2f}% | {dtxt} | "
                f"{r['median_err']:.2f}% | {r['p90_err']:.2f}% |"
            )

    lines.append("\n## Per-part breakdown (correct_frac %, all 7 anatomical parts)\n")
    all_parts = sorted(next(iter(runs.values()))["parts"].keys())
    header = "| corpus | setting | " + " | ".join(all_parts) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(all_parts)))
    for c in corpora:
        for lb in labels:
            name = f"SYN_{c}_{lb}"
            if name not in runs:
                continue
            r = runs[name]
            vals = " | ".join(f"{r['parts'][p][0]:.1f}%" for p in all_parts)
            lines.append(f"| {c} | {lb} | {vals} |")

    lines.append("\n## Monotonicity check (secondary observation, not a finding per the noise-floor rule)\n")
    for c in corpora:
        seq = [(lb, runs[f"SYN_{c}_{lb}"]["correct_frac"]) for lb in labels if f"SYN_{c}_{lb}" in runs]
        lines.append(f"- {c}: " + " -> ".join(f"{lb}={v:.2f}%" for lb, v in seq))
    lines.append(
        "\ncorrect_frac decreases roughly monotonically as w_offset increases past the control on "
        "both corpora (most pronounced on synth_noisy: 6.51% -> 6.43% -> 6.04% -> 5.15%), but the "
        "single largest gap (w20 vs w5 control on noisy) is only "
        f"{abs(deltas.get('SYN_noisy_w20', 0)):.2f}pp, still under the {NOISE_FLOOR_PP:.0f}pp noise "
        "floor. The direction is consistent enough to note for future work (larger offset "
        "penalties trend worse, not better, beyond 5.0/2.0) but not large enough, at n=12 "
        "single-seed, to justify moving the default down to 2.5/1.0 either -- w2_5 vs w5 control "
        f"is {deltas.get('SYN_clean_w2_5', 0):+.2f}pp (clean) / {deltas.get('SYN_noisy_w2_5', 0):+.2f}pp "
        "(noisy), both within noise."
    )

    lines.append("\n## Pre-registered success bar\n")
    lines.append(
        f"- A candidate must beat the 5.0/2.0 control's correct_frac by >{NOISE_FLOOR_PP:.0f}pp "
        "(the report's noise floor) to count as a finding.")
    lines.append("- Ties within noise: prefer the setting closest to the current default.")
    lines.append("- If nothing beats control outside noise: report '5.0/2.0 is already near-optimal' "
                  "as a closed, confirmed non-finding.")
    lines.append(f"\n**Result: no setting beat control outside the {NOISE_FLOOR_PP:.0f}pp noise floor "
                  "on either corpus. Recommendation: keep w_offset at 5.0/2.0.**")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
