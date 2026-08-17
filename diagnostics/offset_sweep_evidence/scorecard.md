# Offset-penalty strength sweep scorecard (Task 3)

## VERDICT: CONFIRMED: 5.0/2.0 is already near-optimal

No setting beats the 5.0/2.0 control's correct_frac by a margin outside the ~3 percentage-point noise floor, on either corpus. This is a closed, confirmed non-finding, not a failed experiment -- per the pre-registered success bar, the deliverable is the confirmation itself.

- Corpus: `diagnostics/moonshot/synth_clean` (12 specimens) and `synth_noisy` (12 specimens), materialized read-only from `feature/registration_moonshot` (git show, branch not checked out) -- not regenerated, seed/pose_scale/shape_scale/scale_scale/noise unchanged from the corpus already on record.
- Recipe: full D1 (hierarchical placement -> moonshot refinement), full budget (1000+1000 its), same iteration counts as committed `D1_low.yaml`. Each (corpus, setting) pair ran its own hierarchical stage (8 total, one per array task) rather than sharing one per corpus, to parallelize across the array job -- w_offset is a moonshot Stage_2/3 loss weight, not read by the hierarchical entrypoint, so any difference between per-setting hierarchical runs on the same corpus is optimizer noise only, not a w_offset effect.
- Hardware: 1x H100 each, Slurm array job 3027161 (8 tasks, ~7-9 min per task), scoring job 3027162.

**Provenance note on the E6 baseline numbers (6.69%/4.81%) in FINAL_REPORT.md §2.2**: those were measured on `feature/registration_moonshot`'s own pipeline state at the time of that report. This sweep runs the D1 chain as it exists on `to-ship` today (commit 9ebe508 + Task 2's evidence commit), which may differ in minor ways (e.g. joint-limit weights, symmetric sampling landed via the porting process). The **within-this-sweep control** (w5 = 5.0/2.0) is therefore the correct comparison baseline for this experiment, not the report's historical 6.69% figure -- both are reported below for context: this sweep's w5/clean control measured 6.69% vs the report's historical 6.69%.

## Per-setting table

| corpus | w_offset (coarse/fine) | correct_frac | delta vs control | median err % | p90 err % |
|---|---|---|---|---|---|
| clean | 2.5/1.0 | 6.91% | +0.22pp | 3.10% | 20.04% |
| clean | 5.0/2.0 (control) | 6.69% | -- | 3.21% | 20.06% |
| clean | 10.0/4.0 | 6.26% | -0.43pp | 3.43% | 20.39% |
| clean | 20.0/8.0 | 6.31% | -0.38pp | 3.65% | 20.73% |
| noisy | 2.5/1.0 | 6.51% | +0.08pp | 3.21% | 22.76% |
| noisy | 5.0/2.0 (control) | 6.43% | -- | 3.30% | 22.87% |
| noisy | 10.0/4.0 | 6.04% | -0.39pp | 3.60% | 22.76% |
| noisy | 20.0/8.0 | 5.15% | -1.28pp | 3.86% | 23.40% |

## Per-part breakdown (correct_frac %, all 7 anatomical parts)

| corpus | setting | antenna | gaster | head | leg distal | leg prox | mandible | thorax |
|---|---|---|---|---|---|---|---|---|
| clean | w2_5 | 5.2% | 1.8% | 6.4% | 3.2% | 8.4% | 6.6% | 10.9% |
| clean | w5 | 6.1% | 1.9% | 6.4% | 3.3% | 7.7% | 6.1% | 10.9% |
| clean | w10 | 6.3% | 1.9% | 5.8% | 2.9% | 7.2% | 6.3% | 10.3% |
| clean | w20 | 6.0% | 1.8% | 4.7% | 2.9% | 6.9% | 6.1% | 13.4% |
| noisy | w2_5 | 4.1% | 1.8% | 6.0% | 3.0% | 8.0% | 5.3% | 10.3% |
| noisy | w5 | 5.1% | 1.7% | 5.7% | 3.1% | 7.6% | 5.6% | 10.9% |
| noisy | w10 | 4.3% | 1.8% | 5.1% | 2.7% | 7.2% | 5.4% | 10.8% |
| noisy | w20 | 4.9% | 1.3% | 3.2% | 2.8% | 6.4% | 4.9% | 9.8% |

## Monotonicity check (secondary observation, not a finding per the noise-floor rule)

- clean: w2_5=6.91% -> w5=6.69% -> w10=6.26% -> w20=6.31%
- noisy: w2_5=6.51% -> w5=6.43% -> w10=6.04% -> w20=5.15%

correct_frac decreases roughly monotonically as w_offset increases past the control on both corpora (most pronounced on synth_noisy: 6.51% -> 6.43% -> 6.04% -> 5.15%), but the single largest gap (w20 vs w5 control on noisy) is only 1.28pp, still under the 3pp noise floor. The direction is consistent enough to note for future work (larger offset penalties trend worse, not better, beyond 5.0/2.0) but not large enough, at n=12 single-seed, to justify moving the default down to 2.5/1.0 either -- w2_5 vs w5 control is +0.22pp (clean) / +0.08pp (noisy), both within noise.

## Pre-registered success bar

- A candidate must beat the 5.0/2.0 control's correct_frac by >3pp (the report's noise floor) to count as a finding.
- Ties within noise: prefer the setting closest to the current default.
- If nothing beats control outside noise: report '5.0/2.0 is already near-optimal' as a closed, confirmed non-finding.

**Result: no setting beat control outside the 3pp noise floor on either corpus. Recommendation: keep w_offset at 5.0/2.0.**
