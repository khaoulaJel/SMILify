# B3 — Pre-registered success bar for the correspondence network (2026-08-26)

Written before B2's trained network is converted into a candidate (via C2's per-leg rigid-
alignment reuse or PHASE10 §4's fitter hook) and run through `run_audit.py` — i.e. before seeing
any leg_acc/seg_acc number for the network, per this investigation's standing discipline (B3's
own spec, and the precedent that caught #9's over-read: decide what a result would mean before
you have it, not after).

## The gaps being targeted (measured freshly here, not assumed from memory)

Pulled directly from `diagnostics/correspondence_accuracy/out/correspondence_confusion.json`
(same-run, same-code-version numbers, not mixed across runs):

| metric | baseline (`SYN_clean_zero_wsl`) | oracle ceiling | gap |
|---|---:|---:|---:|
| leg_acc | 0.8678 | 0.9397 (`Oracle_GT_partition`) | **0.0719** |
| seg_acc | 0.8201 | 0.8745 (`Dense_GT_oracle`) | **0.0544** |

(`GT_as_fitted_ceiling` — ground truth run through the same metric — scores 0.9913/0.9759, not
1.0, per finding #11's own reference-ceiling discipline; not used as the target here since the
oracle rows above are the actual mechanisms the network is meant to approximate, not the metric's
theoretical maximum.)

## Why a fraction-of-gap bar, and why these thresholds

Every prior oracle-vs-learned/estimated comparison in this investigation showed a large gap
between giving the optimizer the perfect answer and estimating it (IK's ~21° oracle vs ~33°+
real-scan estimators, finding #4; G1d's learned initializer tying, not beating, zero-init despite
G1's oracle-style ceiling tests looking promising). A bar demanding near-full gap recovery would
almost certainly fail regardless of whether B1's architecture has real signal; a bar with no floor
would count noise as success. Two tiers, calibrated against that history:

- **Real-effect floor: ≥25% of gap recovered on BOTH leg_acc and seg_acc**, AND statistically
  broad — **sign test p<0.05 AND Wilcoxon signed-rank p<0.05** across all 12 `synth_clean`
  specimens (not the mean, not a paired t-test alone — the same requirement finding #11's dense
  oracle had to clear, and the one #9's group-oracle read failed before being retracted).
  Concretely: leg_acc ≥ 0.8678 + 0.25×0.0719 = **0.8858**; seg_acc ≥ 0.8201 + 0.25×0.0544 =
  **0.8337**, both significant by both tests.
- **Strong-effect tier: ≥50% of gap recovered on both**, same significance requirement.
  leg_acc ≥ **0.9038**; seg_acc ≥ **0.8473**.
- Anything positive but **not significant by sign+Wilcoxon on the full 12 specimens** does not
  count as a real result regardless of the aggregate mean, full stop — this is the exact failure
  mode #9 and A1's `coherent_30deg` check (this session) both already caught.
- Both the full-12-specimen delta AND the outlier-excluded delta (drop the 2 largest movers) will
  be reported side by side regardless of outcome, per standing discipline — a result driven by
  1-2 specimens is a different, weaker claim than a broad one.

## What counts as failure, stated plainly in advance

- Recovers <25% of either gap, OR fails sign+Wilcoxon significance on either metric: **negative
  result**, reported with the same rigor as a positive one (per #7/#8/#9-check-B precedent) —
  not explained away, not quietly re-run with different hyperparameters until it clears the bar.
- A3's linear-probe numbers (0.85-0.87 leg-identity accuracy, R² 0.71-0.84 chain-position) do
  NOT set this bar's expectation upward — A3 explicitly measured a lower-bound/existence
  argument on a network never trained for this task (`RESULTS_A3_architecture_probe_20260825.md`
  caveat), not a preview of trained performance. If B2's trained network clears the linear-probe
  numbers on held-out data but the resulting leg_acc/seg_acc still misses the 25% floor above,
  that is itself informative (the remaining gap is downstream of the correspondence signal itself
  — e.g. in how it's converted to a pose/loss term) and should be reported as such, not treated as
  a contradiction to paper over.

## Sequencing note

This bar is being written after B1 (architecture, done) and alongside B2 (training, in progress)
but strictly before B2's trained network is evaluated end-to-end. Training-loss curves
(`training_history.json`) are diagnostic only during training — they are not what this bar
scores, and a good training loss does not pre-empt this bar's evaluation once the network is
actually converted into a candidate and run through `run_audit.py`.
