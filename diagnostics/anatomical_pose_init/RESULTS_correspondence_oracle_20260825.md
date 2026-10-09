# Correspondence-Oracle Results (Phase 10, Step 1) — 2026-08-25

Answers one question before any sampler, architecture, or network gets built (per
`PHASE10_DESIGN_correspondence_network_20260825.md`): **does correspondence-aware fitting help at
all, tested with perfect (ground-truth) correspondence, before anything estimates it?**

## Setup

Two oracle interventions, both on the same 12 `synth_clean` specimens, same zero-init pose, same
D1 recipe (`--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0`, `D1_low.yaml`) as the
`SYN_clean_zero_wsl` baseline — isolating correspondence-availability as the single manipulated
variable, same discipline as every other arm in this investigation:

- **`Oracle_GT_partition`**: perfect GROUP-level (leg) assignment. Reuses `PartFieldPartition`
  unchanged, fed ground-truth vertex labels instead of a learned network's prediction, via a new
  `--oracle_gt_partition_from` flag in `optimise_hierarchical.py`.
- **`Dense_GT_oracle`**: perfect PER-VERTEX assignment. A new additive loss term
  (`w_dense_gt`, default 0.0 — byte-identical to every existing arm when off, same discipline as
  `--init_joint_rot_from`) added to `trainer_hierarchical.py`'s `forward()`, gated behind a new
  `--dense_gt_correspondence_from` flag. Each resampled target point is matched directly to the
  fitted mesh's own vertex at its TRUE index (nearest vertex on the ground-truth, not fitted,
  target mesh), summed alongside — not replacing — the existing chamfer term.

Both were smoke-tested (2 specimens, 20 iterations) before the full 12-specimen run in each case.

## Results

| | leg_acc | seg_acc |
|---|---|---|
| `SYN_clean_zero_wsl` (baseline) | 0.867 | 0.831 |
| `Oracle_GT_partition` (group-level oracle) | 0.937 | 0.860 |
| `Dense_GT_oracle` (per-vertex oracle) | 0.936 | 0.874 |
| `GT_as_fitted_ceiling` (ground truth used as "fitted", i.e. zero geometric error) | 0.991 | **0.974** |

**Why `GT_as_fitted_ceiling` matters:** before treating 1.0 as the reference for "how much gap is
left," it's worth knowing the metric's own ceiling. Feeding the ground-truth mesh itself through
the identical audit pipeline (as if it were the fitted output) still only scores 0.974 seg_acc /
0.991 leg_acc — near-boundary point-sampling ambiguity (a point landing right at a co/tr boundary
can get a "true" label that disagrees with which vertex is nearest even under perfect
correspondence) caps the metric below 1.0 regardless of fit quality. So `Dense_GT_oracle`'s 0.874
is ~0.10 short of the metric's actual ceiling (0.974), not ~0.13 short of an unreachable 1.0.

### Pre-registered statistics (decided before the result, not after)

**leg_acc, `Dense_GT_oracle` vs zero:** mean delta +0.066. Paired t p=0.133, sign test p=0.387 (7/12
positive), Wilcoxon p=0.076. Excluding the two large movers (synth_003, synth_005): mean delta on
the remaining 10 is +0.0075, split 5/5 — genuinely flat, not "a small effect running through most
of the rest." leg_acc's gain is statistically indistinguishable from `Oracle_GT_partition`'s own
(0.936 vs 0.937) — expected, since leg_acc structurally cannot see within-part (segment-level)
correspondence at all.

**seg_acc, `Dense_GT_oracle` vs zero:** mean delta **+0.057, 12/12 specimens positive.** Paired t
p=0.011, sign test p=0.0002, Wilcoxon p=0.0002. The first result in this investigation that is
broad rather than outlier-driven — bigger than `Oracle_GT_partition`'s own seg_acc gain (+0.044
per-specimen mean, +0.029 aggregate), and significant on two tests that do not depend on any
single specimen.

### Falsifiable check A — are synth_003/synth_005 dense-specific, or generically reachable?

Compared each specimen's gain under `Oracle_GT_partition` against its gain under `Dense_GT_oracle`.
For synth_003 and synth_005 (the two large leg_acc movers): dense/group gain ratio ≈ **0.99 for
both** — essentially identical. So whatever makes these two specimens respond strongly to a
correspondence-shaped intervention is not specific to DENSE correspondence; group-level assignment
alone already captured almost all of it. The genuine within-part (seg_acc) story lives elsewhere:
synth_002 — never previously flagged — shows the second-largest seg_acc gain (+0.126), while
synth_003/005's own seg_acc gains (+0.034, +0.236) are not the top of the distribution either.
**Conclusion: the "these two specimens are special" pattern and the "dense correspondence helps
broadly" pattern are two separate phenomena, not the same one.**

### Falsifiable check B — is the effect smaller at l2 (mesothoracic), consistent with the
documented starvation defect, or not?

Per-(specimen, leg) seg_acc, collapsed by leg position (l1/l2/l3, left+right pooled):

| position | n | mean_zero | mean_delta |
|---|---|---|---|
| l1 (front) | 24 | 0.810 | +0.060 |
| l2 (mid) | 24 | 0.794 | +0.062 |
| l3 (hind) | 24 | 0.814 | +0.068 |

**Inconclusive, and reported as such rather than rounded toward either story.** l2's delta is
essentially identical to l1/l3's (+0.062 vs +0.064 combined) — not meaningfully smaller (which
would have supported the mesothoracic-starvation/geometric-limit explanation already documented in
`geom_leg_init.py`), and not meaningfully larger either (which would have undercut it). This test
does not discriminate between the two hypotheses; it should not be cited as evidence for either.

## What this establishes, and what it does NOT yet establish

**Established:** correspondence is a real, validated lever for the within-part (segment-level)
error this investigation's own `FINAL_REPORT.md`/`why_partitions_null.py` identified as 83.3% of
total correspondence error — the slice no partition-shaped intervention, however fine, can reach.
The `Dense_GT_oracle` result is the first evidence in this entire investigation that this specific
slice responds to a correspondence signal, broadly and significantly, not just in one or two
specimens.

**RESOLVED (weight-dominance check, `Dense_GT_oracle_dominant`, `w_dense_gt_correspondence=50.0`,
~50x the original test):** if the residual gap were a weight-balance artifact, cranking the
correspondence term to dominate the objective should have closed it. It did not.

| | seg_acc | edge (smoke-test diagnostic) |
|---|---|---|
| `Dense_GT_oracle` (w=1.0) | 0.874 | ~0.05-0.07 |
| `Dense_GT_oracle_dominant` (w=50.0) | 0.870 | 0.36-0.80 (confirms dominance actually took hold) |

seg_acc did not improve under the dominant weight — if anything it was marginally lower (0.870 vs
0.874, paired t p=0.245, only 5/12 specimens even improved), while the smoke test confirms the
weight change was not inert: edge distortion rose sharply (0.05→0.36-0.80), meaning the fitter
really was sacrificing mesh smoothness to chase the correspondence signal, and it still could not
close the gap. **Explanation 1 (weight-balance artifact) is falsified.** The residual ~0.10 gap to
the metric's ceiling (0.974) is not a matter of the correspondence signal being under-prioritized —
it persists even when that signal completely dominates every other term in the objective.

**Explanation 2 (a genuine fitter-capacity/geometric limit, independent of loss weighting) is the
supported conclusion.** Some other constraint — shape-space capacity, mesh topology/resolution, or
the model's own deformation limits — caps how well the final fitted mesh can realize even a
maximally-weighted, perfectly-correct dense correspondence signal. This is a second, independent
gap from the already-named oracle-vs-learned-correspondence risk (Phase 10 design doc §5), and it
has its own section there now (§5b) rather than being folded into the sim-to-real risk, since the
mechanism is different: this one is about fitter capacity, not train/test distribution shift.

## Artifacts

- `fitter_3d/optimise_hierarchical.py`: `--oracle_gt_partition_from`, `--dense_gt_correspondence_from`,
  `--w_dense_gt_correspondence` (new, off by default).
- `fitter_3d/trainer_hierarchical.py`: `dense_gt_verts`/`w_dense_gt` (new, additive, gated).
- `diagnostics/anatomical_pose_init/run_oracle_gt_partition_20260825.sh`,
  `run_dense_oracle_20260825.sh` — recipes for the two oracle runs above.
- `diagnostics/anatomical_pose_init/analyze_within_part_by_leg_20260825.py` — falsifiable check B.
- `diagnostics/correspondence_accuracy/run_audit.py`: `per_specimen_seg_acc`,
  `per_specimen_per_leg_acc` (new, per-specimen granularity), `GT_as_fitted_ceiling` row (metric
  ceiling check), `gt_as_fitted.npz` shim.
- `Dense_GT_oracle_dominant` run (`w_dense_gt_correspondence=50.0`) — weight-dominance check,
  resolved (see above): falsifies the weight-balance-artifact explanation.
- `diagnostics/correspondence_accuracy/out/gt_as_fitted.npz` — shim npz (ground truth used as
  "fitted") for the `GT_as_fitted_ceiling` metric-ceiling check.
- Phase 10 design doc §5b now carries this finding as a second, independent risk alongside the
  sim-to-real gap.
