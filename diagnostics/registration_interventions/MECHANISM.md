# `--split_distal` mechanism (Intervention A, Task 7)

Read before interpreting any result below. Code refs: `fitter_3d/trainer_hierarchical.py`,
wired in via `fitter_3d/optimise_hierarchical.py` (flag defined L61-68, threaded through at
L282/L407).

## What it changes

`--split_distal` is a **grouping-granularity flag** for the hierarchical fitter's
target-partitioned chamfer. It does not touch sampling density, loss weights, optimizer,
learning rates, joint limits, or pose initialization.

**Default (flag OFF)**: `anatomical_groups()` (`trainer_hierarchical.py:74-138`) maps each leg's
joints to ONE group per leg (`l{k}_{side}`, k=0..2, side=l/r -> 6 leg groups + `body` = 7 groups
total). Every leg's target-point partition and vertex partition mixes proximal (coxa/trochanter/
femur, 94% of chain area) and distal (tibia/tarsus/pretarsus, 2.4% of chain area) into a single
chamfer term per leg.

**With the flag ON**: each leg is split into TWO groups (`l{k}p_{side}` proximal,
`l{k}d_{side}` distal via `DISTAL_SEGMENTS = ("ti","ta","pt")`, L71/125-127) -> 12 leg groups +
`body` = 13 groups. Everything downstream (`vertex_groups`, `joint_mask_for_groups`,
`TargetPartition`, `HierarchicalStage._partitioned_chamfer`) is generic over the group list, so
no other code path branches on the flag.

## Mechanism, precisely

1. **Target sampling**: UNCHANGED. `n_sample=8000` points are drawn from the target mesh exactly
   as before; `--split_distal` does not resample or reweight target density.
2. **Target assignment**: CHANGED. `TargetPartition.update()` (L262-275) assigns every target
   point to its nearest FITTED-vertex group via 1-NN (`knn_points`, hard argmin). With the flag
   on, a target point that would previously land in `l0_r` (all of leg 0's territory) now lands
   in either `l0p_r` or `l0d_r`, whichever fitted vertex it is nearest to. This is a
   re-partitioning of the SAME sampled points, not new sampling.
3. **Loss weighting / partitioning**: CHANGED. `_partitioned_chamfer()` (L417-483) sums one
   bidirectional-chamfer term PER GROUP, each normalized independently over its own assigned
   points (implicit in per-group KNN calls, `d_st`/`d_ts` computed per group then summed/averaged
   across groups). So a group with 2.4% of the leg's area but its own group ID now contributes a
   term of comparable MAGNITUDE to the 94%-of-area proximal group, rather than being diluted
   inside one leg-wide term where distal points are typically <10 out of ~8000/13≈615 per-leg
   samples (median ~2 for pretarsus alone per Task 6 D2a). This is the entire causal mechanism:
   distal gets an independent, equally-weighted vote instead of a proportionally tiny one.
4. **Optimization variables**: UNCHANGED. `joint_rot`, `betas`, `global_rot`, `trans`,
   `deform_verts` are the same parameters; `--split_distal` only changes `joint_mask_for_groups`
   (which joints are unfrozen per stage, still the same per-leg joints, now addressable at
   proximal/distal granularity if a caller restricts `active_groups` -- the existing H0-H3
   schedule in `optimise_hierarchical.py` does not use per-half `active_groups`, so this
   sub-effect is inactive in the runs here) and the vertex group used for the chamfer restriction.
5. **Resampling behavior**: UNCHANGED. `reassign_every=50` (default) still governs how often
   `TargetPartition.update()` re-runs; the recompute cadence is identical, only what a group ID
   MEANS for distal points changes.
6. **n_t<10 skip threshold** (`_partitioned_chamfer` L437-447): a group with fewer than 10
   assigned target points is skipped that iteration rather than forced to match nearest-anything.
   Under the default (unsplit) grouping this threshold is checked against pretarsus+tarsus+tibia+
   proximal combined (rarely tripped). Under split_distal it is checked against
   tibia+tarsus+pretarsus alone (median ~2-13 samples per Task 6 D2a) -- so split_distal can
   plausibly trip this skip MORE often for the distal half specifically, an empirical question
   this experiment must measure directly (Metric D in the mission spec), not assume away.

## Default-OFF invariance

Confirmed by reading, not run: `anatomical_groups(..., split_distal=False)` is byte-identical to
the pre-existing (pre-Task-7) code path -- the `if split_distal:` branch (L125-129) is new-in-name
only insofar as the flag exists; the `else` branch reproduces the exact prior single-group-per-leg
logic. No caller in the current codebase passes `split_distal=True` by default anywhere;
`optimise_hierarchical.py --split_distal` is `action="store_true"`, default off. All baseline
comparator runs (`SYN_clean_pose0_w5`, `SYN_clean_w5`, `SYN_clean_drop30_w5`,
`SYN_clean_drop60_w5`, all already fit under Task 3/6) used the flag OFF and are reused unchanged
as the baseline arm for this experiment -- not re-run.

## What is held fixed between baseline and Intervention A

Per the mission spec, everything except `--split_distal` on the hierarchical stage:
`--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0`, full default budget
(`body_its=900, leg_its=1200, joint_its=900, deform_its=600`), `n_sample=8000`,
`reassign_every=50`, same `D1_low.yaml` moonshot handoff, same seed (0, default), same corpora.
See `run_A_splitdistal_fit.sbatch` (this directory) vs.
`diagnostics/appendage_evidence/run_pose0_fit.sbatch` / `run_damage_fit.sbatch` and
`diagnostics/registration_failure/run_pose_sweep_fit.sbatch` for the exact baseline commands
being reused as comparators.
