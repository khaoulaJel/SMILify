# Task 3, Phase 1 — why leg_distal fails (R=0.136), before touching any measurement definition

## Question

Does the `leg_distal` failure (ground-truth R=0.136 on `seg_l_*_ti`/`seg_l_*_ta`, the tibia and
tarsus bone lengths) come from bad JOINT positions (upstream correspondence problem — no
measurement definition can fix it), from something specific to the joint-to-joint LENGTH
definition (a definition problem — a different definition could plausibly help), or from POSE
sensitivity (bending)?

## 1a — joint localization error, distal vs proximal leg

Reused the exact ceiling-test pair behind the R=0.136 number (`SYN_clean_w5` fit vs
`synth_clean/ground_truth.npz`, same per-specimen normalisation `calib_features.py` uses).
Computed `||J_fitted - J_true||` per joint via `ms.joints()` (`Jr @ verts`).

| group | n joints | median support (verts) | median position error |
|---|---|---|---|
| leg_prox (co/tr/fe) | 18 | 198.5 | 0.0175 |
| leg_distal (ti/ta/pt) | 18 | 39.0 | **0.0255** |
| body_axis (b_h/b_t) | 2 | 1777.5 | 0.0173 |

Only a **1.5x** gap in *median* error — far too small to explain R collapsing from ~0.75 to 0.14
on its own if error were uniform. But distal joints show mean >> median (e.g. `l_1_pt_l` median
0.091, mean 0.306; `l_2_pt_l` median 0.018, mean 0.290) — a **heavy tail**: a few specimens with
catastrophic mislocalization (worst-case endpoint errors 0.57-1.18, in a frame where full body
extent is ~2), not uniformly-noisy placement. `corr(log support, log error)` across all 51
non-wing joints is weak (r=-0.21) — support alone is not the story either.

Dropping the single worst-error specimen per bone does **not** consistently fix R (`seg_l_3_ti_r`
0.604→0.894, but `seg_l_3_ti_l` 0.039→−0.251) — different specimens fail on different legs/sides,
not one dominant outlier. **Conclusion: scattered, catastrophic per-leg correspondence failures,
not a systematic localization bias.** This already argues against a definition fix: a geodesic
path drawn from the same corrupted vertex region inherits the same catastrophic failures (and
sums many noisy edges rather than one joint-regressor weighted average, so plausibly worse) —
a hypothesis for Phase 3 to test, not yet a conclusion.

## 1b — pose sensitivity, isolated

Generated `synth_clean_pose0`: identical recipe to `synth_clean` (`make_synth_corpus.py`, seed
20260806, n=12, noise=0.0, shape_scale=1.0, scale_scale=0.10) with **only** `--pose_scale 0.0`
instead of `0.25`. Fit with `SYN_clean_w5`'s exact hierarchical→moonshot recipe, changing only
`--mesh_dir` (SLURM job 3038740, `diagnostics/appendage_evidence/run_pose0_fit.sbatch`).

| block | R, pose_scale=0.25 (`SYN_clean_w5`) | R, pose_scale=0.0 (`SYN_clean_pose0_w5`) |
|---|---|---|
| leg_distal | 0.144 | **0.461** |
| leg_prox | 0.768 | 0.655 |
| antenna | 0.763 | **0.189** |
| gaster | 0.589 | 0.701 |
| mesosoma | 0.839 | 0.979 |
| head | 0.956 | 0.960 |
| mandible | 0.770 | 0.762 |

Rerunning the 1a joint-localization probe on this pair: distal-leg median joint error drops to
0.0161 (vs leg_prox 0.0142) — the proximal/distal gap essentially closes — and worst-case
endpoint errors shrink from 0.57-1.18 down to 0.05-0.28. **Pose magnitude (bending) is a real,
substantial driver of the leg_distal catastrophic-correspondence failures identified in 1a** —
this is consistent with self-occlusion/ambiguity on bent thin structures being harder for the
optimizer to resolve than on near-canonical pose.

**But this is not a universal "less pose = more reliable" effect** — antenna gets dramatically
*worse* at pose_scale=0 (0.763→0.189, SNR 1.27→0.57), and leg_prox gets slightly worse too
(0.768→0.655). So pose variation is not simply noise to be minimised; for some thin structures
(antenna) a varied pose may help the optimizer disambiguate what would otherwise be a more
symmetric, more confusable near-canonical configuration. At n=12 per corpus, treat both numbers
as directional, not final — a single specimen set per pose regime is not enough to rule out
sampling noise for individual blocks, though the leg_distal effect (0.144→0.461) and its matching
joint-error-magnitude story are large and mutually consistent, not a lone number.

## What this means for Phase 2

1. The leg_distal failure is **primarily an optimizer/correspondence problem that is aggravated
   by pose**, not primarily a flaw in "joint-to-joint distance" as a measurement definition.
   Candidate measurement definitions (Phase 2/3) should be scored **both** at the existing
   pose_scale=0.25 ceiling test (matches production reality — real specimens are not posed
   canonically) **and** the pose_scale=0.0 corpus, to see whether any candidate is specifically
   more ROBUST to pose-induced correspondence failure, not just better on one fixed sample.
2. Per the working hypothesis stated in the plan: a geodesic/centerline path over the same
   low-support, pose-corrupted vertex patch is not expected to fix a correspondence-level failure
   and could plausibly be more fragile (many small noisy edges vs one weighted-average joint
   position) — Phase 3 must test this rather than assume it either way.
3. The joint-chain candidate (C1, summing co→tr→fe→ti→ta→pt) is now a more clearly motivated
   comparison than it looked before this probe: if catastrophic error is localized to specific
   joints/specimens rather than the whole chain, summing more (partially redundant) segments
   could average some of it out — worth prioritising ahead of geodesic in Phase 3's write-up
   order, though both are still built and scored.

## Reproducing

```
conda activate pytorch3d
# 1a (existing SYN_clean_w5 / synth_clean, no new fit needed)
PROBE_RUN=SYN_clean_w5 PROBE_CORPUS=synth_clean python diagnostics/appendage_evidence/probe_1a_joint_localization.py

# 1b corpus + fit (already run, SLURM job 3038740) + rescoring
python diagnostics/moonshot/make_synth_corpus.py --n 12 --noise 0.0 --pose_scale 0.0 --out synth_clean_pose0
sbatch diagnostics/appendage_evidence/run_pose0_fit.sbatch
python diagnostics/absolute_scale/calib_features.py --runs SYN_clean_pose0_w5 --primary SYN_clean_pose0_w5 --corpus synth_clean_pose0
PROBE_RUN=SYN_clean_pose0_w5 PROBE_CORPUS=synth_clean_pose0 python diagnostics/appendage_evidence/probe_1a_joint_localization.py
```

Artifacts: `diagnostics/appendage_evidence/out/probe_1a_joint_localization_{SYN_clean_w5,
SYN_clean_pose0_w5}.json`, `diagnostics/absolute_scale/out/feature_reliability.json` (overwritten
per run — rerun the `SYN_clean_w5` baseline command above to restore it if needed), fit logs under
`diagnostics/appendage_evidence/logs/`.
