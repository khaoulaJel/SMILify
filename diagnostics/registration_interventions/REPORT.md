# Task 7 — registration intervention study

Status: **IN PROGRESS**. A: REJECTED (§3). B (geometry-only pose init): REJECTED at pre-registered
validation, never wired into a fitting run (§4). C Phase C1 (stratified distal target sampling,
pose0/pose25 only): fit, evaluated, AND per-specimen/bootstrap-characterized (§5) — verdict HOLD:
the first intervention with a real, mechanistically interpretable, non-null effect (moves
cross-leg confusion at pose25), but not robustly dose-responsive once tested with per-specimen
statistics (only the smallest tested quota, q05, clears a paired bootstrap CI) and not
collateral-free (antenna). Not production-ready. Phase C2 (damage corpora) and any new
intervention are explicitly NOT started, pending a decision on how to proceed from here. This
file is updated as evidence lands, not written after the fact.

## 1. Executive verdict

**Intervention A (`--split_distal`) is REJECTED as a standalone fix.** `leg_distal` R moves in
*both* directions across the four tested conditions (pose0: 0.461→0.415 worse; pose25:
0.144→0.226 better; drop30: 0.144→0.185 better; drop60: 0.003→−0.174 worse, and now
anti-correlated with truth) with no consistent sign, and — decisively — the two mechanistic
metrics that would explain a real fix (within-leg segment confusion, distal joint-position error)
are essentially flat everywhere. Pretarsus within-leg mismatch stays at 0.80-1.00 in nearly every
condition, split_distal or not. This is the "nothing changes" outcome flagged in advance as
informative: separating the distal loss term does not, by itself, fix the correspondence ambiguity
or the local-minima trapping that Task 6 identified as the actual bottleneck. The small,
condition-dependent R movements are best read as optimization noise riding on top of an unchanged
underlying mechanism, not a mechanistic improvement — see §3 Failure analysis for the full chain.
Per the mission's own decision framework, this fails Gate 2 (mechanistic improvement) regardless
of Gate 1 (a same-run collateral-damage pattern also shows up, see §3E). **Recommendation: do not
proceed to A+B or A+C. Move to Intervention B (pose-aware init) as an independent test next**,
since it targets the local-minima mechanism directly (the one lever — D5 GT-init — that has so
far produced a large, mechanistically-explained effect) rather than re-tuning the loss-partitioning
mechanism that this experiment shows is insufficient alone.

## 2. Starting diagnosis (Task 3 / Task 6, already established, not re-derived here)

- `leg_distal` ground-truth reliability is poor (R≈0.136-0.144) and this is a **registration**
  failure, not a measurement-definition problem (Task 3: C1/C2 alternate definitions REJECTED).
- Root cause chain (Task 6): thin distal geometry → severe area-weighted sample starvation
  (pretarsus median ~2/8000 samples, 99.5% of resample windows below the `n_t<10` skip
  threshold) → weak/ambiguous distal supervision → the optimizer is TRAPPED in bad local minima
  reachable from zero-init (GT-init nearly quadruples leg_distal R at pose_scale=0.25, 0.144→
  0.604, and does not drift back) → pose-dependent cross-leg/within-leg correspondence failure.
- Two hypotheses tested and NOT supported: pretarsus joint-limit exploitation (D3), offset-penalty
  dilution (D4). Do not re-litigate without new evidence.
- Full detail: `diagnostics/appendage_evidence/REPORT.md`, `diagnostics/registration_failure/REPORT.md`.

## 3. Intervention A — `--split_distal`

### Mechanism
See `MECHANISM.md` in this directory. Summary: gives each leg's distal segments
(tibia/tarsus/pretarsus) their OWN target-partition group and chamfer term, instead of pooling
them with the 94%-of-area proximal segments inside one per-leg term. Changes target ASSIGNMENT
and LOSS PARTITIONING only; sampling density, optimizer, lr, joint limits, pose init unchanged.
Default (flag off) is byte-identical to every pre-Task-7 run.

### Experiment
- Baseline arm: `SYN_clean_pose0_w5`, `SYN_clean_w5` (pose_scale=0.25), `SYN_clean_drop30_w5`,
  `SYN_clean_drop60_w5` — all already fit under Task 3/6, reused unchanged, NOT re-run.
- Intervention arm: same recipe + `--split_distal` on the hierarchical stage only.
  `SYN_clean_pose0_splitdistal_w5`, `SYN_clean_pose25_splitdistal_w5`,
  `SYN_clean_drop30_splitdistal_w5`, `SYN_clean_drop60_splitdistal_w5`.
  Command: `diagnostics/registration_interventions/run_A_splitdistal_fit.sbatch` (SLURM job 3045544).
- Everything else held fixed: `--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0`, full
  default budget, `n_sample=8000`, `reassign_every=50`, `D1_low.yaml` moonshot handoff, seed 0.
- pose_scale=0.10/0.35 sweep (nice-to-have per mission spec) not yet queued; add after the
  priority-4 result is in, if compute allows.

### Metric D (sample starvation) — geometric, no fit needed, already answerable
`diagnostics/registration_failure/out/d2a_sample_starvation.json` (Task 6, pose sweep) already
reports `distal_median`/`distal_p10`/`p_distal_zero` per specimen with `distal_groups` = ti+ta+pt
POOLED per leg — this is EXACTLY split_distal's "distal" group, so Task 6's numbers already
answer "how starved is the group split_distal creates" for pose0..pose35 without any new run.
`probe_A_d2a_damage.py` (this directory) attempted the same for drop30/drop60 but **failed**: the
damage corpora physically remove distal faces (different face count/topology per specimen, not a
template-preserving relabelling), so the template-indexed `face_group` array does not apply
as-is. Not yet resolved — would need per-specimen nearest-template-vertex remapping. Documented
as an open gap, not silently skipped.

### Results

All numbers from `diagnostics/registration_interventions/out/` (`calib_*.log`,
`joint_*_{baseline,splitdistal}.log`, `A_d2b_correspondence.log`/`.json`), produced by
`run_A_eval.sh` against the completed fits (SLURM job 3045544, all 4 tasks COMPLETED, ~9 min
each — well under the 2h budget).

**A. `leg_distal` reliability (calib_features.py, exact Task 3/6 methodology), baseline → split_distal**

| condition | leg_distal R | leg_distal SNR | leg_distal \|bias\| | ALL-blocks R |
|---|---|---|---|---|
| pose0  | 0.461 → 0.415 (**−0.046**) | 0.49 → 0.49 | 8.6% → 8.8% | 0.717 → 0.724 |
| pose25 | 0.144 → 0.226 (**+0.082**) | 0.36 → 0.49 | 12.3% → 10.1% | 0.745 → 0.735 |
| drop30 | 0.144 → 0.185 (**+0.041**) | 0.27 → 0.46 | 24.2% → 25.3% | 0.729 → 0.693 |
| drop60 | 0.003 → **−0.174** (worse, now anti-correlated) | 0.25 → 0.35 | 38.8% → 39.1% | 0.607 → 0.656 |

No consistent sign. The condition where the intervention should help most on the stated
mechanism (drop60 — maximal starvation, maximal need for an independent distal term) is where it
does the most damage.

**B. Distal joint localization (probe_1a_joint_localization.py), median normalized joint-position error**

| condition | leg_distal joint err, baseline | leg_distal joint err, split_distal | Δ | leg_prox joint err, baseline | leg_prox joint err, split_distal |
|---|---|---|---|---|---|
| pose0  | 0.01609 | 0.01602 | ~0 (−0.4%) | 0.01419 | 0.01496 (+5.4%, worse) |
| pose25 | 0.02550 | 0.02576 | ~0 (+1.0%) | 0.01745 | 0.01731 |
| drop30 | 0.07606 | 0.07719 | ~0 (+1.5%) | 0.01852 | 0.01884 |
| drop60 | 0.17718 | 0.16788 | −5.2% (mild improvement) | 0.02330 | 0.02249 |

Essentially flat in 3 of 4 conditions. Nothing resembling the D5 GT-init result (which nearly
quadrupled leg_distal R by closing the proximal/distal joint-error gap) shows up here — the
distal joints are not being materially better localized.

**C. Correspondence (probe_A_d2b_correspondence.py), mean across 12 specimens**

| condition | cross-leg confusion (ti/ta/pt), baseline | cross-leg, split_distal | within-leg mismatch (ti/ta/pt), baseline | within-leg, split_distal |
|---|---|---|---|---|
| pose0  | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.25 / 0.49 / 0.84 | 0.24 / 0.45 / **0.90** (worse on pt) |
| pose25 | 0.21 / 0.22 / 0.28 | 0.18 / 0.20 / 0.24 (mildly better) | 0.35 / 0.43 / 0.80 | 0.38 / 0.42 / 0.81 (flat/worse) |
| drop30 | 0.88 / 0.94 / 0.95 | 0.88 / 0.95 / 0.94 (flat) | 0.51 / 0.83 / 1.00 | 0.54 / 0.73 / 1.00 (mixed) |
| drop60 | 0.97 / 0.95 / 0.99 | 0.96 / 0.94 / 0.99 (flat) | 0.76 / 1.00 / 1.00 | 0.77 / 1.00 / 1.00 (flat) |

Pretarsus within-leg mismatch is 0.80-1.00 in **every single condition, both arms** — the
optimizer essentially never finds the pretarsus's true segment correspondence regardless of
whether it has its own loss term. Cross-leg confusion shows a small, consistent-direction
improvement at pose25 (the one condition with intermediate, non-saturated confusion) but is
already ~0 (pose0) or already saturated near 1.0 (drop30/60) elsewhere, so there is little room
for the mechanism to show up even if it were real.

**D. Sample starvation** — unchanged by construction. `--split_distal` does not add samples, only
repartitions and reweights the same draw; Task 6's existing pose-sweep numbers
(`diagnostics/registration_failure/out/d2a_sample_starvation.json`) already describe the group
split_distal creates, and they are the same numbers regardless of the flag: pretarsus median ~2
target points per leg per resample window, 99.5% of windows below the fitter's own `n_t<10`
group-skip threshold. This matters directly for reading A/B/C above: if the distal group is being
skipped almost as often post-split as the whole-leg group was pre-split (because 2 points still
fails `n_t<10` even as its own group), then split_distal's independent-weighting mechanism cannot
act most of the time — exactly the failure mode literature §8.6 flagged as "an empirical question
this experiment must measure directly." This experiment did not instrument the skip rate directly
(would need a logging change inside `_partitioned_chamfer`, not done — a real gap, noted not
glossed over), but the flat B/C numbers above are consistent with the skip firing at a similar
rate as before and the mechanism rarely engaging.

**E. Collateral effects on other anatomical blocks** (ΔR, baseline→split_distal, calib_features.py)

| condition | antenna | gaster | head | leg_prox | mandible | mesosoma |
|---|---|---|---|---|---|---|
| pose0  | +0.008 | −0.020 | +0.005 | +0.001 | +0.006 | 0.000 |
| pose25 | **−0.088** | −0.034 | −0.030 | −0.008 | **−0.072** | +0.046 |
| drop30 | −0.026 | +0.078 | +0.013 | **−0.043** | +0.019 | −0.020 |
| drop60 | −0.041 | +0.028 | −0.009 | +0.067 | +0.019 | −0.005 |

Not a clean win-with-no-cost. pose25 in particular shows real, non-trivial regressions on
antenna and mandible (both larger in magnitude than the leg_distal gain that condition produced)
— a genuine instance of the resource-reallocation failure this project already knows is possible,
even though nothing here collapses catastrophically.

### Plots

Not generated this pass — the numeric tables above already show no clean, monotonic signal to
plot, and generating the seven mission-spec plots against a rejected/inconclusive mechanism
result would overstate its importance. Revisit if the eventual pose sweep (pose05/10/15/20/35,
not yet run) shows a clearer pattern than the 4-point priority set does.

### Failure analysis

Walking the hypothesized chain (split_distal → better distal supervision → less within-leg
confusion → better distal joint localization → fewer catastrophic failures → higher leg_distal
R) against the evidence above, **the chain breaks at the first link**: within-leg segment
confusion (Metric C) does not meaningfully improve, and where it's already near 0 or near 1 there
was no room for it to. Because the first link doesn't hold, it is unsurprising that the
downstream links (Metric B joint localization, Metric A reliability) also don't show a clean
effect — the small, sign-flipping R movements are the residual of ordinary fit-to-fit variance
(different random target-point draws, different local optimum reached by an otherwise-identical
but not-fully-deterministic optimization), not evidence of the claimed mechanism. Metric D
explains *why* the first link doesn't hold: split_distal changes how the (~2-sample) distal group
is weighted, not how many points that group has to work with, and if the group is still tripping
the `n_t<10` skip at a similar rate, the reweighting mechanism has little to act on. This matches
this project's own literature review, §8.6: "up-weighting a small, sparsely-sampled region... can
amplify noise... precisely why the `n_t<10` skip threshold exists" — the skip threshold and the
split_distal mechanism are in tension by construction, and this experiment suggests the skip is
usually winning.

### Verdict

**HOLD / REJECT as a standalone intervention.** Gate 2 (mechanistic improvement) is not met —
correspondence and joint localization do not improve in the way the mechanism predicts. Gate 3
(no collateral damage) is not cleanly met either (pose25 antenna/mandible regressions exceed that
condition's leg_distal gain). Gate 1 (synthetic reliability) is not even consistently met (2 of 4
conditions worse). Do not promote to production. Do not run A+B or A+C combinations — per the
mission's own discipline, combining an intervention that doesn't demonstrably work with another
one would make any subsequent result uninterpretable. **Next step: run Intervention B
(`--pf_init`) as an independent test**, since it attacks the local-minima/initialization
mechanism that D5 already showed has real leverage (GT-init: 0.144→0.604), rather than continuing
to iterate on the loss-partitioning mechanism this experiment shows is insufficient alone. If
useful, a smaller follow-up before B — instrumenting the actual `n_t<10` skip rate for the split
distal group directly (Metric D's stated gap) — would confirm or refute the specific "starvation
survives the split" explanation above before moving on, but is not required to reach the HOLD
verdict, which stands on the correspondence/localization evidence alone.

## 4. Intervention B — pose-aware initialization (`--pf_init`)

**REJECTED at the pre-registered validation stage — never reached a fitting run.** Design record:
`MECHANISM_B.md`. `--pf_init` (the mission's original suggestion) is dead code on `to-ship`
(`part_anchor_init.py` doesn't exist here) and its fixed version was already tested on
`feature/registration_moonshot`, buying only `edge_logratio +5.1%` and "nothing else," confounded
with a separately-failing frozen learned partition — not a clean prior test of "does
target-informed init help." Built a new geometry-only initialiser instead
(`fitter_3d/geom_leg_init.py`): analytic coxa anchors from H0's shared rigid fit + template rest
geometry, nearest-anchor (not fitted-correspondence) leg assignment, geodesic-graph curve
estimation per leg, closed-form FK-inversion IK. No GT, no learned model, no fitted
correspondence — full audit in the module docstring.

**Pre-registered validation** (`probe_B0_geom_init_validation.py`, SLURM job 3053017, COMPLETED):
runs real H0 fits (identical recipe to baseline) on all 4 conditions, then scores the
initialiser's predicted joint positions against ground truth — evaluation only, GT never seen by
the initialiser itself — BEFORE any fitting run uses it, per the mission's explicit requirement.

**Result: the geometric initialiser is worse than zero-init at EVERY leg joint, in EVERY
condition, with no exceptions of consequence** (coxa joints are a wash, ±0.001, since the
initialiser doesn't move them; every trochanter/femur/tibia/tarsus/pretarsus joint is worse,
usually 2-4x, sometimes >10x):

| condition | leg_prox median\|err\|, zero→geom | leg_distal median\|err\|, zero→geom |
|---|---|---|
| pose0  | ~0.017 → ~0.19 (**~11x worse**) | ~0.028 → ~0.44 (**~16x worse**) |
| pose25 | 0.0564 → 0.1192 (2.1x worse) | 0.1593 → 0.4378 (2.7x worse) |
| drop30 | 0.0631 → 0.1309 (2.1x worse) | 0.2101 → 0.4351 (2.1x worse) |
| drop60 | 0.0650 → 0.1438 (2.2x worse) | 0.2576 → 0.4764 (1.9x worse) |

pose0 is the starkest case: zero-init is already NEAR-CORRECT there (true pose IS zero), so
"doing nothing" gives leg_distal error ~0.02-0.03, and the geometric initialiser actively
destroys that, pushing error up to 0.3-0.7. This rules out any reading where the initialiser is
merely "not helpful" — it is actively harmful, on real target point clouds, in every tested
condition. Full per-joint numbers: `diagnostics/registration_interventions/out/B0_geom_init_validation.json`;
raw log: `diagnostics/registration_interventions/logs/B0_val_3053017.out`.

**This differs sharply from the same mechanism's behavior on a synthetic self-consistency check**
(a target built directly from the rest pose, zero articulation — see `geom_leg_init.py`'s own
design history and `MECHANISM_B.md`), where after two bug fixes (an FK root-pivot error; a
Euclidean-vs-graph-distance flaw) it recovered near-zero rotation for most joints. The gap
between that synthetic result and this real-data failure is itself informative and is the leading
(labeled: **suggestive, not verified**) hypothesis for what went wrong: the synthetic
self-consistency check used points sampled ALONG a 1-D curve (interpolated between joint
centers), where the geodesic graph has no ambiguity. Real target points are an AREA-weighted
sample over the leg's actual surface, which has real width relative to its length, especially on
the bulky proximal segments that hold 94% of a leg's surface area (Task 6). A k-NN graph over
such a sample can plausibly "short-circuit" — a point on the far side of a bulky segment's
surface can be graph-adjacent to a near-side point on an ADJACENT segment through a short local
jump across the segment's own thickness, corrupting the cumulative geodesic-distance-from-coxa
estimate in exactly the joints that then show the largest errors (trochanter/femur, the bulkiest,
most area-dominant segments — consistent with the table above, where leg_prox is already badly
wrong before leg_distal even has a chance). This has NOT been separately verified (e.g. by
inspecting per-specimen band assignments) and should not be treated as established without doing
so.

**Verdict: this specific geometric-initialisation design is REJECTED. Do not wire it into a
fitting run** — a fitting run downstream of an initialisation that is already several times
worse than doing nothing would not be an informative test of "does target-informed
initialisation help," it would be a test of "can the chamfer stages recover from a bad start,"
which is a different and less interesting question this project has already partly answered (Task
6: raising pose lr to recover faster from a bad start made things WORSE, not better). Per the
user's explicit instruction ("if the initializer requires assumptions that are only valid because
GT labels are available, stop and redesign it rather than proceeding" / evidence-gated escalation
only), this stops here pending a decision on how to proceed: (a) redesign the curve-extraction
step specifically to be robust to area-sampled surface width (e.g. restrict the k-NN graph to a
locally-thinned/skeletonized subset of points, or use a wider geodesic band with an explicit
outlier-rejection step), (b) treat this as evidence that geometry-only, learned-model-free pose
initialisation is not straightforward to build reliably at this stage of the project and
deprioritize Intervention B in favor of C (increased distal sampling) or a literature-informed
alternative, or (c) something else the user specifies. Not decided unilaterally — this is
exactly the "equally valuable if it doesn't work" outcome flagged in advance: D5's GT-init
advantage has NOT been reproduced by this class of geometric initialisation, and the reason
appears to be that raw geometric curve-extraction on real, thick, area-sampled leg surfaces is
harder than the mechanism assumed, not that pose initialisation in general is the wrong idea.

## 5. Intervention C — stratified distal target sampling

**Status: IN PROGRESS.** Full pre-registered design, leakage analysis, and pre-flight validation
(ALL CHECKS PASSED, before any fitting job): `MECHANISM_C.md`. Summary:

- **Research question**: does more actual distal TARGET evidence (not a loss-partitioning change,
  not an initialization change) let the existing optimizer avoid the wrong basin? Tests the
  strongest remaining untested mechanism from Task 6/D2a (pretarsus median ~2/8000 samples, 99.5%
  of windows below `n_t<10`) directly, decoupled from split_distal (A, HOLD/REJECT) and from
  geometric initialisation (B, REJECT).
- **Mechanism**: `fitter_3d/stratified_sampling.py`, wired into `HierarchicalStage` via a new
  opt-in `distal_quota` parameter (default `None`, provably unchanged baseline path). Applies to
  H1/H2/H3 TARGET sampling only (H0's global body chamfer and ALL source/fitted-mesh sampling are
  untouched). Distal-face membership comes only from template skinning weights (no GT, no fitted
  correspondence, no learned model).
- **Quotas** (pre-registered, not tuned): baseline (natural, ~11.3% of a leg's own sample share,
  measured from Task 6's own D2a output) / 5% / 10% / 20%. The optional 30% arm was dropped for
  scope, decided before running.
- **Phase C1** (this pass): pose0 and pose25 only (topology-preserving corpora — stratification
  needs template-face correspondence, which drop30/drop60 break by remeshing, confirmed directly
  when this exact issue blocked `probe_A_d2a_damage.py` earlier). Phase C2 (damage-compatible
  design) explicitly deferred, not attempted with a GT-dependent shortcut.
- **Experiment**: baseline arms (`SYN_clean_pose0_w5`, `SYN_clean_w5`) reused unchanged from
  Task 3/6, NOT re-run. New arms: `SYN_clean_{pose0,pose25}_stratq{05,10,20}_w5`, SLURM job
  3054996 (array 0-5), `run_C_stratified_fit.sbatch`. Everything else byte-identical to the
  existing recipe (`--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0`, full budget,
  `D1_low.yaml` handoff, no `--split_distal`).
- **Evaluation**: `run_C_eval.sh` (Metric D: `leg_distal` R/SNR/bias via `calib_features.py`;
  Metric C: joint localization via `probe_1a_joint_localization.py`; Metric B: correspondence via
  `probe_C_d2b_correspondence.py`, same fixed-yardstick design as Intervention A's audit).

### Results (SLURM job 3054996, all 6 arms COMPLETED, `run_C_eval.sh`)

**Metric D — `leg_distal` reliability, baseline -> quota (calib_features.py):**

| condition | q=natural(baseline) | q=5% | q=10% | q=20% |
|---|---|---|---|---|
| pose0  | R=0.461 | R=0.459 | R=0.439 | **R=0.396** |
| pose25 | R=0.144 | R=0.289 | R=0.307 | **R=0.354** |

A genuine, roughly MONOTONIC dose-response in BOTH conditions -- but in OPPOSITE directions.
At pose25, more distal evidence helps, roughly linearly, up to +0.21 R at q20 (0.144->0.354, near
a 2.5x increase). At pose0, more distal evidence HURTS, monotonically (0.461->0.396).

**Metric B — correspondence (`probe_C_d2b_correspondence.py`, mean across 12 specimens):**

| condition | xleg_pt (cross-leg, pretarsus) | wl_pt (within-leg mismatch, pretarsus) |
|---|---|---|
| pose0, baseline->q20  | 0.000 -> 0.000 (already zero) | 0.841 -> **0.927 (worse)** |
| pose25, baseline->q20 | 0.252 -> 0.194 -> 0.157 -> 0.203 (net **improved**, best at q10) | 0.848 -> 0.771 -> 0.823 -> **0.901 (worse at q20)** |

**Metric C — joint localization (`probe_1a_joint_localization.py`):** median `leg_distal` error is
nearly flat with quota at pose25 (0.0255 baseline -> 0.0239/0.0257/0.0257 at q05/q10/q20) despite
the clear R gain -- but the MEAN (tail-sensitive, unlike median) drops materially: mean-of-means
`leg_distal` error 0.172 (baseline) -> 0.145 (q05) -> 0.164 (q20). This is the same heavy-tail
signature Task 3/6 already established for this measurement (R tracks a few catastrophically
mislocalized specimens, not the bulk median) -- consistent with the mechanism being "fewer
catastrophic failures," not "uniformly better localization."

**Metric E — collateral (calib_features.py, other blocks), pose25, baseline->q20:**
antenna 0.763->**0.502** (large, MONOTONIC regression: 0.763->0.639->0.599->0.502), head
0.956->0.919 (0.956->0.950->0.891->0.919, non-monotonic dip then partial recovery), gaster
0.589->0.514, leg_prox 0.768->0.784 (roughly flat/slightly better), mesosoma 0.839->0.836 (flat).
`ALL` (aggregate across 44 features): 0.745(baseline) -> 0.658(q05) -> 0.671(q10) -> 0.731(q20) --
net roughly flat to slightly down at the quota that helps `leg_distal` most. pose0's collateral
pattern is much smaller in magnitude (all blocks within ~0.02 of baseline) since the mechanism
that helps at pose25 doesn't activate there.

### Per-specimen characterization and formal statistics (added after the cohort-level pass above;
this REVISES how confidently the pose25 dose-response should be read — see below)

`probe_C_per_specimen.py` (per-specimen `leg_distal`/`leg_prox`/`antenna` joint error, all 4 arms,
both conditions) + `plot_C_mechanism.py` (4 plots: paired per-specimen slopes, antenna-vs-distal
co-movement, quota-vs-cross-leg-confusion, error distributions). Raw: `out/C_per_specimen.json`.
Plots: `out/C_plot{1,2,3,4}_*.png`.

**Paired bootstrap 95% CIs, mean `leg_distal` error reduction (baseline − arm), pose25, n=12,
5000 resamples:**

| quota | mean reduction | 95% CI | significant? |
|---|---|---|---|
| q05 | 0.0277 | (0.0010, 0.0624) | barely excludes 0 |
| q10 | 0.0134 | (−0.0008, 0.0362) | **CI includes 0** |
| q20 | 0.0087 | (−0.0087, 0.0338) | **CI includes 0** |

**This is a real correction to the cohort-R story above, not a footnote.** Cohort `leg_distal R`
rose monotonically with quota (0.144→0.289→0.307→0.354), but the DIRECT per-specimen mean
absolute joint-error reduction is largest at q05 and DECLINES at q10/q20, and only q05 is even
marginally distinguishable from noise at n=12. `pose0`'s CIs all include 0 at every quota (mean
reduction −0.0000 to 0.0007) — confirming the earlier "pose0 effect" was real in the R statistic
but is not statistically distinguishable from zero at the per-specimen absolute-error level either.
**R (a whole-cohort correlation statistic) and mean absolute joint error are measuring different
things and can move differently** — R is sensitive to whether the fitted-vs-true relationship
across specimens becomes more linear/consistent (which a single very bad outlier partially
improving can shift a lot), not to whether every specimen's absolute position error shrinks. Both
readings are legitimate; neither alone is the whole picture, and the R trend should not have been
presented as clean/monotonic without this caveat attached.

**Does q20 rescue catastrophic fits, or shift the average? — Neither cleanly; it is uneven and
partly UNSTABLE across quotas** (Plot 1, `C_plot1_paired_specimens_pose25.png`):
- catastrophic-specimen mean reduction (top-4 worst baseline): q05=**0.0488**, q10=0.0055,
  q20=**−0.0064** (i.e. WORSE than baseline on average at q20) — the opposite of "q20 rescues the
  worst cases."
- non-catastrophic mean reduction stays flat, ~0.016-0.017, across all three quotas.
- Specimen-level detail explains why the catastrophic average is unstable: `synth_003` (worst
  baseline, error 0.489) improves sharply at q05 (0.318) but REVERTS almost fully at q10/q20
  (0.490, 0.492) — a one-quota fluke, not a stable rescue. `synth_007` (baseline 0.184) improves
  to ~0.06 and STAYS there across all three quotas — a genuine, robust rescue. `synth_010`
  (baseline 0.186, not in the catastrophic top-4) gets progressively WORSE with quota (0.155→
  0.159→**0.228**, ending above baseline) — actively harmed by more distal sampling. So "does C
  rescue catastrophic fits" has three different answers for three different specimens in the same
  n=12 cohort: yes-and-stable, yes-then-reverts, and no-actively-harmed.

**Does antenna regression occur in the same specimens where distal legs improve?** (Plot 2,
`C_plot2_antenna_vs_distal_pose25.png`) — **No strong evidence of that specific story.**
Per-specimen correlation between distal-error improvement and antenna-error improvement is WEAKLY
POSITIVE at pose25 (r=0.245 at q05, 0.167 at q10, 0.088 at q20 — declining but not negative),
meaning specimens that improve on legs do NOT preferentially regress on antenna; if anything they
trend slightly together. The cohort-level antenna regression (§ above) therefore looks more like a
diffuse, population-wide cost of the fixed sampling budget than a specific specimen-level
trade-off ("this specimen's antenna lost its samples to that specimen's legs" is not the right
mental model — the reallocation, if real, is happening within each specimen's own budget, roughly
uniformly, not concentrated in the specimens that most benefited). At pose0 the same correlation
is NEGATIVE and larger in magnitude (r=−0.379 to −0.775), but pose0's `leg_distal` changes are
themselves noise-level (CIs above), so this correlation is plausibly just correlated noise between
two near-zero-signal quantities, not a real mechanism — flagged, not asserted.

**Quota → cross-leg confusion, pose0 vs pose25** (Plot 3, `C_plot3_quota_vs_xleg.png`): visually
confirms the earlier table — pose0 stays at exactly 0 across all quotas (no mechanism to engage),
pose25 declines from 0.252 (baseline) with a minimum around q10 (0.157) and a slight uptick at
q20 (0.203) — non-monotonic in the same way the bootstrap CIs are, not a perfectly clean curve.

### Causal-chain analysis

Tracing starvation -> correspondence -> localization -> reliability, separately per condition:

**pose25** (where SOME of the effect is real, but unevenly and not robustly dose-responsive at
n=12 -- see the per-specimen section below, which qualifies this considerably): more distal
samples measurably reduce CROSS-LEG confusion on pretarsus (0.252->0.157 at q10, the
pose-TRIGGERED mechanism from Task 6 D2b, though non-monotonically -- ticks back up to 0.203 at
q20) but do NOT reduce, and if anything worsen, WITHIN-LEG segment mismatch (0.848->0.901 at q20,
the pose-INDEPENDENT mechanism). `leg_distal` R rises monotonically with quota at the COHORT
level, but the per-specimen paired bootstrap only clears statistical significance at q05 (CI
barely excludes 0), not q10/q20 -- so "the effect strengthens with more distal samples" is true of
R but NOT confirmed true of per-specimen absolute error at this sample size. The distinction that
DOES hold up under the per-specimen check: **more distal evidence measurably helps the SPECIFIC
failure mode it can plausibly help with (which of 6 legs a point belongs to, a global 1-of-6
decision more points naturally disambiguate) without helping the different failure mode it cannot
obviously help with (which of 3 SEGMENTS within the correct leg, a local, fine-grained decision
that more of the SAME kind of area-weighted point does not disambiguate any better), and the
benefit is concentrated in specific specimens, not a uniform population shift.** This distinction
-- cross-leg vs within-leg -- is exactly the one Task 6 D2b drew, and Intervention A's failure to
move either was the headline negative result; this is the first intervention in the whole study to
move ONE of the two mechanisms at all, even if the magnitude is smaller and less uniform than the
cohort R trend alone suggested.

**pose0** (where the effect is real but harmful): there is no cross-leg confusion to fix at
pose0 (already 0.000, Task 6's own finding: cross-leg confusion is POSE-TRIGGERED, not present
at pose0). So the one mechanism stratified sampling can plausibly help is inactive here, and what
remains is a straightforward cost: fewer proximal/body samples (the complement of a fixed 8000
total) for no compensating gain, plus (Metric B) within-leg mismatch getting monotonically
WORSE with quota even here. Both `leg_distal` R and the correspondence audit degrade together,
consistently, not as noise.

**Resource reallocation, confirmed, not just possible**: the pose25 antenna regression is
monotonic and larger in absolute R terms (-0.261) than the leg_distal gain (+0.210) at q20. This
is the Outcome-C pattern pre-registered in the mission spec ("correspondence improves but other
anatomy regresses") -- not a clean win, a real trade-off, and the mission's own literature review
(REPORT.md §8.6) predicted exactly this risk in advance ("up-weighting a small, sparsely-sampled
region... can amplify noise... at the expense of") for a DIFFERENT reason (loss dilution) that
turns out to also describe a SAMPLING reallocation: distal quota necessarily removes points from
elsewhere in a fixed 8000-point budget, and antenna (a separate, also fairly thin, area-starved
structure) is where that cost concentrated here -- plausible, not separately verified this pass.

### Statistical caution

n=12 specimens per condition, as in every prior task in this study -- treated as small-n evidence
throughout. `leg_distal R` is a whole-cohort correlation statistic and is NOT the same claim as
"per-specimen absolute error improved" -- both are reported above, and they disagree in strength
(R: clean monotonic trend; per-specimen bootstrap: only q05 significant). The paired bootstrap
(5000 resamples, percentile CI) is now done (see the per-specimen section above) and is the
correct way to read this result's robustness, not the R trend alone.

### Plots

Generated (4 of the original 9, prioritized by information value given the mixed result -- the
remaining 5, e.g. full confusion matrices and optimisation trajectories, are not yet done, a
stated gap): `out/C_plot1_paired_specimens_pose25.png` (paired per-specimen slopes, catastrophic
specimens highlighted -- shows the instability described below directly), `out/C_plot2_antenna_vs_
distal_pose25.png` (antenna-vs-distal co-movement scatter), `out/C_plot3_quota_vs_xleg.png`
(quota -> cross-leg confusion, both conditions), `out/C_plot4_distribution_pose25.png`
(distribution with individual specimens visible, not just the mean).

### Failure cases (per-specimen, pose25 -- see per-specimen section above for the full table)

Three qualitatively different specimen-level outcomes coexist in the same n=12 cohort, visible in
Plot 1: **stable rescue** (`synth_007`: 0.184->~0.06, held across all three quotas), **unstable
one-quota improvement** (`synth_003`: the single worst baseline specimen, improves sharply at q05
then reverts almost fully by q10/q20 -- looks like noise in a high-leverage point, not a real
effect of higher quota), and **active harm from higher quota** (`synth_010`: 0.186->0.155(q05)->
0.159(q10)->0.228(q20), ending worse than baseline). Averaging these three patterns together is
what produces the "catastrophic mean reduction" column's own instability (0.049 at q05, collapsing
to -0.006 at q20) reported in the per-specimen section -- the aggregate number was hiding three
different stories.

### Literature grounding

§8.6's documented trade-off (independent/reweighted small-region supervision can amplify noise or
cause reallocation) is the one literature point that transfers cleanly here, and it predicted
(before this result) the exact shape of what happened: a real distal gain paired with a real cost
elsewhere. No literature source specifically predicted the cross-leg/within-leg SPLIT in where the
benefit does and does not land -- that is Task 6's own (SMILify-specific) distinction, not a
literature-derived one, and the fact that sampling density tracks it this cleanly is evidence
*for* that distinction being the right way to decompose this project's correspondence failure,
not something borrowed from prior art.

### Interpretation against the pre-registered outcomes (§11 framework)

Matches **Outcome C** most closely ("correspondence improves but other anatomy regresses" ->
investigate normalization/adaptive weighting rather than simply increasing distal samples), with
an added, pre-registration-compatible refinement the mission's outcome list did not anticipate:
the effect is **pose-conditional** (present at pose25, absent/reversed at pose0) in a way that
maps onto Task 6's cross-leg-vs-within-leg mechanism split, not just "works under bent pose" in
general (Outcome E's literal framing).

### Verdict

**HOLD, revised down from the cohort-level pass's more optimistic framing, now that the
per-specimen and bootstrap analysis is in. This is the first intervention in the whole study with
a real, mechanistically interpretable, non-null positive component at pose25 (it moves cross-leg
confusion, which A could not, and q05's bootstrap CI does clear significance) — it should not be
discarded — but it is neither uniformly dose-responsive nor robustly supported at n=12 across the
quota range, and it is not production-ready.**

- **Gate 1 (synthetic reliability)**: only q05 is statistically distinguishable from baseline at
  the per-specimen level (CI barely excludes 0); q10 and q20 — the arms with the HIGHER cohort R
  — are NOT distinguishable from noise at n=12, despite the cohort statistic suggesting they are
  the stronger arms. This is the single most important qualifier on the whole result: **do not
  read the monotonic R curve as "more quota is monotonically better" without this caveat.**
- **Gate 2 (mechanistic improvement)**: PARTIALLY met, and less cleanly than first reported —
  cross-leg confusion improves but non-monotonically (best at q10, not q20), within-leg mismatch
  does not improve at all, and the population-level gain is not a uniform shift: it is the net of
  one stable rescue, one unstable one-quota fluke, and one specimen actively harmed by higher
  quota, averaged together (per-specimen section).
- **Gate 3 (no collateral damage)**: NOT met at pose25 (antenna regression exceeds the leg_distal
  gain at q20) and the intervention is harmful, though not statistically significant, at pose0.
  The per-specimen check does at least rule out the specific worry that antenna loss is
  concentrated in the same specimens that most benefited (weak POSITIVE correlation, not
  negative) — the cost looks diffuse/population-wide rather than a sharp specimen-level trade,
  which is a materially different (and slightly less concerning) collateral story than "helping
  legs directly steals from that same specimen's antenna."

Do NOT wire this into production. Do NOT combine with A or B (both already REJECTED/HOLD) or move
to Phase C2 (damage corpora) yet, per the mission's own sequencing rule — this result is not yet
internally coherent enough to justify that additional compute. Two evidence-based next steps, if
this line is pursued further, in order of how directly they follow from what was just measured:
(1) since q05 is the only quota with a statistically clear per-specimen effect and it is BELOW the
natural baseline fraction (~11.3%), the dose-response hypothesis itself should be re-examined with
finer-grained quotas (e.g. 3%, 7%, natural) rather than assumed monotonic between the three tested
points; (2) a pose-CONDITIONAL quota (apply oversampling only once cross-leg ambiguity is actually
present) would directly test whether the pose0 harm is avoidable without giving up the pose25
q05-level gain. Both are proposals, not commitments — no further compute has been spent on either
without a separate decision to do so.

### What this means for the standing question ("what is limiting the current recipe" /
"what would most improve morphometric data")

We now have a materially better answer to the FIRST half than before this task, and should stay
appropriately cautious on the second. Defensible statement: **the fit is strongly limited by
correspondence, particularly on thin distal-leg geometry — the pretarsus receives a median of
~2/8000 target samples under standard area-weighted sampling. Deliberately increasing distal
target evidence measurably reduces pose-triggered cross-leg confusion and improves `leg_distal`
reliability at bent poses in SOME specimens, but the effect is uneven across specimens, not
cleanly dose-responsive once tested with proper per-specimen statistics, does not touch the
separate pose-independent within-leg segment ambiguity, and introduces a real (if diffuse)
collateral cost elsewhere (antenna).** We do NOT yet have a best recipe — we have a better
characterization of what is limiting the current one than of what the optimal fix looks like, and
that distinction should be preserved in how this is reported, not collapsed into a q=20
recommendation.

## 6. Interaction experiments

Not started.

## 7. Real-data validation

Not started.

## 8. Literature comparison

Non-interactive literature pass (author knowledge + targeted web search, prioritizing anything
from the last ~3 years). Evidence grading follows section 9's categories: **established**
(multiple independent sources, textbook/survey-level consensus), **suggestive** (plausible,
argued in at least one credible source, but not cross-validated the way section 9's own claims
are), **no clear precedent** (searched, nothing directly on point found). This section grounds
the mechanism in prior art only — it does not touch, and should not be read as commentary on,
Intervention A's (still pending) results.

### 8.1 Is the diagnosis itself a known failure mode?

**Established.** ICP-family registration converging to a wrong local optimum under a poor
initial alignment is the founding caveat of the method, present since Besl & McKay (1992) and
formalized in convergence analyses since (e.g. Rusinkiewicz & Levoy, *Efficient Variants of the
ICP Algorithm*, 3DIM 2001, and the Go-ICP line of work — Yang et al., TPAMI 2016 — which exists
specifically because standard ICP has no global-convergence guarantee and gets trapped in basins
determined by the starting pose). That a *better* starting pose reaches a materially better
optimum without drifting back once optimization continues (our GT-init result) is exactly the
signature these papers use to diagnose "wrong basin," as opposed to "insufficient gradient
signal" — under-powered-gradient failures typically do NOT show this asymmetry (recovery from a
good start would erode under continued steps if the objective itself were flat/uninformative
there; it does not in our data).

Sample starvation on thin/small structures under nearest-neighbor correspondence is also
established, though usually discussed as an *overlap* or *sampling-density* problem rather than
under our exact framing: point-to-point ICP variants are known to under-represent low-area
regions when points are drawn proportional to surface area (standard practice, e.g. in
PyTorch3D's `sample_points_from_meshes`, which this pipeline uses), and trimmed/robust ICP
literature (Chetverikov et al., *The Trimmed Iterative Closest Point Algorithm*, ICPR 2002)
exists partly to handle the asymmetric-overlap case, though its target is outlier/non-overlap
rejection, not deliberately *up-weighting* an under-sampled true-overlap region — that distinction
matters for reading point 8.6 below.

Cross-part correspondence confusion under pose change is likewise established: it is the
motivating failure case for essentially all articulated-ICP variants (see 8.2). No source found
frames all three (starvation + multimodality + cross-part confusion) as one causal chain the way
this report does, but each link individually has strong independent precedent.

### 8.2 Is `--split_distal` (semantic/part-restricted correspondence) established practice?

**Established, as a category of technique; our specific instantiation is unusually simple within
that category.** Restricting nearest-neighbor search to same-part correspondence, rather than a
flat whole-shape search, is the core idea of the articulated-ICP literature: Pellegrini et al.
generalize ICP to jointly solve per-segment rigid transforms constrained by joint connectivity
(*A Generalisation of the ICP Algorithm for Articulated Bodies*, BMVC 2007); segment-aware AICP
(SAICP) explicitly processes the kinematic tree root-to-leaf so that unstable distal segments
inherit a good local frame from their already-aligned parent before their own correspondence is
trusted (Ye & Yang, *Accurately Measuring Human Movement Using Articulated ICP…*, 3DV 2014) —
this is close in spirit to our proximal group being fit in earlier hierarchical stages before the
distal partition is introduced. More recent skeleton-aware work (SPEAL, arXiv 2312.08664, 2023)
uses learned per-point skeleton/part attention specifically so correspondence cannot cross part
boundaries in cross-source point cloud registration, and anatomy-aware articulated registration
for medical image segmentation (per-structure correspondence restriction to prevent bleed between
adjacent anatomical regions) is a recognized sub-area. `--split_distal` differs from most of these
in mechanism — it is a *loss-partitioning* change (each group gets an independent, equally
normalized chamfer term) rather than a *search-restriction* change per se, but the practical
effect (a target point already assigned to the distal group cannot be pulled into competing with
94%-of-area proximal points inside a shared average) is the same category of intervention:
segment-restricted correspondence to stop large regions from dominating small, anatomically
distinct ones.

### 8.3 Preventing near-but-distinct-structure correspondence jumps: what else exists?

Established alternatives, none currently used by this pipeline (useful as a map of what else is
available if split_distal alone under-delivers): point-to-plane / normal-aware correspondence
(Chen & Medioni, *Object Modeling by Registration of Multiple Range Images*, Image and Vision
Computing 1992) rejects matches across a normal discontinuity, which would penalize exactly the
tibia/tarsus-boundary crossings we care about, but requires reliable target normals — untested
here; robust kernels / correspondence pruning (trimmed ICP, above; TEASER++'s graph-theoretic
outlier pruning + graduated non-convexity, Yang et al., TRO 2020) address *outlier* matches, a
related but not identical problem to *wrong-but-plausible-inlier* matches (a tarsus point
genuinely IS close to a nearby coxa vertex under bad pose — it is not an outlier, it is a
correspondence ambiguity, which robust kernels alone do not resolve); multiscale / coarse-to-fine
registration (surveyed in Rusinkiewicz & Levoy 2001; more recent: Coarse-to-Fine Generalized-ICP
with trimming, and multi-scale ICP variants used in Open3D) fits coarse structure first to shrink
the basin before fine structure is trusted — this is structurally what this pipeline already does
via its hierarchical body→leg→joint→deform staging, and is the established justification for why
split_distal is introduced only inside the (already-coarsely-aligned) hierarchical stage rather
than from scratch. Semantic/part-segmentation priors (rigid-part decomposition, e.g. Chen &
Koltun-style part-based non-rigid registration; deformation-graph approaches such as Sumner et
al., SIGGRAPH 2007) are the direct ancestor of what split_distal does with an anatomical
(model-known) rather than learned partition.

### 8.4 Pose initialization without ground truth (context for Intervention B)

**Established territory, several distinct families**, none tested here yet. (1) Learned
keypoint/landmark regression to seed optimization — this is exactly what SMPLify (Bogo et al.,
ECCV 2016) and SMPLify-X (Pavlakos et al., CVPR 2019) do for humans: a CNN regresses 2D/3D
joints first, and gradient-based fitting starts from that pose rather than a canonical T-pose.
(2) Multi-hypothesis / multi-restart optimization — running several pose starts and keeping the
lowest-energy result, a generic answer to non-convexity that trades compute for basin coverage;
several 2023-2024 papers explicitly frame multi-initialization as the fix for single-optimum
local-minima failure in 3D pose fitting (e.g. "Multi-initialization Optimization Network for
Accurate 3D Human Pose Estimation," arXiv 2112.12917). (3) Centroid/second-moment matching against
a target-derived partition — closer to what `--pf_init` already implements — is a much older,
simpler idea (rigid part alignment via principal-axis/moment matching predates learned methods
and is standard in early articulated-body tracking) but is comparatively under-documented in
recent papers, which favor learned regressors; we did not find a close 2020s citation using
target moment-matching specifically for insect/arthropod-like multi-legged topologies.
(4) Animal-specific: WLDO (Biggs et al., ECCV 2020) and BARC (Rüegg et al., CVPR 2022, IJCV 2023)
both use learned pose/shape priors and network-regressed initial pose for dog SMAL fitting
precisely because naive zero/mean-pose initialization was found insufficient for optimization-only
fitting — directly analogous to our zero-init trapping result, on a different species.

### 8.5 How do SOTA articulated/animal fitters handle thin structures specifically?

**Suggestive, converging pattern, not a single settled recipe.** SMPL-X's design response to
finger/hand thinness is architectural, not loss-based: hands and face get their own
dedicated part models (MANO, Romero et al., SIGGRAPH Asia 2017; FLAME) with independent shape and
pose spaces, fit either jointly with strong part-specific priors or in dedicated stages —
i.e. "give the thin part its own model," a stronger version of "give it its own loss term."
Recent hand-reconstruction literature (structure-guided modulation, structure-aware inpainting,
2024-2025) explicitly attributes finger failure to "no explicit skeletal/topological prior" plus
"small spatial area," matching our starvation diagnosis nearly verbatim, though in a
learned-regression rather than optimization-registration setting. For animals, SMAL/BARC/WLDO
extend the shape space with per-limb scale factors (bone-length-driven, propagated via skinning
weights) rather than a specialized distal loss term — a shape-space fix, not a correspondence
fix, and not obviously transferable to a rigged (non-learned-shape-space) pipeline like this one.
We found no paper addressing insect/arthropod distal-leg-segment (tibia/tarsus/pretarsus)
registration specifically; the closest analogues are digit/finger and tail-tip fitting in
mammal-focused work.

### 8.6 Independent, non-diluted per-part loss terms: precedent and known trade-offs

**Established as a technique family (weighted/part-decoupled Chamfer variants), with documented
trade-offs that are directly relevant.** Several 2023-2025 papers on point-cloud completion
address exactly the "small structure diluted inside an aggregate Chamfer average" problem:
Flexible-weighted Chamfer Distance (arXiv 2505.14218) decouples precision and completeness
sub-objectives with asymmetric weighting; Learnable Chamfer Distance (Sci. Dir. 2023) dynamically
up-weights regions with larger reconstruction defects; "On the Structural Failure of Chamfer
Distance in 3D Shape Optimization" (arXiv 2603.09925) catalogs cases where plain Chamfer under- or
over-attends to structure at different scales. The documented trade-off, consistent across this
literature and worth taking seriously here: up-weighting a small, sparsely-sampled region makes
its own correspondence estimate noisier per-point (fewer points → higher-variance nearest-neighbor
assignment) and, given equal aggregate weight against a much larger region, can amplify that
noise's effect on the fitted pose rather than damping it — precisely the mechanism our own
`n_t<10` skip threshold is designed to guard against, and precisely why MECHANISM.md flags that
split_distal can trip that threshold *more* often for the distal group specifically as "an
empirical question this experiment must measure directly," not something to assume away. A second
documented risk, less specific to Chamfer but general to any correspondence-restriction change:
part-restricted correspondence is only as good as the partition assigning points to parts, and
that partition itself is a 1-NN classification that can mis-assign early in optimization if the
coarse pose is still far off — this is the direct literature analogue of, and justification for,
running split_distal only inside the hierarchical stage (after body/proximal alignment), not from
a cold start.

### How to apply this to Intervention A's results

If split_distal materially improves `leg_distal` reliability without degrading proximal-leg or
body metrics, that is consistent with (not proof of) the segment-restricted-correspondence
literature (8.2-8.3): the expected signature, per that literature, is a bigger gain on more-bent
poses (pose25 vs pose0) since correspondence ambiguity scales with articulation, and little or no
change under drop30/drop60 damage unless the `n_t<10` skip is separately checked, since removing
distal geometry removes the very points the new group depends on. If it fails or under-delivers,
the two most literature-grounded diagnoses to check first, before concluding the mechanism is
wrong, are: (a) the `n_t<10` skip firing more on the distal group post-split (the noise/dilution
trade-off in 8.6 — check `run_A_eval.sh`'s per-group skip-rate output, not just final R), and
(b) the partition itself being unreliable early (8.3's coarse-alignment precondition) — this would
show up as high early-iteration group-reassignment churn. Either failure mode, per this
literature, argues for combining split_distal with a better starting pose rather than abandoning
it — i.e. feeds forward into Intervention B (`--pf_init`): 8.4 establishes that target-informed
initialization (of which centroid/moment matching against a target-derived partition is a
well-precedented, if old-school, member) is the standard answer to exactly the basin-selection
problem diagnosed in 8.1, and 8.3's coarse-to-fine precondition suggests B should if anything make
A's partition more reliable, not compete with it. None of this predicts B will work; it only
establishes that "fix initialization next if a better loss alone is insufficient" is the direction
the literature points, not an ad hoc guess.

## 9. Final causal interpretation

**On Intervention A specifically** (full study interpretation pending B/C):
- **Established** (this experiment, §3): `--split_distal` does not produce a consistent,
  mechanistically-explained improvement in `leg_distal` reliability. Within-leg segment
  confusion and distal joint-position error — the two metrics the mechanism predicts should move
  first — are essentially flat across all 4 tested conditions.
- **Established** (Task 6, unchanged by this experiment): the underlying sample-starvation
  numbers for the group split_distal creates are identical with or without the flag, since the
  flag reweights rather than resamples.
- **Suggestive, not established**: the specific explanation that the `n_t<10` skip threshold is
  neutralizing the reweighting mechanism (§3 Failure analysis) — plausible and literature-
  consistent (§8.6) but not directly instrumented in this run; would need a dedicated skip-rate
  probe to confirm.
- **Unsupported**: any claim that split_distal is harmful in general — the collateral pattern
  (§3E) is real but modest and condition-dependent, not a clear net-negative signal either. The
  correct reading is "insufficient evidence of a real effect in either direction," not "proven
  harmful."

## 10. Recommended production change

None yet. No production defaults have been modified.
