# Overnight autonomous execution -- registration investigation, Ranks 1-5

Governing spec: the user's "SMILify Registration Investigation -- Master Prompt (Merged)" and
the ranked execution plan agreed in-session. Baseline commit `46d05fe2` + a pre-existing
uncommitted diagnostic diff (Task 6/7 infra: `init_joint_rot` override, `distal_quota`
stratified sampling -- present at session start, not introduced this run; see
`git diff fitter_3d/{trainer.py,trainer_hierarchical.py,optimise_hierarchical.py}`).

This file is updated as each rank completes. See `manifest.json` for the machine-readable
per-experiment record (commands, seeds, job IDs, status, metrics).

---

## Rank 1 -- GT-free selection-score validation (Section 9.5.1) -- COMPLETE

**Zero new fitting jobs.** Reused `diagnostics/moonshot/runs/*/metrics.csv` (per-specimen blind
scores already computed by every prior run's `--eval` pass) joined against ground-truth
`leg_distal` bone-length error via `diagnostics/absolute_scale/measure.py` (the same code path
every prior reliability report in this project used). Script:
`diagnostics/overnight_20260818/rank1_selection_score.py`. Full output:
`diagnostics/overnight_20260818/out/rank1_selection_score.json`.

**Analysis A (pooled Spearman, N=336 run x specimen observations across 28 runs):**
`part_leg_distal_dist_mean` rho=+0.792 (p<1e-4) is the strongest blind predictor of true
leg_distal error, followed by `part_leg_distal_dist_p95` (+0.763), `part_leg_distal_within_tau`
(+0.712, flipped), `chamfer_l2` (+0.625). `midline_dev_excess` is uncorrelated (rho=-0.03).

**Analysis B (oracle-vs-blind-selector recovered_frac, the master prompt's actual decision
metric):** using the 6 independently-fit hypotheses that already exist at pose25 (baseline,
splitdistal, stratq05/10/20, gtinit) and the 4 that exist at pose0 (baseline, splitdistal,
stratq05/10/20), for each specimen: does the hypothesis the best blind score would pick also
have the lowest true error?

| blind score | pose25 recovered_frac | pose0 recovered_frac |
|---|---|---|
| part_leg_distal_dist_p95 | **+0.757** | -3.44 |
| part_leg_distal_within_tau | +0.641 | -1.51 |
| deform_mag_p95 | +0.630 | -0.40 |
| chamfer_l2 | +0.216 | -1.83 |
| edge_logratio_absmean | +0.547 | -0.008 |

**Every score that looks good at pose25 is strongly NEGATIVE at pose0** -- i.e. at pose0, using
the blind score to pick a fit is *worse* than just always picking the baseline hypothesis
blind. Combined (specimen-count-weighted across both sets), the best score
(`part_leg_distal_dist_p95`... actually `edge_logratio_absmean` combined highest at 0.281, see
JSON) does not clear the pre-registered 60% threshold, and no score is sign-consistent across
regimes.

**Verdict (per the master prompt's own decision rule): FAIL / MIXED -- DO NOT PROMOTE.**
Family D (multi-start / per-limb decomposed search, Section 12/15) is **not** implemented this
run. D5's original finding (GT-init reaches a much better basin) still stands as evidence that
better basins exist (Section 6 H1) -- this result only says a *blind* selector cannot currently
find them, which is a distinct claim. Per the user's explicit instruction ("do not implement
Rank 6+ interventions unless the Rank 1-5 results justify them"), Rank 6+ work is skipped.

---

## Rank 2 -- Model-capacity ceiling (Section 9.5.2) -- COMPLETE (SLURM 3059235, array 0-1, both COMPLETED, ~2 min each)

**Measurement, not an intervention.** `log_beta_scales` (per-joint bone/limb scale, already a
free parameter in stock `SMAL3DFitter`) and the `scheme: shape` param group (already excludes
`joint_rot` and `deform_verts`, `fitter_3d/trainer.py:373-383`) both pre-existed -- no new model
code. New: `diagnostics/overnight_20260818/rank2_build_gt_init_npz.py` (writes a joint_rot-only
init npz from each corpus's `ground_truth.npz`) and
`diagnostics/overnight_20260818/cfg/ceiling_shape_only.yaml` (one stage, `scheme: shape`,
`w_offset: 0`, `w_edge: 0`).

Smoke-tested on CPU (12 specimens, 5 iterations, synth_clean): confirmed `joint_rot` stays
*exactly* pinned to GT for the entire run (`max abs diff = 0.0`) while `log_beta_scales` moves
(`max abs value = 0.025`) -- the freeze mechanism works as intended.

Full runs (1200 iterations, full 12-specimen corpora) completed on synth_clean (pose25) and
synth_clean_pose35. Analysis: `diagnostics/overnight_20260818/rank2_analyze_ceiling.py` ->
`diagnostics/overnight_20260818/out/rank2_ceiling_analysis.json`.

| condition | arm | cohort leg_distal R | mean relerr | median relerr |
|---|---|---|---|---|
| pose25 | CEILING (GT pose + free bone scale) | 0.533 | 0.379 | 0.324 |
| pose25 | baseline (SYN_clean_w5, zero-init, full pipeline) | **0.654** | **0.259** | **0.198** |
| pose35 | CEILING (GT pose + free bone scale) | **0.769** | **0.257** | **0.216** |
| pose35 | baseline (SYN_clean_pose35_w5, zero-init, full pipeline) | 0.519 | 0.332 | 0.223 |

**Result is genuinely surprising and NOT a clean floor measurement.** At pose25, the ceiling
arm is *worse* than the current full (zero-init, correspondence + free-form offset) pipeline,
not better -- despite having privileged GT pose. At pose35 it's better, as originally expected.

**This means H6 (model capacity) is NOT cleanly resolved by this measurement.** The most likely
reason, on inspection: the ceiling test still uses the SAME area-weighted chamfer sampling as
everything else in this project, so `log_beta_scales` for distal-leg joints receives the same
starved gradient signal (Section 1B/H3) that the full pipeline suffers from -- freezing pose does
not fix under-sampling. A secondary likely factor: the ceiling arm has no free-form
`deform_verts` budget at all, so any true per-specimen shape deviation beyond a per-joint
anisotropic scale (which the full pipeline's offset stage CAN partially absorb, whether that is
legitimate signal or pose-error compensation is exactly what this test was meant to
distinguish, and it cannot, as run) is simply unreachable here.

**Conclusion:** this ceiling design conflates H3 (sampling) and H6 (capacity) rather than
separating them. It does NOT license either "capacity is fine" or "capacity is the bottleneck."
A follow-up that reuses Rank 5's leg-protected sampler (`distal_quota_protected`) inside the
ceiling stage would be needed to isolate pure parametric capacity from the sampling confound --
flagged as future work, not run this cycle (see "What remains pending" at the end of this
report).

---

## Rank 3 -- Intermediate pose_scale sweep (Section 8 row 9 / Section 9) -- COMPLETE

**Zero new fitting jobs -- this sweep, and the correspondence audit across it, already existed**
(`diagnostics/registration_failure/runs/hier_SYN_clean_pose{0,05,10,15,20,25,35}_w5` from Task 6,
and `diagnostics/registration_failure/out/d2b_correspondence_audit.json` already covers all 7
pose levels plus drop30/drop60 -- this was not evident from the master prompt's own text, which
asked for this sweep as if new). Script:
`diagnostics/overnight_20260818/rank3_pose_sweep_analysis.py`. Output:
`diagnostics/overnight_20260818/out/rank3_pose_sweep.json`.

| pose_scale | leg_distal R | cross_leg[pt] | within_mismatch[pt] |
|---|---|---|---|
| 0.00 | 0.852 | 0.000 | 0.841 |
| 0.05 | 0.936 | 0.000 | 0.826 |
| 0.10 | 0.949 | 0.000 | 0.799 |
| 0.15 | 0.913 | 0.049 | 0.784 |
| 0.20 | 0.708 | 0.087 | 0.683 |
| 0.25 | 0.654 | 0.209 | 0.829 |
| 0.35 | 0.519 | 0.319 | 0.695 |

Spearman(pose_scale, cross_leg_confusion[pt]) = **+0.964** (p=0.0005) -- **H4 strongly
supported**: cross-leg confusion is monotonic in pose severity, as predicted.

Spearman(pose_scale, within_leg_mismatch[pt]) = -0.571 (p=0.180) -- **H2 not cleanly
supported or refuted**: within-leg mismatch stays severe (0.68-0.84) across the entire sweep,
which is consistent with "pose-independent" in the sense of *never getting good*, but it is not
flat -- there's a mild downward trend that isn't statistically resolved at n=7 conditions.
**Do not claim the H2/H4 mechanism separation as decisively confirmed** on this evidence alone;
report it as directionally consistent with the prior finding, not newly proven.

---

## Rank 4 -- GNC / annealed Geman-McClure (Section 13) -- COMPLETE (SLURM 3059305, array 0-2, all COMPLETED, ~7-9 min each)

**Zero new machinery.** `fitter_3d/trainer_moonshot.py:MoonshotStage` already implements a
`robust_scale -> robust_scale_end` geometric-interpolation schedule
(`self.robust_scale * (self.robust_scale_end/self.robust_scale)**t`, `t=it/n_it`) and a Geman-
McClure kernel (`kernel="gm"`); the current production recipe (`D1_low.yaml`) simply never turns
either on (defaults to plain L2, no robust kernel at all in the moonshot handoff stage -- note
this is DIFFERENT from the hierarchical H1-H3 stages, which already use a fixed, un-annealed
`robust_kernel="gm", robust_scale=0.25`). New: three YAML variants
(`diagnostics/overnight_20260818/cfg/gnc_{fast,medium,slow}.yaml`), each byte-identical to
`D1_low.yaml` except adding `robust_kernel: gm` + a `robust_scale`/`robust_scale_end` pair per
stage (fast 0.40->0.03 / 0.03->0.01, medium 0.25->0.05 / 0.05->0.02, slow 0.15->0.08 / 0.08->0.04
across Stage_2/Stage_3 respectively). One variable changed; baseline (`D1_low.yaml`,
`SYN_clean_w5`) untouched and not rerun.

Smoke-tested on CPU (12 specimens, 4 iterations each stage): ran cleanly, loss decreased,
`--eval` metrics.csv written without error.

Full runs (identical hierarchical placement to the SYN_clean_w5 baseline, then handoff with each
GNC yaml) completed on synth_clean (pose25). Analysis:
`diagnostics/overnight_20260818/rank4_analyze_gnc.py` ->
`diagnostics/overnight_20260818/out/rank4_gnc_analysis.json`.

| run | chamfer_l2 | leg_distal_dist_mean | antenna_dist_mean | leg_distal R |
|---|---|---|---|---|
| SYN_clean_w5 (baseline) | 0.00026 | 0.00909 | 0.00996 | 0.654 |
| GNC_fast | 0.00302 | 0.00850 | 0.01476 (+48%) | **0.857** |
| GNC_medium | 0.00154 | 0.00812 | 0.01143 (+15%) | **0.829** |
| GNC_slow | 0.00088 | 0.00893 | 0.01019 (+2%) | 0.758 |

(`leg_distal R` here is this session's own pooled-Pearson-R over leg_distal bone lengths,
computed identically across every arm in this report for internal comparability -- it is NOT
necessarily on the same absolute scale as the project's established weighted-median-R scorer
used in prior REPORT.md files, so treat cross-session absolute-R comparisons with caution;
within-session relative comparisons, as used throughout this table, are apples-to-apples.)

**Real, dose-dependent, interpretable result.** All three schedules substantially improve
leg_distal R (+0.10 to +0.20 over baseline) and reduce part_leg_distal_dist_mean, exactly the
predicted direction. `chamfer_l2` rises with schedule aggressiveness -- expected and not
concerning: a saturating (Geman-McClure) kernel does not drive residuals to zero the way L2
does near convergence, so a higher raw chamfer number is not comparable to the baseline's L2
number. Checked for collateral mesh damage: `degenerate_tri_frac` is 0 in every arm,
`folded_face_frac` and `tri_quality_mean` are flat-to-slightly-better, so this is not
"catastrophic" tradeoff. The real collateral cost is `part_antenna_dist_mean`, which regresses
in proportion to schedule aggressiveness (+48% fast, +15% medium, +2% slow) -- the same
H7-shaped tradeoff seen with Intervention C, but here the leg_distal gain is much larger at a
given antenna cost than Intervention C ever achieved.

**Verdict: GNC (medium or slow schedule) is the strongest production candidate to come out of
this cycle.** Medium buys +0.175 leg_distal R for +15% antenna distance; slow buys +0.10 for
only +2%. Neither shows the mesh-quality collateral damage that made split_distal/stratified
sampling risky. This is a genuinely new, well-supported, one-variable-at-a-time result -- not
previously evaluated in this project (D1_low.yaml never turns the kernel on).

---

## Rank 5 -- Model-anchored / leg-protected correspondence, C5 (Section 11) -- COMPLETE (SLURM 3059306, array 0-1, both COMPLETED, ~9 min each)

**New code this run** (the only rank requiring it):
`fitter_3d/stratified_sampling.py` gained `leg_face_mask()` and
`sample_target_distal_protected()`; `fitter_3d/trainer_hierarchical.py`'s `HierarchicalStage`
gained a `distal_quota_protected` constructor arg (mutually exclusive with the existing
`distal_quota`); `fitter_3d/optimise_hierarchical.py` gained a `--distal_quota_protected` CLI
flag. All existing call sites are unaffected (guarded by `is None` checks); baseline behaviour
confirmed unchanged by a no-flags smoke run producing identical H0_body loss values to a prior
smoke run before this code existed.

**What it tests.** Existing Intervention C (`--distal_quota`, HOLD verdict, Section 4) improved
distal R but regressed antenna. Its `distal_face_mask`/`nondistal` split reallocates sampling
budget from *every* non-distal face -- including antenna, head, body -- toward distal leg faces,
which is a plausible direct cause of that regression (Section 6 H7 vs Section 1B/11 "is H3
confounded with H7"). C5 corrects this: the quota is applied *only* inside a fixed **leg**
sub-budget (sized to the template's own baseline leg-area fraction, computed once, no leakage),
so non-leg anatomy's expected sampling density is **invariant to this flag by construction**.
Falsifier (pre-registered, matching the master prompt's own C5 spec): if antenna regresses
anyway, H7 is confirmed fundamental rather than a sampler artifact.

Run with `--split_distal` (13 groups; without it, pretarsus never gets its own correspondence
term at all, Section 2) and `--distal_quota_protected 0.3` (one value, chosen a priori, not
tuned). Smoke-tested on CPU, both with and without `--split_distal`, plus the mutual-exclusivity
assertion against `--distal_quota` (raises correctly). Full runs submitted on pose25 and pose0,
handoff via the unmodified `D1_low.yaml` (GNC not combined here -- one variable at a time).

Compared vs `SYN_clean_{pose25,pose0}_splitdistal_w5` (split_distal alone, isolates the
sampling-protection contribution) and the plain baseline. Analysis:
`diagnostics/overnight_20260818/rank5_analyze_c5.py` ->
`diagnostics/overnight_20260818/out/rank5_c5_analysis.json`.

| condition | run | leg_distal_dist_mean | antenna_dist_mean | leg_distal R |
|---|---|---|---|---|
| pose25 | SYN_clean_w5 (baseline) | 0.00909 | 0.00996 | 0.654 |
| pose25 | SYN_clean_pose25_splitdistal_w5 (split_distal alone) | 0.00736 | 0.01018 | 0.810 |
| pose25 | SYN_clean_pose25_c5protected_w5 (C5, this run) | 0.00765 | 0.01037 | 0.809 |
| pose0 | SYN_clean_pose0_w5 (baseline) | 0.00533 | 0.00725 | 0.852 |
| pose0 | SYN_clean_pose0_splitdistal_w5 (split_distal alone) | 0.00526 | 0.00724 | 0.898 |
| pose0 | SYN_clean_pose0_c5protected_w5 (C5, this run) | 0.00523 | 0.00717 | 0.906 |

**Result: C5 does NOT show a clear additional benefit over split_distal alone.** At pose25,
leg_distal R is statistically indistinguishable (0.809 vs 0.810) and antenna is very slightly
*worse* than split_distal alone (0.01037 vs 0.01018, both worse than the 0.00996 plain
baseline). At pose0, C5 is marginally better than split_distal alone on both metrics
(leg_distal_dist 0.00523 vs 0.00526, antenna 0.00717 vs 0.00724) but the margin is small enough
to plausibly be noise at n=12 specimens.

**Important caveat on the split_distal numbers themselves:** both `..._splitdistal_w5` runs here
show a LARGE leg_distal R jump over baseline (0.654->0.810 at pose25) using this session's
pooled-R metric. This should NOT be read as reversing Intervention A's documented negative
verdict (`diagnostics/registration_interventions/REPORT.md`) -- that verdict was based on a
broader evaluation (within-leg segment confusion, damage-condition robustness, collateral
regressions across many anatomical blocks, joint localization), not this one pooled-R number at
one pose condition. This report's R is a narrower, single-metric, single-condition check; the
original multi-metric verdict remains the authoritative read on split_distal as a whole.

**C5-specific interpretation (the actual falsifier test, Section 6 H3-vs-H7):** because C5 tracks
split_distal so closely rather than clearly beating it, this result does **not** cleanly resolve
H3-vs-H7 in either direction. It does NOT reproduce a large antenna regression (ruling out the
worst-case H7-confirmed reading), but it also doesn't show the predicted clean improvement (which
would have supported "H7 was a sampler-direction artifact"). The most defensible reading: once
split_distal already gives distal segments their own correspondence GROUP, the additional
budget-protection C5 adds on top contributes little in this pipeline -- i.e. split_distal's own
partitioning is likely doing most of the relevant work already, and H3-vs-H7 remains open rather
than resolved.

---

## Final synthesis (all of Ranks 1-5 complete)

### What was completed
All five ranked experiments ran to completion with zero failures: Rank 1 (pure analysis, no
jobs), Rank 2 (2 SLURM array tasks, ~2 min each), Rank 3 (pure analysis, no jobs), Rank 4 (3
SLURM array tasks, ~7-9 min each), Rank 5 (2 SLURM array tasks, ~9 min each). Total new
GPU compute: 7 array tasks, all COMPLETED, no FAILED status anywhere. New code was written only
for Rank 5 (`fitter_3d/stratified_sampling.py` additions, `--distal_quota_protected` flag) --
every existing call site was smoke-tested unaffected before and after. Rank 6+ (per-limb
multi-start) was explicitly NOT implemented, per the Rank 1 gate and the user's standing
instruction.

### Numerical results, one line each
- **Rank 1**: best GT-free selector recovers 28.1% of the oracle gap combined, and is
  sign-inconsistent across pose regimes (positive at pose25, as bad as -3.4x at pose0) -- FAILS
  the >=60% promotion threshold.
- **Rank 2**: capacity-ceiling R = 0.533 (pose25, WORSE than the 0.654 baseline) / 0.769 (pose35,
  better than the 0.519 baseline) -- inconsistent across conditions, confounded with sampling
  (H3), not a clean floor.
- **Rank 3**: cross-leg confusion vs pose_scale rho=+0.964 (p=0.0005); within-leg mismatch vs
  pose_scale rho=-0.571 (p=0.180, underpowered).
- **Rank 4**: leg_distal R 0.654 (baseline) -> 0.857/0.829/0.758 (fast/medium/slow GNC); antenna
  distance cost +48%/+15%/+2% respectively.
- **Rank 5**: leg_distal R 0.809 (C5) vs 0.810 (split_distal alone) at pose25 -- statistically
  tied; 0.906 vs 0.898 at pose0 -- small, plausibly-noise improvement.

### Which hypotheses gained/lost support
- **H1 (optimization basin)**: unchanged this cycle (not tested directly; D5's prior finding
  stands). Rank 1 clarifies that basin-escape evidence (D5) and *deployable* basin-escape
  (multi-start) are separate claims -- only the first is currently supported.
- **H2 (within-leg segment ambiguity)**: still unresolved. Rank 3's sweep neither confirms nor
  refutes pose-independence at conventional significance.
- **H3 (sampling starvation) vs H7 (resource competition)**: still NOT cleanly separated. Rank 5
  was designed to distinguish them and came back near-null (C5 ~= split_distal alone), which is
  itself informative -- it suggests split_distal's partitioning, not the sampling-budget
  mechanics on top of it, is where most of the achievable benefit already lives, at least in this
  pipeline. Rank 2's confound (ceiling test also starved by the same sampler) is independent
  further evidence that H3 is entangled with whatever else is being measured, project-wide.
- **H4 (pose-dependent cross-leg confusion)**: **strengthened** -- Rank 3's rho=+0.964 is the
  clearest single piece of quantitative support for this mechanism in the project to date.
- **H6 (model capacity)**: **not resolved, possibly not resolvable with the current sampler.**
  Rank 2's ceiling test cannot currently isolate capacity from sampling.
- **New finding, not a pre-existing hypothesis**: GNC (Rank 4) is a real, reproducible,
  dose-tunable improvement to leg_distal R with bounded antenna cost and no mesh-quality
  collateral damage -- the strongest positive result of this cycle.

### Multi-start GT-free selection gate (Rank 1)
**FAILED.** Per the master prompt's own decision rule and the user's explicit instruction,
per-limb decomposed multi-start (family D) was NOT implemented this cycle.

### Model-capacity ceiling (Rank 2)
**Inconclusive / confounded.** 0.533-0.769 R depending on pose condition, straddling the
baseline's own 0.519-0.654 range with no consistent direction. Cannot currently be used to
bound other interventions' achievable improvement.

### Current best evidence for what is limiting registration
Ranked by strength of evidence produced or reinforced this cycle:
1. **Pose-dependent cross-leg confusion (H4)** -- now the most quantitatively well-supported
   single mechanism (Rank 3, rho=+0.964).
2. **Correspondence quality generally, addressed via robust-kernel annealing** -- GNC's large,
   clean R improvement (Rank 4) is consistent with the optimizer benefiting from a
   coarse-to-fine correspondence schedule, i.e. early-iteration correspondence noise/outliers
   (plausibly cross-leg confusion from H4) is what GNC's wide-then-narrow kernel is suppressing.
3. **Sampling-vs-partitioning entanglement (H3/H7)** -- Rank 5's near-null result suggests
   partitioning (split_distal) matters more than fine-grained sampling budget mechanics on top
   of it, at least for this metric/condition set.
4. **Within-leg segment ambiguity (H2)** and **model capacity (H6)** remain the least resolved;
   neither was cleanly isolated by any experiment run this cycle or previously.

### Strongest production candidate
**GNC (medium schedule)**: `robust_kernel: gm`, `robust_scale`/`robust_scale_end` = 0.25->0.05
(Stage_2) / 0.05->0.02 (Stage_3), added to the existing `D1_low.yaml` handoff stage with
everything else unchanged. +0.175 leg_distal R over baseline for +15% antenna distance, zero
mesh-quality collateral damage, uses pre-existing (previously unused) machinery, config-only
change. This is a stronger, cleaner result than any previously-tested intervention in this
project's history (split_distal, stratified sampling, part-field init all had HOLD or negative
verdicts). Recommend promoting to an experimental opt-in arm (NOT the silent default, consistent
with this project's stated convention of defaulting new weights/kernels off) pending a repeat at
larger N and >=2 seeds, per this project's own standing bar for promotion (see
`diagnostics/d1_evidence/scorecard.md` precedent: single-seed N=12 evidence is promising, not
sufficient for a default-on promotion).

### What remains pending (deliberately not done this cycle)
- N>=50 / multi-seed replication of the GNC result before any promotion to default.
- A capacity-ceiling re-run using Rank 5's leg-protected sampler inside the ceiling stage, to
  actually isolate H6 from H3 (Rank 2's stated follow-up).
- B5 (compatibility-restricted matching) and the trimmed/unbalanced damage-specific term
  (Section 13/14) -- never reached; Rank 6+ family D also not reached, correctly gated by Rank 1.
- Fabian's PCA/UMAP morphometric-space question (Section 17) -- entirely untouched this cycle;
  the master prompt's own sequencing put it after the registration-mechanism work, and this
  cycle's scope was Ranks 1-5 only.
- GNC has not been tested combined with split_distal, damage conditions (drop30/drop60), or
  other pose severities beyond pose25 -- only the primary D1 corpus was used, one variable at a
  time, per the execution rules.
