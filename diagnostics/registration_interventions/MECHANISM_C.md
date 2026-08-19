# Intervention C — stratified distal target sampling: pre-registered design

Written BEFORE any fitting job is submitted. Values below are fixed and will not be adjusted
after seeing results.

## Research question

Does giving the optimizer more actual DISTAL TARGET EVIDENCE (more of the existing `n_sample`
budget landing on tibia+tarsus+pretarsus faces), with the objective, correspondence mechanism,
pose model, deformation machinery, optimizer, and `--split_distal` all held OFF/unchanged, reduce
the wrong-basin registration failure Task 6/D5 diagnosed?

## Exact mechanism changed, and what does not change

**Changed**: ONE thing — how TARGET points are drawn each sampling/reassignment step in
`HierarchicalStage.run()` (`trainer_hierarchical.py`). Baseline draws `n_sample=8000` points from
the whole target mesh, area-weighted (`pytorch3d.ops.sample_points_from_meshes`, one call). The
intervention draws the SAME `n_sample=8000` total, but from two pools: a DISTAL pool (faces whose
dominant skinning joint is `ti`/`ta`/`pt` on ANY leg, pooled across all 6 legs) at a fixed target
fraction (the "quota"), and a NON-distal pool (everything else) for the remainder — each pool
internally still area-weighted, so within each pool sampling is exactly as unbiased as baseline.

**Unchanged, verified in code and by the validation script before any GPU job**:
- Objective: same `_partitioned_chamfer`, same per-GROUP structure. `--split_distal` stays OFF —
  legs remain single (not proximal/distal-split) groups, exactly baseline's own grouping.
- Correspondence/partition mechanism: `TargetPartition` is untouched; it still assigns the
  (now-stratified) drawn points to groups by nearest-fitted-vertex exactly as today.
- SOURCE (fitted-mesh) sampling: untouched, still plain `sample_points_from_meshes(mesh, n_sample)`
  — only the TARGET draw is stratified. This isolates "more distal target evidence" from any
  change to how the source is compared.
- Pose model, deformation penalty, optimizer, learning rates, joint limits, `reassign_every`,
  stage schedule, iteration counts: byte-identical to the existing baseline recipe.
- Total `n_sample` per draw: identical (verified exactly, not approximately, in the validation
  script).

## Leakage analysis

Which faces are "distal" is determined ENTIRELY from the TEMPLATE's own skinning weights
(`weights.argmax(1)` -> dominant joint -> `ti`/`ta`/`pt` if the joint name says so) — the same
static, scan-independent fact `face_group_by_segment` (Task 6's own D2a/D2b probes) already uses,
not learned, not GT, not fitted. This is valid ONLY because the topology-preserving synthetic
corpora (`synth_clean_pose0`, `synth_clean`/pose25) give every specimen's target mesh the exact
same face indexing as the template (no remesh) — confirmed already in Task 6's D2a
(`assert f_idx.shape == faces.shape`). **This does NOT extend to `drop30`/`drop60`**, which remesh
(different face count per specimen, confirmed directly when `probe_A_d2a_damage.py` hit this same
wall earlier this task). Per the explicit instruction, Phase C1 (this experiment) covers ONLY
pose0/pose25; Phase C2 (a damage-compatible design) is deferred and NOT attempted with a
GT-dependent shortcut.

## Quotas (pre-registered, not tuned to results)

Natural (unmodified, baseline) reference point, measured directly from Task 6's own D2a output
(`diagnostics/registration_failure/out/d2a_sample_starvation.json`, `by_segment_pooled_clean_pose25`):
median samples per leg pool ti=29, ta=24, pt=2, summed against a per-leg total (co+tr+fe+ti+ta+pt)
median of 487 -> **natural distal (ti+ta+pt) fraction ~= 11.3%** of a leg's own sample share. (This
is an empirical MEDIAN-SAMPLE fraction, not the template's raw AREA fraction quoted elsewhere as
~2.4% for tarsus+pretarsus alone — the two numbers measure different things and are not expected
to match; the empirical one is the correct reference for calibrating a SAMPLE quota.)

Tested quotas, chosen to span below/near/above that natural point, exactly as specified by the
mission: **baseline (natural, ~11.3%, no intervention) / 5% / 10% / 20%**. The optional 30% arm is
DROPPED from this pass to control compute/scope given the number of other open items in this
report; this is a scope reduction decided BEFORE running anything, not a post-hoc exclusion, and
is stated here for the record.

## Validation performed BEFORE any SLURM job (Requirement 5)

`diagnostics/registration_interventions/validate_stratified_sampling.py`, CPU-only, no GPU, no
fitting. Checks, per quota, on real target meshes (pose0 and pose25 corpora): (1) total sample
count == 8000 exactly; (2) achieved distal fraction matches the requested quota to within
rounding; (3) non-distal pool remains a valid area-weighted sample (spot-checked against plain
`sample_points_from_meshes` restricted to the same face set); (4) no ground truth, fitted
correspondence, or learned model is referenced anywhere in the sampling code path (grep-level
check + code review, recorded here); (5) per-leg balance of the pooled distal draw (the design
draws distal points as ONE pool across all 6 legs rather than 6 separately-quota'd pools, justified
by the template's own <0.3% bilateral/radial symmetry, Task 6 probe 13 — verified empirically here
that each leg receives a comparable share, not merely assumed); (6) baseline (quota=None) path is
reproducibly IDENTICAL in distribution to the unmodified `sample_points_from_meshes` call (same
RNG usage pattern), i.e. the new code path is provably a no-op when not requested.

Results of this validation are logged in `out/validate_stratified_sampling.log` and summarized in
`REPORT.md` §C before any fitting job was submitted.

## Validation results (ALL CHECKS PASSED, before any fitting job submitted)

- 1632/20466 template faces (7.97%) marked distal by the majority-vote rule.
- Per-leg distal-area share: front (l1) 18.25%, middle (l2) 13.23%, hind (l3) 18.52%, each side
  matching its mirror to within numerical precision (confirms bilateral symmetry, consistent with
  Task 6 probe 13's <0.3% figure) but front/middle/hind differ by up to 3.43 percentage points
  from equal-sixths -- pooling all 6 legs into one draw is still justified (no leg is starved
  relative to the others by more than this), but this asymmetry is recorded, not hidden.
- Total sample count: exactly 8000 for every quota tested (0.05/0.10/0.20), both corpora.
- Achieved distal fraction: exactly matches the requested quota (400/800/1600 of 8000).
- Baseline path (`distal_quota=None`): confirmed at the source level to call plain
  `sample_points_from_meshes` with no stratified-sampling code involved.
- No ground truth, fitted correspondence, or learned model referenced in
  `fitter_3d/stratified_sampling.py` (confirmed: the module imports only `torch`).
