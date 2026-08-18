# D1 N=50 / multi-seed promotion criteria (pre-registered before running)

Written before any N=50 job is submitted. Not adjusted after seeing results, per this
project's established discipline (see FINAL_REPORT.md "Methodology rules the investigation
paid for: pre-register the outcome and kill condition").

## Question

Does D1 (hierarchical -> moonshot, `D1_low.yaml` as confirmed unchanged by Task 2/Task 3)
graduate from "experimental-mainline candidate" (current status per
`fitter_3d/README.md` / `diagnostics/d1_evidence/scorecard.md`) to "shipped default"?

## Design

- N=50: full `bench50_clean` set (Task 1 used the first 10 of this same 50; those 10 are
  reused unchanged, the other 40 materialized fresh from the same source).
- 3 seeds: `--seed 0` (matches Task 1/2/3 exactly, so seed 0's N=50 numbers are also a
  10-specimen-subset consistency check against Task 1's N=10 numbers), `--seed 1`, `--seed 2`.
- Metrics: same instrument as Task 1 -- `diagnostics/moonshot/metrics.py`
  (`deformation_metrics`, `symmetry_metrics`) via `optimise_moonshot.py --eval`:
  `edge_logratio_absmean`, `deform_mag_mean/p95/max`, `folded_face_frac`, `dihedral_p99`,
  `fscore@0.01`, `chamfer_l2`.
- No stock-arm (Arm S) re-comparison here -- Task 1 already settled D1-vs-stock on every
  metric at N=10; this run is about D1's *own* stability at scale and across seeds, not a
  repeat of that A/B.

## Promotion bar (PROMOTE if ALL of the following hold; otherwise HOLD at experimental status)

1. **No catastrophic per-specimen failures.** Zero NaN/collapse/crash across all 50
   specimens x 3 seeds (150 runs). A catastrophic failure is: NaN in any metric, or
   `folded_face_frac > 0.10`, or `deform_mag_max` more than 10x the cross-seed median for
   that specimen (a single specimen going wild that the mean would hide).
2. **Cross-seed variance is small enough to trust a single future run.** For each primary
   metric (`edge_logratio_absmean`, `deform_mag_mean`), the coefficient of variation (std/mean)
   across the 3 seeds' per-specimen values, averaged over the 50 specimens, is < 15%. This is
   the operational answer to "can we trust one future seed=0 run on a new specimen set" --
   large seed-to-seed swings would mean no single run is representative.
3. **N=50 integrity numbers are not worse than the N=10 numbers by more than noise.**
   Compare each seed's 50-specimen means against Task 1's N=10 D1 means
   (`diagnostics/d1_evidence/scorecard.md`): `edge_logratio_absmean` and `deform_mag_mean`
   must not regress by more than 15% relative. (The N=10 set is a subset of the N=50 set, so
   this is testing whether the other 40 specimens behave consistently with the first 10, not
   an independent replication.)
4. **fscore/chamfer do not collapse** in the same sense as Task 2's gate: mean `fscore@0.01`
   across all seeds must not be worse than Task 1's N=10 D1 mean by more than 0.05 absolute,
   and mean `chamfer_l2` must not be worse by more than 50% relative.

If bar 1 fails on more than 2/150 runs (>1.3%), or bars 2-4 fail on either primary metric,
the verdict is **HOLD** (stay experimental-mainline candidate), with the specific failure
mode reported plainly -- not softened into "PROMOTE with caveats."

## Infra checks before launching (same as Task 1/2/3)

- `config.PLOT_RESULTS`: irrelevant to this chain -- confirmed in Task 2/3 that neither
  `optimise_hierarchical.py` nor `optimise_moonshot.py` reads this flag (grep, no hits). At
  N=50 this remains true (the flag's own logic is inside stock `fitter_3d/optimise.py`, not
  called here). Verified again immediately before launch, logged in the run's slurm output.
- Same hardware class (1x H100, `c23g`), same conda env (`pytorch3d`).
- Mesh set materialized read-only from `feature/registration_moonshot:diagnostics/moonshot/
  bench50_clean/*.obj` (git show, branch not checked out), gitignored, not committed --
  same pattern as `diagnostics/d1_evidence/mesh_set/`.

## Addendum: recipe correction before launch (disclosed deviation)

The plan above was drafted, and a smoke test run, against `D1_low.yaml` (Task 2/3's confirmed
unchanged recipe) before a separate scale_cap validation (`w_scale: 0.052`, 838-specimen
full-corpus A/B, `feature/investigation:diagnostics/SHIPPED_RECIPE.md` /
`diagnostics/morphometrics/ab_scale_cap/`) was surfaced. That work independently validated
`w_scale: 0.052` on both moonshot stages: non-inferior genus classification, integrity flat,
and a real anterior joint-scale outlier-tail reduction (head/mandible/antenna max ratio down
58-70%). Verified directly against `feature/investigation` before acting (not taken on faith):
`D1_PROD_SCALECAP.yaml`'s exact diff against the scale_cap-free baseline, the promotion
rationale in `SHIPPED_RECIPE.md`, and that `w_scale`/`scale_barrier` are already wired (weight
0.0 by default) in this branch's `fitter_3d/trainer_moonshot.py`.

**Corrected recipe: `diagnostics/moonshot/cfg/D1_low_scalecap.yaml`** -- `D1_low.yaml` plus
`w_scale: 0.052` on both `Stage_2_deform_coarse` and `Stage_3_deform_fine` (identical addition
to `D1_PROD_SCALECAP.yaml`'s diff against its own baseline). No other change from Task 2/3's
confirmed values (`edge_mode: shrink`, `w_offset: 5.0/2.0` unchanged).

Also checked before launch, per the three drift risks named alongside the scale_cap request:
- **Skip-if-exists / recipe fingerprinting**: this evidence's own sbatch driver
  (`run_n50_seeds.sbatch`) has no caching/skip logic at all -- every seed's hier + moonshot
  stages always run fresh via plain `mkdir -p` + direct invocation, so the fingerprinting bug
  described for `diagnostics/morphometrics/run_m1_fit_all.sh` (a different driver, not used
  here) does not apply.
- **SDF-stratification hand-unpickling bug** (`diagnostics/moonshot/sdf_stratify.py`, found on
  `feature/investigation`: raw `dd["weights"]`/`dd["v_template"]` from the pickle can be
  10229-long while every real fit is 10235-long post-symmetrisation): this evidence's metric
  suite is `diagnostics/moonshot/metrics.py` only, never `sdf_stratify.py`. Checked
  `to-ship`'s actual model file directly (`3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`):
  raw `dd["weights"]`/`dd["v_template"]` are already (10235, ...), matching
  `smal_model.smal_torch.SMAL()`'s output exactly -- no mismatch exists on this model file, so
  this bug class does not manifest here even where a similar hand-unpickling pattern exists
  (`diagnostics/d1_evidence/eval_stage_npz.py`, Task 1, already committed) -- verified, not
  assumed, so Task 1/2/3's already-committed evidence is unaffected.
- Corrected-recipe smoke test (N=50, single check, `--seed 0`, tiny its) re-run before the full
  array: clean, no NaN, loss decreasing, `w_scale` term present and logging (`scale=0.00000`
  through 400/1000 its -- expected, `scale_barrier` only engages once `log_beta_scales` exceeds
  the free band, consistent with `D1_PROD_SCALECAP.yaml`'s own note that it engages later in
  training).
