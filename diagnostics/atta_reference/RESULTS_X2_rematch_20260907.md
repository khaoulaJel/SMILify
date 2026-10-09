# X2 rematch — determinism check, fresh bundle, headline-result impact

**Date: 2026-09-07.** Follow-up to `diagnostics/anterior_mechanism/RESULTS_X2_step0_reporter_validation.md`
(numpy vs Blender CSV gap). Blender is not available in this environment — this covers only the
numpy/fitting half.

## Task 1 — determinism: **NOT deterministic**

Re-ran the exact ARM A recipe (`submit_atta20_arm_a_master.sbatch`'s recipe, same worktree
`/hpcwork/nao48500/atta20_master` @ `b5bf9565`, same `cfg_arm_a_master.yaml` stages/weights, same
20 meshes at `/hpcwork/nao48500/atta20/{01..20}.obj`) via SLURM job **3790072** (account
`rwth2151`, partition `c25g`, COMPLETED, exit 0:0, 5 min runtime), output at
`diagnostics/atta_reference/runs/ARM_A_master_rerun_20260907/Stage_3_deform_fine.npz`.

Compared `verts` (20, 10235, 3) against the original
`runs/ARM_A_master/Stage_3_deform_fine.npz`:

| check | result |
|---|---|
| `np.array_equal` | False |
| `np.allclose(atol=1e-6)` | False |
| `np.allclose(atol=1e-3)` | False |
| max abs diff | 0.0565 model units |
| mean abs diff | 0.00275 model units |

**Verdict: the master fit path is NOT deterministic.** This matches prior project memory
(Z4/Z5: "Fitter is NOT deterministic: ~0.01 run-to-run on gen@20") — likely nondeterministic GPU
reduction order (chamfer/nearest-neighbour ops) plus no fixed random seed in `optimise.py`, not an
environment/library drift since August (same worktree, same commit, same env both times). Labels,
keys, and shapes matched exactly; only the optimized values differ.

**Consequence for Task 2:** "the rerun matched the original, so either is interchangeable" does not
hold. The rerun is used anyway, on a different justification: it is *fresh* — importable into
Blender with zero delay — which is the actual property Step 0 needed, not bit-identity with the
Aug/Sep run.

## Task 2 — fresh bundle

`diagnostics/atta_reference/blender_bundle_rematch_20260907/`:
- `ATTA20_ARM_A_rematch.npz` (job 3790072's `Stage_3_deform_fine.npz`, produced 10:31 today)
- `OmniAnt_25PCs_joint_limited.pkl` (sha256-verified byte-identical to the original bundle's copy)
- `smil_importer.zip`, `ant_body_lengths.csv` (unchanged from the original bundle)
- `README_REMATCH.md` — exact Blender steps, ending with: export Joint Distances and save as
  `diagnostics/atta_reference/blender_export_rematch_20260907/OmniAnt_25PCs_joint_limited_joint_distances.csv`,
  all in one sitting, no save/reopen in between.

This is the user's next action — nothing further can be done from this environment (no Blender
binary or module here).

## Task 3 — does this threaten `REPORT_ATTA_HEADWIDTH.md`'s headline result?

**No.** Two independent points:

1. **The headline comparison never used the numpy regressor.** `REPORT_ATTA_HEADWIDTH.md`'s
   head-width table (median 1.77% error, allometric exponents 1.189 vs 1.235 and 0.380 vs 0.395)
   compares `results/head_width_results.csv`'s `replicated_head_width_mm` column — read straight
   from the Blender-exported `SMPL_Object_joint_distances.csv` — against
   `measurements/ant_body_lengths_and_estimated_head_widths.csv`, Fabian's **physical** reference
   measurements (mm, from real specimens). The numpy `J_regressor @ verts` recomputation used in
   Step 0 never enters that comparison; the whole-skeleton check (55 joints, median 1.05%) is the
   same Blender-CSV-vs-physical-reference comparison, not Blender-CSV-vs-numpy. The Step 0 gap is
   an internal self-consistency check the report itself never relied on.

2. **The same kind of gap is a standing pattern, not a one-off, and this report's own result
   already absorbs it.** The original bundle's npz was written 2026-09-06 00:01, the CSV that
   fed the report wasn't exported until 12:47 the same day — a 12h45m gap. Today's determinism
   check shows a *fresh, same-config, same-commit* rerun differs from the original run by mean
   abs 0.0028 / max 0.057 model units purely from run-to-run optimizer noise — the same order of
   magnitude Step 0 attributed to the npz-vs-CSV mismatch. In other words, this gap has likely
   always been present to some degree (any two fitter invocations differ), and
   `REPORT_ATTA_HEADWIDTH.md`'s reported accuracy (median 1.77% on head width, 1.05% whole-skeleton)
   is the number **with** that noise already baked in, not a number that noise would additionally
   erode. Nothing here is new information that retracts the report; it explains its size, and
   argues the report's stated numbers are if anything realistic-or-conservative, not optimistic.

**Explicit answer: NO, nothing here retracts or casts doubt on `REPORT_ATTA_HEADWIDTH.md`'s
headline result.** Its comparison was always against the physical reference CSV, never against the
numpy recomputation, and the magnitude of run-to-run fitter noise found here is consistent with
(not larger than) what the report's own accuracy numbers already reflect.
