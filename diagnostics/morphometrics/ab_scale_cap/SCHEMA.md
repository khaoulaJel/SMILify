# AB_scale_cap_vs_D1 — pre-registered schema + handoff runbook

Per execution plan §4. **Written before any arm has run.**

## Precondition (§4.0) — UPDATE 2026-08-13: corpus is reachable after all

`/media/fabi/Data/...` (Fabian's own machine) is still unreachable from this HPC cluster,
confirmed again 2026-08-13. But the user pointed out the same corpus is also mirrored on the
shared UM6P_2026 Google Drive, under a shortcut-target folder not discoverable from the
folder URLs alone (needed `root_folder_id` navigation via `rclone`, configured remote
`gdrive:`):

- `DATA/mesh_registration/ALL_ANTS_CLEAN` — 81 `.obj` files, exact match to the expected
  count, AND the exact filename convention `measure.py`'s `parse_taxonomy()` was written for
  (20 numeric-stem files handled by its `stem[0].isdigit()` special case, 61 genus-named
  files like `acanthomyrmex.obj`/`aenictus-binghamii.obj` — literally the example filenames
  in that function's own docstring). This is strong confirmation it's the same corpus, not
  just a count coincidence.
- `DATA/mesh_registration/custom_processing/antscan_proofread_castes/worker_ALT` — 757
  `.obj` files, exact match. Named `worker_ALT`, not `worker` (this script's original path) —
  confirmed by the user as the correct corpus despite the name difference.

Total size: 3.1 GiB + 165 MiB ≈ 3.3 GiB — synced to
`/p/scratch/cias-7/jellal1/SMILify_DATA/{ALL_ANTS_CLEAN,custom_processing/antscan_proofread_castes/worker_ALT}`
via `rclone copy` 2026-08-13, verified 757/757 and 81/81 files present locally, 0 errors in
the transfer log. `run_m1_fit_all.sh`'s `W`/`CLEAN`/`CONDA_SH`/`REPO_DIR` are now
environment-variable-overridable (see its own header comment) so it can run from this
cluster without touching Fabian's machine-specific defaults.

**This document's "BLOCKED" framing below is now only partially true — data access is no
longer the blocker. What has NOT changed: neither arm has actually been fit.** See "Status"
at the bottom for exactly what's ready vs. still to do.

Original note, kept for context: **do not substitute `bench50_clean` or any smaller corpus
and call it this result** — a result from the wrong corpus is a different,
already-run, already-known-to-be-underpowered experiment (see `INVENTORY_V2.md` row
T0.4e: 1 discordant specimen out of 22, no statistical power). That concern is now resolved
in the sense that the RIGHT corpus is reachable; it still applies as a standing rule for any
future check.

## Schema

```yaml
name: AB_scale_cap_vs_D1
primary_endpoint: genus classification enrichment vs chance (lot-blind), same protocol as
  diagnostics/morphometrics/REPORT_MORPHOMETRICS.md (analyse.py's loo_1nn + perm_test)
secondary: integrity metrics (deform_mag, edge_logratio, midline_dev, folded_face_frac),
  stratified by SDF/skinning-weight anatomical region (diagnostics/moonshot/sdf_stratify.py)
tertiary: E6 correct% / median_err on full synth_clean only
arms:
  A: diagnostics/moonshot/cfg/D1_PROD.yaml        (tag prefix AB_A_, no scale_cap)
  B: diagnostics/moonshot/cfg/D1_PROD_SCALECAP.yaml (tag prefix AB_B_, w_scale: 0.052)
corpora:
  - all worker chunks (757, chunked by run_m1_fit_all.sh's CHUNK=64 default -> ~12 chunks)
  - ALL_ANTS_CLEAN (81)
model: 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
fit_chain: fitter_3d.optimise_hierarchical (--skip_h3) -> fitter_3d.optimise_moonshot
force_refit: true   # both arms MUST use FORCE_REFIT=1 -- see run_m1_fit_all.sh's own
                     # recipe-fingerprint skip logic; force_refit here is about never
                     # accidentally reusing arm A's fits for arm B or vice versa, not about
                     # the fingerprint mechanism being untrustworthy
seeds: N/A for deterministic optimization; record torch/CUDA versions in the run log regardless
exclusion: none a priori; the fittability gate is a column, not a filter (existing M1 design)
```

## Power check (§4.4a) — DONE, 2026-08-13, before either arm has run

Computed directly from the synced corpus's filenames, reusing `measure.parse_taxonomy()` and
`analyse.accession_lot()` unchanged (no fit needed for genus/lot counts — genus is parseable
from filename alone):

| `min_n` | genera clearing the bar | specimens in the classification test | accession lots (worker only) |
|---|---|---|---|
| 2 | 137 | 768 (of 838 total, 20 filename-unparseable `ALL_ANTS_CLEAN` numeric stems excluded) | 37 |
| 3 (analyse.py default) | 92 | 678 | 37 |
| 5 (genus_table.py default) | 49 | 537 | 37 |

Compare to the actual T0.4e check this replaces: `bench50_clean` at `min_n=2` had 9 genera,
22 specimens — the full corpus at the SAME `min_n=2` threshold has 137 genera, 768
specimens: ~35x the specimen count, ~15x the genus count.

**Minimum detectable effect**, two-proportion normal-approximation power calculation
(α=0.05 two-sided, power=0.80), using `bench50_clean`'s own baseline accuracy (22.7%, 5/22)
as the only real prior estimate of `p1` available:

| corpus | n | min. detectable accuracy | delta | lift needed to reach significance |
|---|---|---|---|---|
| `bench50_clean` (actual T0.4e check) | 22 | 63.5% | +40.8pp | 2.80x |
| full corpus, `min_n=5` | 537 | 30.3% | +7.6pp | 1.33x |
| full corpus, `min_n=3` | 678 | 29.4% | +6.7pp | 1.30x |
| full corpus, `min_n=2` | 768 | 29.0% | +6.3pp | 1.28x |

**Caveat, stated plainly**: this is the standard independent-two-proportion approximation,
used as a simple, conservative stand-in for the actual test (`analyse.py`'s paired
leave-one-accession-lot-out 1-NN + permutation null, evaluated on the SAME specimens under
both arms) — not a simulation of that exact test's null distribution. It is directionally
right and order-of-magnitude correct, not a precise number for that specific paired test.

**Reading, stated before either arm has run**: `bench50_clean` could only have detected an
effect at or above a 2.80x lift — far larger than the 1.20x (22.7%→27.3%) it actually
observed, which is exactly why that result was correctly called "not statistically
distinguishable from noise" in `INVENTORY_V2.md`. The full corpus's minimum detectable lift
(~1.28-1.33x) is close to, and possibly still slightly above, the 1.20x `bench50_clean`
suggested — meaning the full-corpus A/B is meaningfully more powered, but is not guaranteed
to reach significance even if the true effect matches `bench50_clean`'s point estimate
exactly. State this now so a flat full-corpus result isn't a surprise requiring a post-hoc
excuse: **the full corpus is the right test to run, but a null result on it would not
necessarily mean scale_cap has zero effect at the observed bench50 magnitude — it would mean
that magnitude, if real, is smaller than this design's detection floor.**

## Exact commands, in order

```bash
source /home/fabi/mambaforge/etc/profile.d/conda.sh   # or wherever conda lives on the
conda activate pytorch3d                                # machine actually running this
cd /home/fabi/dev/SMILify                                # match run_m1_fit_all.sh's own cd

# 1. Arm A -- baseline, no scale_cap
TAG_PREFIX=AB_A_ YAML_SRC=diagnostics/moonshot/cfg/D1_PROD.yaml FORCE_REFIT=1 \
  bash diagnostics/morphometrics/run_m1_fit_all.sh

# 2. Arm B -- candidate, scale_cap
TAG_PREFIX=AB_B_ YAML_SRC=diagnostics/moonshot/cfg/D1_PROD_SCALECAP.yaml FORCE_REFIT=1 \
  bash diagnostics/morphometrics/run_m1_fit_all.sh

# 3. Per-arm morphometrics. CONFIRMED by reading analyse.py directly (not assumed):
#    --worker_runs with no value defaults to auto-discovering ONLY the literal "MORPH_W*"
#    prefix (analyse.py's own default-resolution block), so AB_A_W*/AB_B_W* must be passed
#    explicitly -- they will NOT be picked up automatically. N below = however many chunks
#    run_m1_fit_all.sh's CHUNK=64 default produced for 757 workers (~12; check
#    diagnostics/morphometrics/stage/nchunks.txt after step 1/2 above for the exact count).
python -u diagnostics/morphometrics/analyse.py \
  --worker_runs AB_A_W0 AB_A_W1 AB_A_W2 ... AB_A_W$((N-1)) --clean_run AB_A_CLEAN --min_n 3 \
  > diagnostics/morphometrics/ab_scale_cap/analyse_A.log
python -u diagnostics/morphometrics/analyse.py \
  --worker_runs AB_B_W0 AB_B_W1 AB_B_W2 ... AB_B_W$((N-1)) --clean_run AB_B_CLEAN --min_n 3 \
  > diagnostics/morphometrics/ab_scale_cap/analyse_B.log

# genus_table.py takes the identical --worker_runs/--clean_run pattern (confirmed by
# reading it directly), same explicit-list requirement as analyse.py above
python -u diagnostics/morphometrics/genus_table.py \
  --worker_runs AB_A_W0 AB_A_W1 AB_A_W2 ... AB_A_W$((N-1)) --clean_run AB_A_CLEAN --min_n 5
python -u diagnostics/morphometrics/genus_table.py \
  --worker_runs AB_B_W0 AB_B_W1 AB_B_W2 ... AB_B_W$((N-1)) --clean_run AB_B_CLEAN --min_n 5

# 4. Integrity + SDF-region stratification, per arm
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
python -m diagnostics.moonshot.sdf_stratify --run_dir diagnostics/moonshot/runs/AB_A_W0 \
  --out diagnostics/morphometrics/ab_scale_cap/integrity_A_W0.csv
# ... repeat per chunk/arm, or extend sdf_stratify.py to loop over a tag-prefix glob before
#     the real run (cheap to add, not added here since there is no corpus to test it against)

# 5. Tertiary: E6 on full synth_clean, same protocol as diagnostics/moonshot/
#    score_synth_roundtrip.py, one run per arm's config (synth_clean does not need the
#    unreachable corpus -- this piece COULD be run from this HPC cluster right now; not run
#    in this session because it answers the tertiary endpoint only, and running it before
#    the primary/secondary endpoints are even reachable would not change what ships)
```

## Required diagnostic outputs

| Output | Path pattern | Content |
|---|---|---|
| Fit npz | `diagnostics/moonshot/runs/AB_{A,B}_*/Stage_3_deform_fine.npz` | verts, labels, params |
| Integrity CSV | `ab_scale_cap/integrity_{A,B}.csv` | per-specimen metrics from `metrics.py`, plus SDF/skinning-weight region column (`sdf_stratify.py`) |
| Genus table | `ab_scale_cap/genus_table_{A,B}.csv` | same columns as `REPORT_MORPHOMETRICS.md`'s existing table |
| E6 summary | `ab_scale_cap/e6_synth_clean.json` | correct%, median, p90, within/between split, per arm |
| Config snapshot | already on disk: `diagnostics/moonshot/cfg/D1_PROD.yaml`, `D1_PROD_SCALECAP.yaml` | exact weights, no copy needed |
| Run log | SLURM logs (or equivalent) + `git rev-parse HEAD` | reproducibility |

## Pre-registered decision rule (§4.4) — restated here so it travels with the schema

Ship `scale_cap` into `D1_PROD.yaml` (promote `D1_PROD_SCALECAP.yaml`'s `w_scale: 0.052` into
the default) only if, on the FULL corpus:
- Primary metric (genus enrichment) is non-inferior to baseline, **and**
- Integrity is not significantly worse on the pre-specified tests, **and**
- Anterior scale pathology is measurably reduced (not just on `bench50_clean` — INVENTORY_V2.md
  row T0.4d already showed that; this needs to hold on the real production corpus).

If genus drops: keep `D1_PROD.yaml` without scale_cap; document the magnitude-vs-taxonomy
mismatch explicitly, don't treat it as a wash.
If genus rises: promote `D1_PROD_SCALECAP.yaml`'s weights into `D1_PROD.yaml`; update
`SHIPPED_RECIPE.md` and this file's own status line below.

## Status

**READY TO RUN, as of 2026-08-13. Not yet run.** Corpus access is no longer the blocker (see
above — synced locally, verified 757/757 + 81/81, byte counts and filename convention both
match). `run_m1_fit_all.sh` is ready (recipe-fingerprinted, `TAG_PREFIX`/`YAML_SRC`/
`FORCE_REFIT`/`CONDA_SH`/`REPO_DIR`/`W`/`CLEAN` all wired for exactly this two-arm use, from
this cluster). `D1_PROD.yaml` and `D1_PROD_SCALECAP.yaml` are both ready and diffed only by
`w_scale`. The power check (§4.4a) is done. **What's left is purely a compute-time decision,
not a data-access one**: fitting 838 specimens through the full hierarchical+moonshot chain,
twice (once per arm) — this script runs both worker chunks and the clean corpus in parallel
across 2 GPUs per invocation; on the smoke-test's per-specimen timing (3 synth_clean meshes,
hierarchical ~3.5 min + moonshot ~2 min combined, single GPU, no chunking parallelism), a
rough order-of-magnitude estimate for 838 specimens split across ~13 chunks of 64 is several
GPU-hours per arm, not minutes — this has NOT been estimated precisely and should be treated
as a real compute-budget decision on a shared cluster, not launched by default. Needs an
explicit go-ahead (and, if run via SLURM rather than an interactive session, the target
partition/account) before actually starting either arm.
