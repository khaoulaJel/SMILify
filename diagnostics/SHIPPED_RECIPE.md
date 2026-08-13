# Shipped recipe — D1_PROD

One page: what actually runs in production, what's in it, what was deliberately left out and
why. For the full evidence trail behind every number here, see `V2_CHANGELOG.md` (primary
source, chronological) and `khaoula_v2/REPORT_V2.md` (synthesis).

## Command

```bash
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3 \
  --results_dir $R/${t}_hier

python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
  --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml \
  --init_from $R/${t}_hier/H2_joint.npz \
  --results_dir $R/$t
```

Driver: `diagnostics/morphometrics/run_m1_fit_all.sh` (recipe-fingerprinted skip-if-exists,
`FORCE_REFIT=1` to override). Stages: `H0_body -> H1_legs -> H2_joint` (skeleton placement,
`H3_deform` skipped via `--skip_h3` — its npz is never consumed by the moonshot handoff),
then moonshot `Stage_2_deform_coarse -> Stage_3_deform_fine`. Final artefact:
`Stage_3_deform_fine.npz`.

## Model

`3D_model_prep/OmniAnt_25PCs_joint_limited.pkl` — carries authored `joint_limits`
(PR #98/#97, merged to `master`); same file, byte-identical (md5 `08b69daf119a6eda0cd699b8894ee7f8`)
on `master`, `upstream/feature/registration_moonshot`, and this branch's working tree —
verified, not assumed, 2026-08-13.

## Weights — `diagnostics/moonshot/cfg/D1_PROD.yaml`

| weight | Stage_2 | Stage_3 | status |
|---|---|---|---|
| `w_chamfer` | 1.0 | 0.5 | unchanged from M1/E8 baseline |
| `w_edge` | 0.8 | 0.2 | `edge_mode: shrink` — `rest` tested combined with this recipe, REJECTED (correctness 6.69%→1.52%) |
| `w_normal` | 0.005 | 0.002 | unchanged |
| `w_laplacian` | 0.01 | 0.001 | unchanged |
| `w_offset` | 5.0 | 2.0 | grid-optimum (T0.1 full sweep: 0.2/0.08, 2/0.8, 5/2, 10/4, 20/8, 200/200 — 5.0/2.0 wins, non-monotonic as predicted) |
| `w_sym` | 0.5 | 0.5 | unchanged |
| `w_beta_prior` | 0.0 | 0.0 | stays off |
| `w_midline` | 2.0 | 2.0 | unchanged |
| `w_limit` | 0.006 | 0.006 | authored joint-limit rotation prior, present in both stages — this is what `D1_low.yaml` got wrong (silently absent in the handoff stage) |
| `scheme` | `all` | `all` | pose never frozen during deform |
| sampling | symmetric area-weighted, both meshes | — | confirmed unconditional in `trainer_moonshot.py`, not a config flag |

`args.results_dir` in the YAML is inert — `run_m1_fit_all.sh` always passes `--results_dir`
explicitly per specimen/chunk.

## What is NOT in this recipe, and why

| item | why not shipped |
|---|---|
| `w_scale` (scale_cap, 0.052) | **SHIPPED 2026-08-13**, promoted into `D1_PROD.yaml` after passing the pre-registered full-corpus A/B (execution plan §4, 838 specimens): genus classification non-inferior (11.4% vs 11.7% LOO-1NN, within the null's own noise — **no lift is claimed for genus**, it is a wash), integrity not worse (<1% shift on every term), anterior joint-scale outlier tail measurably reduced (max ratio down 58–70% head/mandible/antenna), replicating `bench50_clean`'s original 22-specimen finding at full power. Full numbers: `diagnostics/EXECUTION_PLAN.md` §5, `diagnostics/morphometrics/ab_scale_cap/`. |

**Adoption is two independent, currently-aligned conclusions, not one.** (1) `scale_cap` ships as
part of the *default* genus-optimized recipe because genus didn't drop and integrity didn't
worsen — a non-inferiority argument. (2) `scale_cap` is independently validated as a
*mesh-validity* mechanism on its own original design target (anterior joint-scale pathology),
regardless of what genus classification does. These currently point the same way, but they
are not the same claim, and a future change that touches the genus-optimized recipe (a new
`w_offset`, a different prior, anything in §2.1's P0/P1 table) **must re-check pathology
separately before dropping `w_scale`** — non-inferiority on genus was necessary for conclusion
(1) but is not what justifies (2), and a recipe change that stays genus-neutral could still
reintroduce the anterior-scale explosion `scale_cap` exists to prevent.
| `edge_mode: rest` | Tested combined with the `w_offset` 5.0/2.0 winner: correctness collapsed 6.69%→1.52%, median error roughly doubled. Interacts negatively with the offset penalty, not merely redundant. Rejected, not reopened. |
| `w_trans` (trans_cap) | Implemented (`fitter_3d/joint_limits.py`), never meaningfully engaged in any test run to date (`betas_trans` barrier ≈0 throughout — D1's other regularizers already keep translation in-band on every corpus tested). No prior calibration reference exists anywhere in the branch. Deprioritized, not a validated negative — just untested for real. |
| `w_geocoh` (geodesic motion coherence) | Added as an optimizer term, swept 0.1/1.0/10.0 — worse than baseline at every weight, monotonically, no interior optimum. Closed, decisive negative. |
| Dense correspondence as a shipped mechanism (SDF partitioning, HKS, raw/refined DINO, arc-length ordering) | All closed on evidence — see `FINAL_REPORT.md` §3 and `khaoula_v2/REPORT_V2.md` §2. None beat the fitted pipeline's own 4.41% within-part median error; several were structurally capped regardless of tuning. SDF's only shipped role is diagnostic stratification (`sdf_stratify.py`), not a correspondence mechanism. |

## Honest current-best numbers (E6, full `synth_clean`)

6.69% exact-correct vertices, 3.21% median error. Dense per-vertex correspondence is not the
success metric (~83% of remaining error is within the correct anatomical part, structurally,
across six independent mechanism families tested). The metric that matters: genus
classification currently sits at ~6.1x chance (lot-blind, original corpus), replicated at
~3.8x on an independent corpus.

## Status of the pending A/B (§4)

**Not run yet, but no longer blocked on data access.** `/media/fabi/Data/...` (Fabian's own
machine) is still unreachable from this HPC cluster, but the same corpus is mirrored on the
shared UM6P_2026 Google Drive and was synced here 2026-08-13 (757/757 worker + 81/81
ALL_ANTS_CLEAN, verified against `measure.py`'s own filename-parsing convention, not just a
count match). `run_m1_fit_all.sh` and both arm configs are ready to run from this cluster.
What's left is a compute-time/scheduling decision (likely several GPU-hours per arm), not a
data problem — see `diagnostics/morphometrics/ab_scale_cap/SCHEMA.md`'s Status section for
the exact numbers and the pre-registered power check (§4.4a), now computed on the real
corpus (92 genera / 678 specimens at `min_n=3`, vs. `bench50_clean`'s 9/22).
