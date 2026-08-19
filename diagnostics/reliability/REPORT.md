# Task 2 — specimen-aware registration reliability for morphometric inference

## Verdict: HOLD — no scheme beats `equal` weighting with evidence that clears this project's
own pre-registered bar, at n=24 synthetic specimens. Keep `hard50` in production for now; the
`continuous_*` schemes show a real, mostly-positive directional trend on absolute error but it is
not distinguishable from noise at this sample size, and the one cell that *is* statistically
distinguishable is a regression, not an improvement.

This is a closed, confirmed non-finding at the current sample size, not a failed experiment — per
this project's established discipline (`FINAL_REPORT.md`, `offset_sweep_evidence/scorecard.md`),
that is reported as-is rather than dressed up.

## 1. Does spatial information exist in the registration-quality diagnostics?

Yes, verified by reading the code, not assumed. `measure.mesh_quality` (deform-magnitude /
edge-distortion / normal-roughness) computes each quantity per-vertex, per-edge, or per-face and
only reduces to one global scalar per specimen at the very last step. `part_quality`
(`diagnostics/absolute_scale/measure.py`, added for this task) is the same three quantities
reduced per anatomical region instead — using the region scheme (`group_of`) this codebase
already uses to group measurement columns, not a new part definition. So: part-specific
reliability is derivable, and was derived, rather than invented or skipped in favour of a
global-only score.

Every measurement column resolves to exactly one region via `calib_features.block_of` (already
existing, reused, not reinvented) — there turned out to be no genuine cross-part column in this
codebase's naming scheme to require the planned geometric-mean combination rule; it is
implemented in principle (`weights.py`) but unused in practice here.

## 2. Weighting schemes tested

Four, per the approved plan, all built only from `mesh_quality`/`part_quality` (never
genus/species/subfamily — verified by construction, see `weights.py`'s docstring):

- `hard50` — existing production filter (`measure.quality_filter`, boolean top 50%).
- `equal` — no filtering, the baseline every other scheme is compared against.
- `continuous_global` — one weight per specimen, rank-based (frozen, no free parameters) on the
  global composite.
- `continuous_part` — one weight per *measurement*, from the composite of the region that
  measurement is read off (frozen before any ground-truth comparison — see `weights.py`).

Formulas (weighted Pearson R, weighted MAE, weighted relative error) are defined once in
`diagnostics/reliability/weights.py` and reused identically for every scheme/region — see that
file for the exact expressions (matches the approved plan verbatim).

**One implementation bug found and fixed during this work, noted because it changed the
numbers substantially**: an earlier version of `synth_eval.py` z-scored and mean-centered
measurements before computing relative error, which pushed many ground-truth values near zero
and produced meaningless (200–900%) relative errors. Bone lengths and part extents are strictly
positive by construction (`measure.py`'s own docstring), so relative error is computed on raw
values — fixed before any result below was read or interpreted.

## 3. Primary evaluation: synthetic ground truth (n=24, `synth_clean` + `synth_noisy` pooled)

Full per-region, per-scheme table: `diagnostics/reliability/out/synth_eval.json`
(`diagnostics/reliability/synth_eval.py` reproduces it). Headline:

| region | baseline R (`equal`) | ΔR `continuous_global` | ΔR `continuous_part` | ΔR `hard50` |
|---|---|---|---|---|
| leg_distal | 0.113 (worst) | +0.057 | +0.044 | +0.101 |
| gaster | 0.478 | +0.026 | **−0.124** | −0.021 |
| antenna | 0.619 | +0.053 | +0.112 | −0.077 |
| mandible | 0.644 | +0.001 | −0.017 | +0.002 |
| leg_prox | 0.665 | −0.002 | +0.055 | −0.172 |
| mesosoma | 0.787 | +0.050 | **−0.113 (only cell whose bootstrap CI excludes zero)** | +0.079 |
| head | 0.921 (best) | +0.017 | +0.013 | +0.041 |

**The task's key question — does weighting preferentially help low-reliability regions more than
high-reliability ones?** Spearman correlation between a region's baseline R and its ΔR under
`continuous_global`: **ρ = −0.536 (p = 0.215, n = 7 regions)** — negative is the direction the
hypothesis predicts (more help where baseline reliability is lower), and it is the largest such
effect of any scheme, but it does **not** reach significance at only 7 regions. `continuous_part`
(ρ = −0.071, p = 0.879) and `hard50` (ρ = −0.036, p = 0.939) show essentially no such relationship
— `continuous_part`'s gains and losses do not track baseline reliability at all; it helps
`leg_distal`/`antenna`/`leg_prox` but actively hurts `gaster`/`mesosoma`/`mandible`.

**MAE tells a more consistent story than R.** Weighted MAE improves (goes down) under
`continuous_global` in 6 of 7 regions and under `continuous_part` in 5 of 7 — including the two
regions where `continuous_part`'s R got *worse* (`gaster` ΔMAE −0.225, `mesosoma` ΔMAE +0.036,
mixed). So weighting is fairly reliably shrinking absolute error even in cases where it is not
reliably improving rank-order correlation — a real if modest effect, but reported honestly as an
MAE effect, not inflated into an R claim it does not support.

**Applying the pre-registered decision rule as written (not adjusted after seeing results):** of
the 21 scheme×region deltas (3 schemes × 7 regions), most clear the ~3-percentage-point practical
threshold in at least one of ΔR/Δrelerr (expected — the threshold is generous at this sample
size), but **only one clears the statistical bar** (paired-bootstrap 95% CI on ΔR excluding zero):
`mesosoma` / `continuous_part`, ΔR = −0.113, CI [−0.211, −0.023] — and it is a **regression**, not
an improvement. No scheme anywhere produces a statistically distinguishable *improvement* in R at
n=24. This is the deciding fact for the verdict above.

## 4. Secondary evaluation: n=50 real corpus (exploratory only, not decisive)

Attempted on `diagnostics/d1_n50_evidence/runs/seed0/d1/Stage_3_deform_fine.npz` (50 real,
genus-labelled worker specimens, fit outputs already on this cluster — see
`diagnostics/reliability/n50_eval.py`). Result: **uninformative, not a failure of any weighting
scheme.** `bench50_clean` was built for registration-quality benchmarking diversity (close to one
specimen per genus, by design), so after the standard min-n≥3-per-genus lot-blind filter this
project's genus-lift protocol requires, only 8 specimens across 2 genera survive — far too few for
any classifier, weighted or not, to say anything. This is a corpus-selection mismatch, not
evidence against reliability weighting; a real test of downstream genus lift needs a corpus built
for taxonomic replication (min a few specimens per genus), which `bench50_clean` was never meant
to be.

**Cross-corpus retrieval was not attempted** — it needs the `ALL_ANTS_CLEAN` corpus alongside the
worker corpus, which this 50-specimen evidence set does not include.

## 5. Full 838-specimen corpus

**Blocked, as flagged in the approved plan.** `diagnostics/absolute_scale/morphometrics_source.csv`
/ `calibration_by_specimen.json` were built on fitted-mesh outputs (`Stage_3_deform_fine.npz`
equivalents) that do not exist on this cluster — only the pre-fit target scans do
(`/hpcwork/nao48500/worker_ALT`). Computing real-corpus registration quality at this scale would
require re-running the D1 fit on ~750 specimens, a multi-hour multi-GPU job explicitly out of
this task's scope per the approved plan. Not attempted.

## 6. Recommendation

Do not replace `hard50` in production on this evidence. The direction of the effect
(`continuous_global` trending toward helping unreliable regions more, and both continuous schemes
trending toward lower absolute error broadly) is worth a larger synthetic corpus if one becomes
cheap to generate — n=24 is simply too small to distinguish a real, modest effect from noise, and
this project's own convention (`FINAL_REPORT.md` §3.2, `offset_sweep_evidence`) is to report that
honestly rather than round a directional trend up to a finding. If a larger synthetic corpus or
the missing worker fit-outputs become available, `synth_eval.py`/`n50_eval.py` are written to run
unchanged at whatever N is available — rerunning them is the natural next step, not new code.

## Deviation from the approved plan

The plan proposed adding a `--weighting` flag directly to `diagnostics/absolute_scale/
analyse_source.py`. Its `main()` calls `build_table()`, which requires `MORPH_W*` fit-run
directories under `diagnostics/moonshot/runs/` that do not exist on this cluster (§5) — so a flag
on `main()` could not have been exercised here regardless. Instead, `n50_eval.py` imports and
reuses `analyse_source.py`'s module-level functions (`accession_lot`, `loo_1nn`, `perm_test`,
`select_features`, `features`, `pcs`) directly against the n=50 evidence corpus, which both
verifies the ported module (`analyse_source.py` now imports cleanly — it could not before this
task, since `measure.py`/`calib_features.py` were missing from the working tree) and avoids adding
an untestable flag to a script whose primary entry point cannot run here.

## Reproducing

```
conda activate pytorch3d
python diagnostics/reliability/synth_eval.py   # primary: synthetic ground truth, n=24
python diagnostics/reliability/n50_eval.py     # secondary: n=50 real corpus, exploratory only
```

`diagnostics/moonshot/runs/D1_N50_SEED0` is a local, gitignored symlink to
`diagnostics/d1_n50_evidence/runs/seed0/d1` (created so `n50_eval.py` could reuse
`measure.measure_run`'s existing `MOON/runs/<name>` convention rather than special-casing the
path) — recreate it if missing:

```
ln -sfn "$(pwd)/diagnostics/d1_n50_evidence/runs/seed0/d1" diagnostics/moonshot/runs/D1_N50_SEED0
```
