# TASK 4 — pair-scoped gentle penetration loss (gaster-legs only), bench50_clean

**Not** the original task spec's TASK 4 (torch-mesh-isect) — this is a follow-up experiment
requested directly, testing whether restricting the w_penetration loss to gaster-legs only
(instead of all 10 non-adjacent pairs) turns TASK1/TASK2's documented gaster-legs win into a
clean net win. Pre-registered before running (success criteria below, fixed in advance).

## Code change

`fitter_3d/trainer.py`: new `Stage.__init__` kwarg `penetration_train_pairs` (defaults to
`None` = all non-adjacent pairs, unchanged prior behaviour for every existing config). When
set, restricts which pairs the **w_penetration loss** trains on. `compute_eval_metrics`
(the per-pair CSV) is untouched and always scores all 10 pairs regardless, so collateral
effects on untrained pairs stay fully visible — verified directly: the pair-scoped run's
`Stage_3_deform_fine_per_pair_penetration.csv` still has all 10 pairs (checked against the
unscoped smoke run, identical pair set). Smoke-tested on 3 specimens
(`diagnostics/khaoula_review/smoke_offset_gentle_pairscoped.yaml`) before the full run —
penetration loss nonzero, no errors, eval CSV correctly still has all 10 pairs.

Config: `fitter_3d/ants_cfg_all_offset_gentle_pairscoped.yaml` — identical to TASK2's
`ants_cfg_all_offset_gentle.yaml` (scheme:'all', gentle w_penetration 0.02/0.05, w_offset
5.0/2.0, w_limit 100.0 in all stages) plus `penetration_train_pairs: [[gaster, legs]]` on
both deform stages. Only that one line changed.

## Jobs (bench50_clean, 50/50 specimens both runs, both COMPLETED cleanly, exit 0, no errors)

- Unseeded: job 15525517, 9:43 elapsed → `fit3d_results_all_offset_gentle_pairscoped/`
- Seed 1: job 15525518, 13:40 elapsed → `fit3d_results_seeded/offset_gentle_pairscoped_seed1/`

Reproduce scoring: `python3 diagnostics/khaoula_review/task4_hardrule_reanalysis_PROBE.py`
(raw output: `task4_hardrule_reanalysis_out.txt`). Renders:
`python diagnostics/khaoula_review/task4_outlier_render_PROBE.py`
(`out/task4_outliers_lateral_XZ.png`, `out/task4_outliers_anterior_YZ.png`).

Compared against TASK2's already-run `fit3d_results_all_offset_baseline` (unaffected by this
change, w_penetration=0) and `fit3d_results_all_offset_gentle` (TASK2, all-10-pairs).

## Pre-registered success criteria — result

| criterion | target | result | verdict |
|---|---|---:|---|
| gaster-legs count | ≤ TASK2 level (≈ −20%) | **−16.8%** | CHECK — weaker than TASK2, not better |
| all-pairs aggregate | ≤ 0% (or clearly better than TASK2's +5.4%) | **−13.6%** | **PASS** |
| mean F-score@0.01 delta | ≥ −0.01 | **−0.0170** | **FAIL** |
| worst-specimen F-score delta | ≥ −0.08 | **−0.0925** | **FAIL** |
| no collapse on outliers | — | none seen | PASS |

**2 of 4 numeric criteria fail, 1 is weaker than the comparison point it was supposed to at
least match.** This is not the clean win the hypothesis predicted.

## What actually happened

| | baseline | TASK2 (all-10) | TASK4 (pairscoped) |
|---|---:|---:|---:|
| gaster-legs count | 418.94 | 334.46 (−20.2%) | 348.54 (**−16.8%**) |
| all-pairs count | 560.88 | 591.08 (+5.4%) | 484.46 (**−13.6%**) |
| fscore@0.01 mean delta | — | −0.0032 | **−0.0170** |
| fscore@0.01 worst | — | −0.1125 | −0.0925 |
| deform_mag delta | — | +5.72% | **+21.56%** |
| edge_logratio delta | — | +6.44% | **+21.08%** |

**The all-pairs aggregate is a genuine win** — first arm in this series to actually beat
baseline (−13.6%), not just cost less than baseline. Full per-pair table confirms the
mechanism: untrained pairs mostly return to baseline or better (antenna-legs 10.39→5.01,
head-legs 54.62→**31.86, better than baseline's 40.12**, mandible-thorax 5.79→1.73). Removing
the gradient on those pairs removes the collateral damage on them, as hypothesized.

**But it comes at a cost the hypothesis didn't predict**: with nowhere else to "escape" to,
all the deformation pressure concentrates on gaster-legs, and integrity cost roughly
**quadruples** (deform_mag/edge_logratio both ~+21% vs TASK2's ~+6%). Surface fit also costs
more (mean fscore delta −0.017 vs TASK2's −0.003) despite the trained pair's own count
improving *less* (−16.8% vs −20.2%) than when trained alongside the other 9 pairs — training
on gaster-legs alone does not train it better; it appears the other pairs' gradients were
incidentally helping resolve gaster-legs contact too, not just damaging elsewhere. A
squeezed-balloon effect, not a clean isolation of the win.

## Seed check — this result is noisier than the effect size

Seed 1 (no seed-matched baseline was run — deliberately cheap, per the "unseeded + 1 seed
check" pre-registration; flagged explicitly as approximate for this reason) gives gaster-legs
count 390.14 vs the unseeded run's 348.54, and all-pairs 514.10 vs 484.46 — a roughly
**40–70% swing in the delta-from-baseline size between the two runs of the same arm**
(gaster-legs: −6.9% at seed1 vs −16.8% unseeded; all-pairs: −8.3% vs −13.6%). Mean
per-specimen |delta| between the two seeds of the *same arm* is **170 counts**, larger than
the ~70-count treatment effect the headline number is built on. This does not mean the effect
is fake — both seeds agree on direction (gaster-legs down, all-pairs down) — but it means the
**magnitude** of the headline −16.8%/−13.6% numbers should not be trusted to better than
roughly ±10 percentage points without a proper seed-matched multi-seed rerun, which this cheap
check was not designed to provide.

## Visual check

Top 3 by |Δgaster-legs count| (`Temnothorax_congruus` −775, `Strumigenys_crassicornis` −676,
`Aphaenogaster_pallida` −604) and top 3 by |Δfscore@0.01| (`Myrmecina_sp.` −0.0925,
`Dilobocondyla_fouqueti` −0.0661, `Oxyopomyrmex_saulcyi` −0.0576). Lateral (XZ) and anterior
(YZ), baseline vs pairscoped-gentle side by side, coloured by |deform_verts|:
`out/task4_outliers_lateral_XZ.png`, `out/task4_outliers_anterior_YZ.png`.

**No collapse or tearing in any of the 6** — both arms are recognisable, correctly-posed
ants, consistent with "real but non-catastrophic" integrity cost, same pattern as every prior
arm in this series, just larger in magnitude.

## Verdict

Restricting the loss to gaster-legs does what it was supposed to do to the *other nine
pairs* — collateral damage there is largely eliminated, and the all-pairs aggregate turns
genuinely positive for the first time (−13.6%). **It does not do what the hypothesis
predicted to gaster-legs itself or to surface-fit/integrity**: the trained pair's own count
win is weaker (not stronger) than training jointly, F-score cost roughly quadruples relative
to TASK2's already-modest cost, and the seed check shows the headline magnitude is not
well-pinned by a single unseeded run. This is a genuine partial result — real evidence that
scope-restriction removes the *collateral* half of TASK1's problem — but it is not the "net
win, no new loss needed" outcome the hypothesis anticipated, and should not be reported to
Fabian as a solved case. If pursued further, the next step is a proper 3-seed rerun of both
arms (matching TASK1's own seeded-rerun protocol) before any recommendation change, since the
current single seed-check already shows the effect size is not stable at that precision.
