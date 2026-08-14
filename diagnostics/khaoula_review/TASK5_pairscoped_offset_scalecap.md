# TASK 5 — scoped gaster-legs + w_offset + scale/trans barriers, bench50_clean

One bench50 arm, per direct instruction: "scoped gaster–legs + w_offset + scale/trans
barriers... the last structural lever that is both on Fabian's list and untested on your
stack." Pre-registered before running.

## Code change

`fitter_3d/trainer.py`: wired `w_scale`/`w_trans` (scale_barrier/trans_barrier on
`log_beta_scales`/`betas_trans`, `fitter_3d/joint_limits.py`), ported verbatim from
`trainer_moonshot.py:513-521` — same call, same names, so a weight tuned on one pipeline
means the same on the other. This mechanism was implemented and calibrated
(`w_scale=0.052`, T0.4 full-corpus A/B, `diagnostics/EXECUTION_PLAN.md` §5, ADOPTED into
production `D1_PROD.yaml` for the moonshot/hierarchical genus-classification pipeline) but
**never run on `fitter_3d/trainer.py`** (the pipeline this whole penetration-loss study
uses) before this task. `w_trans` has no calibrated value anywhere in the branch — the only
value ever tried (1.0, `diagnostics/INVENTORY_V2.md` T0.4b) left the barrier ≈0 throughout,
"not meaningfully engaged." Used it anyway (not invented) since it's the sole precedent and
the request named both scale and trans explicitly; not expected to move anything on its own.

Config: `fitter_3d/ants_cfg_all_offset_gentle_pairscoped_scalecap.yaml` = TASK4's
pair-scoped gentle+offset config (gaster-legs-only `penetration_train_pairs`, w_offset
5.0/2.0, w_limit 100.0, scheme:'all') plus `w_scale: 0.052`, `w_trans: 1.0` on both deform
stages. Smoke-tested on 3 specimens first — no errors, `scale_barrier`/`trans_barrier` loss
keys present, `trans_barrier` ~0 at 10 iterations (expected, matches T0.4b), `scale_barrier`
0 at 10 iterations but engaged (0.0186) by the end of the full 1000-iteration run.

## Job

Unseeded, bench50_clean, 50/50 specimens, job 15526415, 8:59 elapsed, exit 0, no errors.
(First submission, 15526408, failed at 1:27 on `OSError: Disk quota exceeded` writing
`Stage_0_init.npz` — filesystem quota, unrelated to this code; user freed space,
resubmission ran clean.) → `fit3d_results_all_offset_gentle_pairscoped_scalecap/`.

Reproduce: `python3 diagnostics/khaoula_review/task5_hardrule_reanalysis_PROBE.py`
(raw output: `task5_hardrule_reanalysis_out.txt`). Renders:
`python diagnostics/khaoula_review/task5_outlier_render_PROBE.py`.

## Pre-registered success criteria — result

| criterion | target | result | verdict |
|---|---|---:|---|
| all-pairs aggregate | ≤ 0% (keep the net win) | **−16.9%** | **PASS** |
| gaster-legs count | ≤ −15% (softer than TASK2's −20.2% accepted) | **−15.0%** | **PASS** (exactly at the line) |
| mean F-score@0.01 delta | ≥ −0.01 | **−0.0195** | **FAIL** |
| worst-specimen F-score delta | ≥ −0.08 | **−0.0890** | **FAIL** |
| no collapse on outliers | — | none seen | PASS |

**2 of 4 fail, same two as TASK4** (mean and worst F-score) — adding the scale/trans
barriers did not close that gap, and mean F-score got slightly worse, not better.

## What the barriers actually changed (vs TASK4's pairscoped-gentle, no scalecap)

| | TASK4 (no scalecap) | TASK5 (+scalecap) | delta |
|---|---:|---:|---:|
| gaster-legs count | −16.8% | −15.0% | +7.4 count (slightly weaker) |
| all-pairs count | −13.6% | **−16.9%** | −18.4 count (**stronger net win**) |
| mean F-score delta | −0.0170 | −0.0195 | −0.0026 (slightly worse) |
| worst F-score delta | −0.0925 | −0.0890 | +0.0035 (marginally better) |
| deform_mag delta | +21.56% | +27.00% | worse |
| edge_logratio delta | **+21.08%** | **+4.90%** | **much better (4.3x smaller)** |

Mixed, not uniformly better or worse. The scale/trans barriers **substantially improved
edge-distortion integrity** (+21.1% → +4.9%, matching the barrier's intended job — bounding
absurd per-joint scale/translation reduces the local mesh stretching that produces
edge-length distortion) and **strengthened the all-pairs aggregate win further**
(−13.6% → −16.9%). But they did **not** fix mean/worst F-score, and cost slightly more in
deform_mag. Full per-pair table (in the raw output) shows most individual pairs shift by a
few counts in either direction — no single pair drives the aggregate improvement, consistent
with a genuine structural effect rather than one specimen's fit changing.

## Visual check

Top 3 by |Δgaster-legs count| (`Strumigenys_crassicornis` −826, `Aphaenogaster_pallida`
−714, `Temnothorax_congruus` −700) and top 3 by |Δfscore@0.01| (`Atopomyrmex_mocquerysi`
−0.0890, `Mesoponera_ambigua` −0.0698, `Strumigenys_stenorhina` +0.0662). Lateral (XZ) and
anterior (YZ), baseline vs pairscoped-gentle+scalecap, coloured by |deform_verts|:
`out/task5_outliers_lateral_XZ.png`, `out/task5_outliers_anterior_YZ.png`.

**No collapse or tearing in any of the 6** — same "real but non-catastrophic" pattern as
every prior arm.

## Decision, per the pre-registered branch instructions

Step 1 (this arm) still fails F-score (mean −0.0195 < −0.01 bar). Per the instruction's own
step 2: **treat soft proximity as optional and gated, not the solution.**

- **Ship**: pair-scoped gentle+offset(+scalecap optional) as a config option, gated by the
  standing required per-specimen F/integrity check (`REVIEW_penetration_update.md`'s
  recommendation, unchanged by TASK1-5).
- **Default remains**: D1 / no penetration loss (`w_penetration=0`). Nothing in this task
  changes any default — `w_penetration`, `penetration_train_pairs`, `w_scale`, `w_trans` are
  all off unless a config sets them.
- **Product bar** ("net aggregate ≤ 0 and mean F within −0.01, simultaneously") has **not**
  been met by any soft-proximity-loss arm across TASK1-5. TASK5 gets closer on the aggregate
  (−16.9%, the best yet) and on edge-distortion integrity, but not on mean F-score, which is
  the harder constraint every arm has failed.
- **Step 3 (part-filtered conical/BVH loss) is not attempted here** — it is explicitly gated
  behind a product decision ("only if product still requires lower residual contact"), not a
  research default, and that decision belongs to whoever owns the product bar, not this task.
