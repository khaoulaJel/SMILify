# TASK 1 — gentle proximity penetration loss + scheme:'all', bench50_clean, re-scored under the Hard Rule

Pure re-analysis. No new GPU training. Existing runs (`scripts/penetration_joint_study/`,
step 1 + the reseeded rerun in `FINDINGS.md`) already cover exactly what TASK 1 asked for:
bench50_clean (50 specimens), `scheme: 'all'` in both deform stages, gentle penetration weight
(0.02/0.05) vs `w_penetration=0` baseline, unseeded + 3-seed reruns with `w_limit` correctly
active in all four stages. Re-scored here with the Hard Rule panel (per-pair gaster-legs count,
F-score@0.01, deform_mag/edge_logratio, outlier renders) instead of the aggregate
`penetration_num_penetrating` FINDINGS.md used.

Reproduce: `python diagnostics/khaoula_review/task1_hardrule_reanalysis_PROBE.py`
(raw output: `task1_hardrule_reanalysis_out.txt`). Renders:
`python diagnostics/khaoula_review/task1_outlier_render_PROBE.py`
(`out/task1_outliers_lateral_XZ.png`, `out/task1_outliers_anterior_YZ.png`).

## 0. Premise correction, checked before scoring anything

The task brief for this run states gaster-legs is "the only pair any penetration term
targets." **That is not true of the current code.** `fitter_3d/trainer.py:491` builds
`self.non_adjacent_pairs = get_non_adjacent_pairs(PART_GROUPS_COARSE)`
(`fitter_3d/part_groups.py:186`), which returns **every** anatomically non-adjacent coarse-part
pair — 10 of them (`antenna-gaster`, `antenna-legs`, `antenna-thorax`, `gaster-legs`,
`head-gaster`, `head-legs`, `mandible-antenna`, `mandible-gaster`, `mandible-legs`,
`mandible-thorax`) — and `penetration_loss.py` trains and is scored on all of them
simultaneously, confirmed by the fact all 10 appear in `Stage_3_deform_fine_per_pair_penetration.csv`
for both arms. Scoring gaster-legs alone, as instructed, is still useful (see below) but is
**not** the complete targeted-pair picture — it's one of ten. The full per-pair table is included
so this isn't a repeat of the "score on count alone" mistake the Hard Rule exists to prevent.

## 1. Gaster-legs only (as instructed)

| | unseeded (n=50) | 3-seed pooled (n=150) |
|---|---|---|
| baseline mean count | 414.66 | 406.09 |
| gentle mean count | 328.54 | 326.71 |
| delta | **-86.12 (-20.8%)** | **-79.38 (-19.5%)** |
| per-specimen improved/worse/flat | 31/17/2 | 88/60/2 |

Per-seed: seed0 -22.6%, seed1 -18.5%, seed2 -17.2% — **same direction in all three seeds**,
magnitude stable within ~1.3x. This reverses the sign of FINDINGS.md's aggregate headline
(+16.8% / +14.36% worse) for this specific pair.

Seed-stability at the per-specimen level (majority vote, gaster-legs only): 18/50 unanimous
improved, 10/50 unanimous worse, **22/50 (44%) flip sign across seeds** — the same seed-noise
fraction FINDINGS.md found for the contaminated aggregate. The population-level direction is
solid; individual-specimen verdicts are not, same caveat as before.

## 2. Full per-pair table (unseeded arm) — why gaster-legs improving doesn't mean the loss "worked"

| pair | baseline mean | gentle mean | delta | pct |
|---|---:|---:|---:|---:|
| gaster-legs | 414.66 | 328.54 | -86.12 | **-20.8%** |
| head-gaster | 0.08 | 0.06 | -0.02 | -25.0% |
| mandible-antenna | 55.88 | 76.90 | +21.02 | +37.6% |
| head-legs | 63.62 | 132.68 | +69.06 | +108.6% |
| mandible-legs | 9.52 | 26.50 | +16.98 | +178.4% |
| antenna-legs | 14.00 | 54.42 | +40.42 | +288.7% |
| mandible-thorax | 3.16 | 16.56 | +13.40 | +424.1% |
| antenna-thorax | 4.26 | 23.70 | +19.44 | +456.3% |
| antenna-gaster | 0.00 | 0.84 | +0.84 | — |
| mandible-gaster | 0.00 | 0.00 | 0.00 | — |
| **TOTAL** | **565.18** | **660.20** | **+95.02** | **+16.8%** |

(TOTAL reproduces FINDINGS.md's step-1 number exactly — 565→660, +16.8% — confirming this table
is the correct decomposition of their aggregate, not a different measurement.)

**Reading**: the gentle loss trades gaster-legs contact (the anatomically dominant, largest-count
pair — a real reduction, replicated across seeds) for substantially worse contact everywhere
else, mostly in the extremities against legs/thorax. Those other pairs start from small absolute
counts, so the *relative* blowups (up to +456%) look dramatic but the *absolute* damage
(+13 to +69 counts per pair) is individually smaller than the gaster-legs win. Summed, the
losses outweigh the gain: net aggregate is worse, matching FINDINGS.md. Whether the gaster-legs
number alone is "the" answer depends on which pair Fabian actually cares about — the brief calls
it out as the one that matters anatomically (body-cavity vs. limb), so it's reported both ways
rather than picking one.

## 3. F-score@0.01 (surface-fit sanity check)

| | unseeded | 3-seed pooled |
|---|---|---|
| mean delta | -0.0048 | -0.0060 |
| median delta | -0.0034 | -0.0041 |
| range | -0.0596 .. +0.0310 | -0.0501 .. +0.0194 |

Small mean cost, consistent with Khaoula's original "<3% on every specimen" claim being
*mostly* right but not universally — `Heteroponera_panamensis` loses 0.060 (6 pp) at
tau=0.01, `Anochetus_risii` loses 0.035, both well outside a 3% band framed in relative terms.
Source: `Stage_3_deform_fine_eval_metrics.csv`, produced by the training script itself (same
code path both arms, no re-derivation needed).

## 4. Integrity (deform_mag, edge_logratio)

| | unseeded | 3-seed pooled |
|---|---|---|
| deform_mag delta | +0.00152 (+4.13%) | +0.00173 (+4.75%) |
| edge_logratio delta | +0.03070 (+5.63%) | +0.03868 (+7.19%) |

Real, consistent-direction cost, modest in magnitude — not the GWN-alone collapse pattern
(no specimen shows the "legs fused into gaster" failure mode). Computed off the same saved
`Stage_3_deform_fine.npz` verts via `diagnostics/moonshot/eval_run.py` (this project's existing
off-the-shelf metrics tool — no new metric code written for this).

## 5. Visual renders — outliers

Flagged: top-3 by |Δgaster-legs count| (`Strumigenys_crassicornis` -695,
`Polyrhachis_dives` -506, `Ectatomma_brunneum` -472) and top-3 by |ΔF-score@0.01|
(`Heteroponera_panamensis` -0.060, `Anochetus_risii` -0.035, `Acanthostichus_aff.brevicornis`
+0.031). Lateral (XZ) and anterior (YZ) views, coloured by `|deform_verts|`, baseline vs gentle
side by side: `out/task1_outliers_lateral_XZ.png`, `out/task1_outliers_anterior_YZ.png`.

**No collapse or tearing in any of the 6.** Both arms are recognisable, correctly-posed ants;
the baseline/gentle columns are visually close, consistent with the modest integrity deltas in
§4 — this is a real but non-catastrophic effect, not a repeat of the GWN-as-loss failure mode.

## Answer to Fabian's question, stated plainly

**Does gentle proximity penetration loss + scheme:'all' help, on bench50?**

- On the single pair called out as the one that anatomically matters (gaster-legs): **yes**,
  count drops ~20%, consistently across 3 seeds, at matched (slightly worse) F-score and
  integrity, no collapse.
- On the full set of pairs the loss actually trains on (10 pairs, not 1): **no** — the
  gaster-legs win is outweighed by worse contact in every other pair, net aggregate +16.8%
  worse, reproducing FINDINGS.md's original conclusion.
- Both of these are true at once; they are not in tension, they are answers to two different
  (and previously conflated) questions. The task brief's framing assumed the second question's
  scope but the first question's premise ("only pair"), which is why this section exists.
- Per-specimen: still a coin flip with 44% seed-instability, same as the aggregate-scored
  version — this loss's effect is a real population-level trade, not a reliable per-specimen fix.
