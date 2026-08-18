# D1 N=50 x 3-seed validation scorecard

## VERDICT: HOLD (stay experimental-mainline candidate)

- Recipe: `diagnostics/moonshot/cfg/D1_low_scalecap.yaml` (D1_low.yaml + w_scale: 0.052, see PROMOTION_CRITERIA.md addendum for why this recipe was used instead of D1_low.yaml).
- N=50 (full `bench50_clean`), seeds {0,1,2}, full budget (1000+1000 its), 1x H100 each (Slurm array job 3034796, ~17 min/seed).
- Promotion bar and design pre-registered in `PROMOTION_CRITERIA.md` before this job was submitted.

## Bar 1: no catastrophic per-specimen failures

**PASS** -- 0 catastrophic events out of 150 (seed, specimen) runs.

## Bar 2: cross-seed variance (CV = std/mean across 3 seeds, averaged over 50 specimens)

**PASS** (threshold: CV < 15%)
| metric | mean CV across specimens |
|---|---|
| edge_logratio_absmean | 3.04% |
| deform_mag_mean | 3.01% |

## Bar 3: N=50 vs Task 1's N=10 D1 numbers (per seed)

**FAIL** (threshold: <15% relative regression)
| seed | edge_logratio_absmean | deform_mag_mean |
|---|---|---|
| 0 | 0.15283 | 0.00482 |
| 1 | 0.15237 | 0.00476 |
| 2 | 0.15194 | 0.00476 |
| N=10 reference (Task 1) | 0.14856 | 0.00401 |
- FAIL detail: seed0 deform_mag_mean: 0.00482 vs N=10 ref 0.00401 (+20.1%)
- FAIL detail: seed1 deform_mag_mean: 0.00476 vs N=10 ref 0.00401 (+18.7%)
- FAIL detail: seed2 deform_mag_mean: 0.00476 vs N=10 ref 0.00401 (+18.7%)

## Bar 4: fscore/chamfer don't collapse

**FAIL**
| seed | fscore@0.01 mean | chamfer_l2 mean |
|---|---|---|
| 0 | 0.4984 | 0.00064 |
| 1 | 0.5018 | 0.00064 |
| 2 | 0.5015 | 0.00063 |
- FAIL detail: seed0 fscore@0.01=0.4984 worse than ref 0.6251 by >0.05
- FAIL detail: seed0 chamfer_l2=0.00064 worse than ref 0.00038 by >50%
- FAIL detail: seed1 fscore@0.01=0.5018 worse than ref 0.6251 by >0.05
- FAIL detail: seed1 chamfer_l2=0.00064 worse than ref 0.00038 by >50%
- FAIL detail: seed2 fscore@0.01=0.5015 worse than ref 0.6251 by >0.05
- FAIL detail: seed2 chamfer_l2=0.00063 worse than ref 0.00038 by >50%

## Full integrity metric means (mean across 50 specimens, per seed)

| metric | seed0 | seed1 | seed2 |
|---|---|---|---|
| edge_logratio_absmean | 0.15283 | 0.15237 | 0.15194 |
| deform_mag_mean | 0.00482 | 0.00476 | 0.00476 |
| deform_mag_p95 | 0.01298 | 0.01290 | 0.01282 |
| deform_mag_max | 0.18807 | 0.18400 | 0.18613 |
| folded_face_frac | 0.00525 | 0.00496 | 0.00521 |
| dihedral_p99 | 70.80110 | 69.86047 | 70.56721 |
| tri_quality_mean | 0.74625 | 0.74817 | 0.74867 |
| fscore@0.01 | 0.49835 | 0.50176 | 0.50145 |
| chamfer_l2 | 0.00064 | 0.00064 | 0.00063 |

## Diagnostic addendum: why Bar 3/4 failed (not adjusted after the fact, added for interpretation)

Bars 3 and 4 compare the **N=50 mean** against Task 1's **N=10 mean** as the reference. That
comparison conflates two different things: the recipe change (`w_scale: 0.052` added) and the
specimen-set change (40 new, harder specimens added). Isolating them, restricted to just the
original 10 specimens (same mesh set as Task 1/Task 2, this run's seed0/1/2):

| metric | Task 2 Arm A (N=10, no w_scale) | This run, same 10 specimens (w_scale=0.052), seed0/1/2 |
|---|---|---|
| fscore@0.01 | 0.6251 | 0.610 / 0.621 / 0.615 |
| chamfer_l2 | 0.00038 | 0.00043 / 0.00040 / 0.00042 |
| edge_logratio_absmean | 0.1470 | 0.1488 / 0.1462 / 0.1515 |
| deform_mag_mean | 0.00398 | 0.00409 / 0.00402 / 0.00413 |

All four land within noise of the no-`w_scale` reference on the *same 10 specimens* -- no
regression attributable to `w_scale`. The other 40 specimens average `fscore@0.01` ~0.47 across
all 3 seeds (vs ~0.61-0.62 on the original 10, regardless of seed) -- these are simply harder
targets, present in the full 50 but absent from Task 1's N=10 subset (which was the first 10
*alphabetically*, not selected for representativeness).

**This means Bar 3/4 as literally pre-registered were a flawed proxy for "did `w_scale` hurt
accuracy" -- they test "is N=50 as easy as N=10", which was never guaranteed.** The verdict
above (HOLD) is reported as pre-registered and is not overridden by this diagnosis. But the
honest read is: this run does not show evidence that `w_scale` or scale at N=50 degrades D1 --
it shows the original 10-specimen benchmark undersamples difficulty. A clean promotion decision
needs a same-specimen-set comparison (N=50 with vs without `w_scale`, or a properly pre-registered
N=50 baseline instead of reusing the N=10 numbers as ground truth) -- not run here, since
launching more compute against a still-open methodological question is a decision worth stating
rather than making unilaterally.
