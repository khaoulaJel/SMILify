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

## Follow-up re-analysis (same fitted data, no new compute): fair reference + difficulty driver

Two follow-up checks on the diagnostic addendum above, both computed from the already-fitted
50-specimen x 3-seed data (`diagnostics/d1_n50_evidence/metrics.csv`) -- no new fits.

### 1. Corrected reference: stratified 10, not alphabetical-first-10

The alphabetical-first-10 (Task 1's reference set) turns out to be a biased sample: mean
`fscore@0.01` 0.615 vs the full-50 mean of 0.500, a ~23% gap driven entirely by sample
composition. A stratified reference -- the median specimen of each of 10 equal-sized bins of
the fscore-ranked 50 (bin boundaries and selection rule fixed before computing any comparison
stats, to avoid picking a reference that flatters the result) -- spans the full difficulty
range (0.21-0.74) and gives a recipe-consistent (both scale_cap) baseline:

| metric | full N=50 mean | stratified-10 reference | rel. delta |
|---|---|---|---|
| fscore@0.01 | 0.50052 | 0.49883 | +0.3% |
| chamfer_l2 | 0.00064 | 0.00063 | +1.1% |
| edge_logratio_absmean | 0.15238 | 0.14974 | +1.8% |
| deform_mag_mean | 0.00478 | 0.00485 | -1.5% |

All four land within ~2% of the fair reference.

**Correction (this claim was overstated when first written): this is NOT "bars 3-4 pass" in
the original sense, and does not, by itself, justify PROMOTE.** The stratified-10 reference
was carved out of the same 50-specimen pool it's being compared against. A representative
subsample of a population will always closely match that population's mean -- that's close to
true by construction, not something being tested. The original bars 3-4 question was a real
generalization test: does Task 1's N=10 result (an independent prior sample) predict the full
N=50? Once the alphabetical-first-10 is shown to be a biased sample, there is no longer any
independent reference left to test generalization against -- the corrected comparison above
only shows **internal consistency within the N=50 pool** (no specimen subset behaves
unexpectedly differently from the whole), which is a weaker claim.

**Honest framing: the original consistency check used a flawed reference; once that bias is
identified and removed, no internal inconsistency was found across the specimen pool -- but
this does not confirm N=50 matches an independent prior sample, because no independent sample
now exists to test against.** The alphabetical-bias diagnosis itself is solid and useful (it
explains *why* the original bars 3-4 failed, and rules out `w_scale` as the cause -- see the
same-10-specimens isolation above, which *is* a valid same-recipe comparison since it reuses
Task 2's genuinely independent, pre-existing Arm A numbers). But it does not license a revised
PROMOTE verdict on its own.

**Bars 1-2 (PASS, independent of this issue) plus this internal-consistency check are enough
to justify continuing to use `D1_low_scalecap.yaml` for real work** (see Practical note below)
-- they are not enough to justify changing the pre-registered promotion status. **Verdict:
HOLD stands** (unchanged from the original pre-registered result). A genuine test of bars 3-4
would need a held-out reference not carved from the same 50 -- e.g. fitting additional
specimens outside `bench50_clean` (such as the `ALL_ANTS_CLEAN` corpus named in
`FINAL_REPORT.md` §4, if accessible) as a truly independent check. Not run here -- worth
deciding explicitly whether it's worth the additional compute before any promotion claim goes
into the paper, rather than deciding unilaterally.

### 2. What actually drives the difficulty spread (the appendage-complexity hypothesis, tested and refuted)

Hypothesis tested: are the hard 40 specimens disproportionately long/complex-appendage genera
(trap-jaw, long-legged forms) vs compact-bodied genera -- i.e. is registration difficulty an
appendage-fitting problem specifically? **Not supported by this data; the pattern points the
other way.**

- Per-part distance ratio (hard 20 / easy 20, by fscore): `part_body_dist_mean` is the metric
  that degrades MOST between easy and hard specimens (2.42x), more than any single appendage
  part (antenna 1.68x, mandible 1.96x, leg 1.78x, leg distal 1.75x). If difficulty were
  appendage-specific, appendages should degrade disproportionately *more* than the body; they
  degrade *less*.
- Appendage-to-body error ratio is actually *higher* in the easy group (leg_distal/body 1.46,
  mandible/body 1.14, antenna/body 1.20) than the hard group (1.05, 0.92, 0.83) -- consistent
  with appendages being intrinsically hard to nail precisely even under good conditions, but
  the opposite of "hard specimens fail because of their appendages."
- **Natural within-species control**: `Anochetus_risii` has two specimens in this corpus.
  CASENT0877608 scores 0.724 fscore; CASENT0877609 (same species) scores 0.306 -- a >2x gap
  with genus/body-plan held constant. Genus-level appendage complexity cannot explain this.
- **The actual driver: `deform_mag_mean` correlates with `fscore@0.01` at r=-0.973 across all
  50 specimens** -- specimens needing more free-form deformation to fit score dramatically
  worse. The `Anochetus_risii` pair confirms it directly: CASENT0877608 (high fscore) needed
  `deform_mag_mean=0.00327`; CASENT0877609 (low fscore) needed 0.00647, almost 2x, and its
  `part_body_dist_mean` is 2.8x worse too (0.01844 vs 0.00655) -- the whole mesh is off, not
  just the appendages.

**Reframed finding**: difficulty is driven by how far the hierarchical placement stage lands
from the true pose/shape (forcing the moonshot free-form term to compensate), not by
appendage/genus morphology. This is the same mechanism `FINAL_REPORT.md` §4 already names as
the working downstream filter (`quality_composite`, a z-sum of deform/edge/normal-roughness,
keep the best ~50%) -- this run is a second, independent confirmation of that filter's premise
on real bench50_clean data, not a new appendage-specific lever. The standing appendage-metric
weakness (leg ratios R=0.136, per `FINAL_REPORT.md` §4) looks like a separate, downstream
morphometrics-measurement issue, not evidence that this registration difficulty spread is
appendage-driven -- worth keeping the two apart rather than merging them into one story.

**Roadmap implication (this changes the next-step recommendation, not just a closed side
question)**: appendage-specific measurement (centerline/geodesic length extraction) is not
supported as the next lever by this evidence -- the r=-0.973 correlation and the within-species
`Anochetus_risii` control both say the difficulty spread is about how far hierarchical
placement lands from truth, independent of which body part is being measured downstream. The
more valuable next step this result points to: characterize *what makes hierarchical placement
fail on specific specimens* (pose ambiguity in the initial estimate? scan noise/incompleteness?
a bad H0/H1/H2 initialization?) -- that is the actual lever this data identifies, not a
per-appendage measurement strategy.

### Practical note

Bars 1-2 (no catastrophic failures, ~3% cross-seed CV) were never in question and gate safety,
not just administrative promotion status. Regardless of where the promote/hold documentation
lands, `D1_low_scalecap.yaml` was already safe to use for real morphometrics work throughout
this analysis.

## Second follow-up: genuine held-out corpus (ALL_ANTS_CLEAN, 80 specimens, 3 seeds)

Per the earlier correction, bars 3-4 need a reference not carved from the N=50 pool. Pulled
`ALL_ANTS_CLEAN` (80 unique specimens after de-duplicating one identical file, `feature/
investigation`'s independent 838-specimen scale_cap A/B corpus) from `gdrive:UM6P_2026/DATA/
mesh_registration/ALL_ANTS_CLEAN` (the corpus was previously unreachable -- only present as git
symlinks to `/media/fabi/Data/...` or an unreachable Julich `/p/scratch` mount; a working
`gdrive:` rclone remote made it reachable). Ran the identical `D1_low_scalecap.yaml` recipe,
full budget, 3 seeds (0/1/2), on all 80 specimens (`diagnostics/d1_n50_evidence/
all_ants_clean_holdout/`, gitignored, not committed -- same treatment as other raw mesh sets in
this repo). This is a genuinely independent reference: different files, different source,
never touched by any config/tuning decision in this investigation.

### Result: bars 3-4 FAIL again, but for a different, more informative reason

The holdout corpus fits **dramatically better** than `bench50_clean`, not worse:

| metric | N=50 (bench50_clean) mean | ALL_ANTS_CLEAN holdout mean | direction |
|---|---|---|---|
| fscore@0.01 | ~0.500 | ~0.896 | holdout 79% higher |
| chamfer_l2 | ~0.00064 | ~0.00010 | holdout 6.3x lower (better) |
| edge_logratio_absmean | ~0.152 | ~0.108 | holdout 29% lower (better) |
| deform_mag_mean | ~0.0048 | ~0.0018 | holdout 62% lower (better) |
| folded_face_frac | ~0.0051 | ~0.0026 | holdout ~50% lower (better) |

(all consistent within <1% across seeds 0/1/2 on both corpora)

**This is not evidence of a recipe problem -- the two named corpora have very different
intrinsic registration difficulty**, most likely because `ALL_ANTS_CLEAN` (curated
genus-representative meshes) is a cleaner/easier target class than `bench50_clean`'s raw
museum/field scans, independent of anything D1 does. Bars 3-4 as designed (mean must not
regress vs. an external reference by more than a fixed threshold) implicitly assume the
reference and test set are drawn from populations of comparable difficulty. We have now shown
this assumption fails in **both directions**: the N=10 bench50_clean subset was biased easy
relative to its own N=50 (first correction), and the ALL_ANTS_CLEAN corpus is far easier than
either (this correction). **No single external reference, however independent, can validly
gate an absolute-accuracy bar this way** -- accuracy is corpus-dependent, and independence of
the reference does not imply comparable difficulty.

### What this run does validly establish

- **Bar 1, cross-corpus**: 0 catastrophic failures across bench50_clean (150 seed x specimen
  runs) AND ALL_ANTS_CLEAN (240 seed x specimen runs) -- 390 total runs, two independently
  sourced real ant-mesh corpora, zero NaN/collapse/folding blowups.
- **Bar 2, cross-corpus**: cross-seed CV ~3% on bench50_clean, ~1% on ALL_ANTS_CLEAN (even
  tighter) -- D1 is stable across seeds on both corpora independently, not just one.

### Revised verdict (final for this evidence)

**HOLD**, unchanged -- but the reason has moved. Bars 1-2 are now validated across two
independent corpora (stronger evidence than the original single-corpus design), which is a
genuinely stronger basis for treating `D1_low_scalecap.yaml` as safe to use. Bars 3-4, as
literally specified in `PROMOTION_CRITERIA.md`, are not a well-posed test: they cannot be
satisfied by any external reference without first establishing that reference is
difficulty-matched to the target corpus, which no readily available corpus was shown to be.
**Recommendation for any future promotion decision: replace bars 3-4 with corpus-relative
checks (e.g. cross-seed CV per corpus, as bar 2 already does) rather than an absolute mean
comparison against an external reference corpus**, and drop the implicit assumption that
"independent" and "comparable difficulty" are the same property.
