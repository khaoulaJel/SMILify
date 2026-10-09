# V2 — pre-registration. Does `deform_verts` produce the corpus-wide impossible traits, and
# does removing it cost biological signal?

Written 2026-09-07 BEFORE the run, from V1's finding. Results appended after.

## The claim under test

T1 found **69 of 757 production fits (9.1%) emit at least one biologically impossible trait**.
T1's proposed mechanism (area-sampled inflation of thin structures) was **refuted by T2**, and the
question has been open since. V1 (n=14) found impossible-trait count tracks `deform_verts`
magnitude across nine arms (Spearman 0.932) and that both zero-deformation arms had **zero**
violations. V2 tests that on the whole corpus.

`trainer.py:352` applies deformation as the final operation, `verts = verts + deform_verts`, after
LBS and translation. So `verts - deform_verts` **exactly** recovers the undeformed fit. No
refitting, no GPU, no annotation.

## The confound that must be ruled out, stated first

Removing deformation moves every specimen onto the pure SMAL manifold, and with betas at |z| ~ 0.81
that manifold sits near the mean ant — which is plausible **by construction**. So "zero violations"
could be trivial: not "deformation caused the violations" but "the undeformed fit is a bland
average ant". V2 is only informative if the undeformed fits **retain between-specimen variation**.
Two checks below do that work, and the result is void without them.

## Arms

Traits via `trait_extract.traits()` on the same 757 Z8 fits, twice:
- **WITH** deformation — the production values
- **WITHOUT** — `verts - deform_verts`

## Measured

1. flagged fraction under T1's pre-set bounds, counted as **unique specimens** (T1's convention)
2. per-specimen transitions: flagged→valid, valid→flagged
3. **spread**: sd of log trait ratios across the corpus, WITH vs WITHOUT
4. **genus F-ratio** (T4's statistic, between-genus / within-genus variance), WITH vs WITHOUT
5. bilateral asymmetry, WITH vs WITHOUT

## Pre-registered bars

**VOIDING CONTROL.** Traits recomputed WITH deformation must reproduce `traits_Z8.npz` to better
than 1e-6 relative, and the flagged count must come out at **69/757**. Any deviation means this
pipeline is not T1's and no row is comparable. (The script that built `traits_Z8.npz` is not in the
repo, so this is also its reconstruction.)

**H1 — deformation makes the violations.** Removing it drops the flagged fraction from 9.1% to
**below 3%** (i.e. at least two thirds of violations are deformation-made). V1 predicts ~0.

**H2 — NOT trivial (the confound above).** The undeformed traits must retain at least **60%** of
the WITH-deformation spread, per ratio. Below that, the undeformed fit has collapsed toward the
mean and H1 is an artefact of blandness. **H2 failing voids H1's interpretation.**

**H3 — removing deformation does not cost biological signal.** Mean genus F-ratio WITHOUT
deformation is at least **0.8x** the WITH value. PASS => a shippable recommendation: the
deformation field is a trait-validity liability that buys no genus discrimination. FAIL => a real
trade-off, and deformation carries morphology the traits need.

## What cannot be concluded

Still no surface ground truth. This measures **validity, variation and discriminability**, never
accuracy. And this is an ablation of existing fits, not a re-optimisation: it answers "did
deformation produce these values", not "what would a fitter trained without deformation produce".
