# Pre-registration — Z6: is 0.60 a failure, or the floor?

**Written and committed BEFORE the fits.** Date 2026-08-31. Folder `diagnostics/noise_floor/`.
Corpus: **workers only**.

## 1. Why this and not another intervention

Six hypotheses for the worker residual are closed: worker scan quality (Z1), anterior scale
magnitude (X1), anterior rotation freedom (Z2), unmodelled joint channels (Z3), free-form capacity
(Z4), free-form asymmetry as removable noise (Z5 — a bounded trade, not a fix). Workers sit at
`gen@20/spread ≈ 0.60` against an *instrument* floor of 0.00 under exact correspondence.

That instrument floor is not an achievable target on real scans, and this project has already been
badly misled once by measuring a residual against a number that could not be reached (Z1). So the
question is no longer what else to fix. It is:

> **H: two specimens of the same species are near-identical animals. If the pipeline cannot predict
> one from the other, the residual is per-specimen noise and no fitter change will move it.**

## 2. Corpus

`diagnostics/noise_floor/paired50` — **25 species × 2 conspecific specimens = 50 workers**, drawn
by `z6_build_paired_corpus.py` (seed 0, deterministic) from the 757-worker corpus.

- **n = 50 exactly**, matching `bench50_clean`, so gen/spread's leave-one-out normaliser
  `N/(N-1)` is identical — REFUTE1's R2 flagged that this differs across corpus sizes.
- **One species per genus (25 genera)**, asserted at build time, so the population spread stays
  broad and cannot inflate the ratio. That denominator confound is what made the original
  worker-vs-clean comparison unreadable.
- **Determined species only.** `sp.`, `cf.`, `aff.`, `nr.` are excluded: two specimens labelled
  `Camponotus_sp.` are two *unidentified* members of a genus, not conspecifics, and including them
  would put non-conspecifics into the same-species arm and bias the floor downward.
- Most pairs are consecutive accessions, i.e. same lot and probably nest-mates. This makes the
  estimate **generous** — if anything it flatters the pipeline, which is the safe direction.

Fitted with `D1_PROD.yaml` unchanged, seed 0, `COUPLE_JOINT_BLENDSHAPES=0`, `w_deform_sym` off.

## 3. Endpoints

**PRIMARY — `gen@20/spread`, leave-one-out, with the conspecific present in the training set.**
This is the best case the pipeline can be given: predict an ant from its own nest-mate.

- **≤ 0.35** → the pipeline transfers well between near-identical animals. 0.60 is therefore **not
  a noise floor**; it reflects between-species breadth that 20 modes cannot span, and a
  shape-space capacity project is licensed.
- **≥ 0.55** → a conspecific buys essentially nothing. **Per-specimen noise dominates and ~0.60 is
  approximately the floor.** Fitter-side work on this metric should stop.
- between → partial; report the fraction.

**SECONDARY — leave-PAIR-out**, the same metric with the conspecific *also* removed. The gap
LPO − LOO is how much the model exploits a near-identical animal. Reported for the composed shape
and for `betas` alone (which transferred at 0.2936 on `bench50_clean`).

**ZERO-MODEL CONTROL — replicate ratio.** Mean within-pair distance ÷ population spread, on
rest-space shape. This needs no model at all. **Null = √2 ≈ 1.414 by construction** (two
independent draws from a population sit √2 spreads apart on average); 0 would be identical
specimens. This is a second by-construction null in the manner of Z4's 0.500, and it says directly
whether conspecifics are even recognisable as such in this representation.

## 4. VOIDING checks

1. **Corpus is genuinely paired** — 25 species, 25 genera, exactly 2 specimens each, all
   determined; asserted from `pairs.json` against the fitted mesh names.
2. **Shape space open** — betas mean |z| > 0.5.
3. **Fit not collapsed** — final chamfer within 15% of `bench50_clean`'s ~0.00115. A different
   corpus, so this is a sanity bound, not an A/B; a large excursion means the paired corpus is not
   comparable and the reading is void.
4. **Spread not degenerate** — population spread within 2× of `bench50_clean`'s, so a narrow
   denominator cannot manufacture a high ratio. This is the confound Z0 was built to catch.

## 5. What each outcome licenses

- **Floor (≥ 0.55)** → stop optimising this metric on the fitter. State the achievable target
  explicitly in the reports, re-read the whole Z-series against it, and move the question to
  whether the *downstream morphometrics* are good enough — which is what the workers are for.
  Also licenses reconsidering `w_deform_sym` (Z5) purely as a noise-reduction term against a
  realistic target rather than against 0.00.
- **Not a floor (≤ 0.35)** → the residual is model capacity across the breadth of Formicidae. That
  licenses a shape-space project: rebuild from worker registrations with more modes, and test
  generalisation properly. Z4 already says the free-form field will not supply it, so this would
  have to come from the parametric side.
- **Partial** → report both numbers and the fraction; license neither project on this evidence
  alone.

## 6. Out of scope

Every closed path (X1, Z2, Z3, Z4, Z5), correspondence descriptors, scan degradation ladders, and
any weight sweep. No arm here is an intervention — Z6 measures what is attainable, it does not try
to attain it.
