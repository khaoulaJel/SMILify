# T3 pre-registration — turning today's findings into a measurement recipe

Fixed **before** running. T1/T2 produced diagnosis and changed nothing the pipeline outputs. T3 is
the conversion step: three interventions, each derived from a specific finding, each a few lines of
code, judged on one criterion that needs no annotation.

## Interventions

| | change | finding it comes from |
|---|---|---|
| **A** | report ML, SL, FL as the **mean of left and right**, not the right side alone | bilateral asymmetry is 6.8-10.8% on these traits (T1) — that is discardable noise |
| **B** | **drop specimens with biologically impossible traits** | 9.1% of fits emit them (T1) |
| **C** | **weight by chamfer** (drop the worst quality decile) | registration error genuinely predicts trait failure at +0.465 (T1, corrected) |

**Explicitly NOT tried: weighting by bilateral asymmetry.** It is inverted on the failure class
(impossible fits are *more* symmetric, p = 0.002), so it would preferentially keep broken fits.
This is the asymmetry-inversion finding paying for itself.

Combinations tested: baseline, A, B, C, A+B, A+B+C.

## The criterion — genus discriminability, and why not CV

The obvious criterion, "lower within-genus variation", is **gameable**: any intervention that
compresses all variation toward a constant minimises it while destroying the measurement. So the
criterion is the **F-ratio of between-genus to within-genus variance** across the 27 genera with
n >= 8 (386 specimens). Compression shrinks numerator and denominator together, so it cannot be
gamed; the ratio rises only if genera separate *relative to* the noise within them.

Computed per trait ratio (HW/HL, SL/HL, ML/HL, PetL/WL) and averaged. **95% CIs by bootstrap over
genera** (2000 resamples), because an intervention that changes n must not be scored as if it did not.

**Ship rule, fixed now:** an intervention ships only if its mean F-ratio improves and **the bootstrap
95% CI of the paired difference excludes zero**. A nominal improvement inside the CI does not ship.

## Secondary check — pre-specified taxonomic contrasts

Guards against fitting noise structure. Taken from ant morphology, **not** from this pipeline's own
`genus_table.json` (which would be circular):

- **C1** HW/HL: *Cephalotes* > *Odontomachus* — broad armoured head vs elongate trap-jaw head.
- **C2** HW/HL: *Pheidole* > *Odontomachus* — Pheidole majors are broad-headed.
- **C3** SL/HL: *Camponotus* > *Cephalotes* — long formicine scapes vs short scapes recessed in scrobes.
- **C4** SL/HL: *Formica* > *Cephalotes* — same contrast, independent genus.

These are a sanity check on direction, not the ship rule. Confidence differs between them (C1 is
textbook; C2 depends on caste composition, which we do not control) so **no single contrast can veto
an intervention**; a majority reversing would.

## Voiding checks — fail closed

- A genus drops out if fewer than 8 specimens survive an intervention; the genus count is reported
  with every result. Below 15 genera the comparison is void.
- Trait ratios are computed by the same `trait_extract.traits()` call for every arm; no arm gets a
  separately tuned pipeline.
- Any arm that retains fewer than 60% of specimens is reported but not shipped, regardless of F.

## What ships

The winning combination is committed into `trait_extract.py` as the default, versioned against the
landmark file hash. That is the deliverable — a changed pipeline output, not another finding.
