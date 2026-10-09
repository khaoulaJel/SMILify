# Roadmap — anatomical supervision as the destination, the hinge prior as the baseline

Written 2026-09-07 after V1/V2/V3. This records a deliberate **priority inversion** against the
conclusion V3's own write-up reached, and the reasoning for it.

## The inversion, stated plainly

V3 concluded: the violations are a mild continuous drift, therefore a weak hinge penalty on the
biological bounds should suffice. That reasoning is sound and the experiment is cheap.

**But it is a baseline, not the destination.** A hinge prior can only push values back inside a
box drawn by hand. It cannot make the fit anatomically *correct*, and it risks buying a lower
violation count by compressing legitimate anterior variation — the same trade V3 cannot detect,
because a bound-based flag says nothing about accuracy.

So the hinge prior is run, but the question is never "does 9.1% fall?". It is:

> **Does it reduce pathological anatomy WITHOUT suppressing legitimate biological variation?**

and it is judged against the supervised objective, not against D1 alone.

## Why supervision is the destination

Two established results point the same way and have never been combined:

- **G7**: joint supervision cuts joint error substantially (38.9% → 9.0%), at a surface cost.
- **T4**: surface landmarks carry **2.8×** more biological signal than rig-derived measurements.

And V3 removed the objection that would have made supervision pointless: the anterior is **not**
a θ/β identifiability collapse (p = 0.476), so anatomical information genuinely is missing from
the objective rather than merely unreachable by it.

**The remaining hypothesis is now clean and testable:** the optimiser finds surface-compatible but
anatomically incorrect solutions because the surface data term contains no explicit anatomical
information. That is fixable.

## The result worth aiming for

Not "less bad", but a **Pareto improvement**:

| | surface err | joint err | impossible |
|---|---:|---:|---:|
| D1 | 1.00 | 1.00 | 9.1% |
| + joint (the trade we already have) | 1.15 | 0.55 | ~3% |
| **+ joint + landmarks (the target)** | **1.03** | **0.55** | **~2%** |

The third row would demonstrate the objective is **underconstrained**, not that the model cannot
fit ants. That is a scientific result rather than a patch.

## The blocked step, and its unblock

Everything above needs annotations on the specimens that actually fail. **Only 1 of the 69
pathological specimens is annotated.** `v4_annotation_targets.py` selects a 15-specimen request,
**stratified by severity** rather than worst-first — because V3 showed the severity distribution is
a smooth continuum, so a top-N request would answer "can supervision rescue catastrophes" when the
question is "does it fix the continuum". Healthy controls are included so an intervention can be
shown not to damage fits that were already fine.

Selected (see `v4_annotation_targets.json`; `Eciton_burchellii_CASENT0744558` is already annotated
and therefore a free pilot):

- **A, just over the bound (3)** — Hypoponera_sp. (0.007), Ooceraea_biroi (0.013), Tetramorium_pacificum (0.050)
- **B, medium (4)** — Tetramorium_semilaeve (0.175), Pheidole_oxyops (0.178), Cephalotes_simillimus (0.235), Pheidole_midas (0.238)
- **C, extreme (3)** — Temnothorax_nylanderi (0.399), **Eciton_burchellii (0.603)**, Parasyscia_sp. (0.722)
- **D, healthy controls (5)** — Polyergus_rufescens, Myrmecia_croslandi, Dolichoderus_sibiricus, Pheidole_sp., Typhlomyrmex_rogenhoferi

Per specimen: anatomical joint positions, plus a small set of anterior surface landmarks.

## The experiment sequence, once annotations exist

**V5 — the 2×2.** D1 / +joint / +landmarks / +both. Four measurement families **simultaneously**,
because any one alone is gameable:
1. anatomical error — joint-position error, impossible-trait rate *and severity*
2. surface accuracy — chamfer, vertex/correspondence error
3. biological validity — genus signal, bilateral consistency
4. the trade-off — an explicit Pareto curve, not a single operating point

**V6 — λ sweep** over {0, 0.01, 0.03, 0.1, 0.3, 1, 3} on both supervision terms, plotting surface
error against anatomical error and against impossible-trait rate. The object is a **Pareto
improvement over D1**, not a low violation count at large λ.

**V7 — landmark coverage.** 0/1/2/4/6/8 landmarks. *How much anatomical supervision does SMILify
actually need?* If 4–6 well-chosen landmarks suffice, that is both an elegant result and the thing
that scales — dense joint annotation for every specimen never will.

**V8 — realistic predicted landmarks.** G9's noise model was too friendly (independent isotropic
Gaussian). Rerun with error that is correlated, anisotropic, larger distally, occasionally missing,
and systematically biased. Converts the claim from "we fixed 15 hand-annotated ants" into "an
automatic landmark detector of accuracy X would make this work corpus-wide".

**V9 — from QC to correction.** V3's AUC 0.760 predicts *which* fits fail. The next level is
predicting *how to correct* them: D1 → failure probability → anatomically constrained refinement →
corrected fit. A real extension rather than a filter.

## One direction deliberately deferred: a learned anterior POSE prior

Rather than bounding each ratio independently, learn what anatomically plausible anterior pose
looks like — the correlations among head orientation, mandible and antennal rotations, and the
bilateral relationships — as a multivariate or low-dimensional prior over anterior θ. V3 makes this
plausible precisely because the problem is *not* θ/β identifiability.

**Deferred until after V5**, and the reason matters: learning plausibility from the current fits
means learning it from the very model whose plausibility is in question. The annotated set is what
breaks that circularity.
