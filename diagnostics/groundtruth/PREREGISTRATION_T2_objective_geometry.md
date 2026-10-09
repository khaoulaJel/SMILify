# T2 pre-registration — does the data term reward inflating thin structures?

Written **before** the run, per the standing rule. Bounds, predictions and voiding checks fixed here.

## The claim under test

T1 found that fits with biologically impossible mandible ratios have **lower** chamfer residual
(Spearman −0.356 between registration error and |log deviation of ML/HL|), and are **more**
bilaterally symmetric than sound fits (4.7% vs 11.0% ML asymmetry). The proposed explanation:

> The data term samples the target surface roughly uniformly by area. The mandible is a small
> fraction of body area, so **inflating it improves local surface coverage while costing almost
> nothing globally** — the objective actively rewards anatomically impossible geometry on thin
> structures.

If true, this is a property of the method class (surface fitting to appendage-bearing bodies), not
of this implementation, and it is consistent with G6/G7 finding no term that constrains anatomical
position.

## Predictions, fixed before the run

**P1 — the mandible is a negligible share of the objective.**
Fraction of target scan points whose nearest fitted vertex lies in the mandible region is **< 5%**.
REPORT §4 puts tibia+tarsus+pretarsus at 2.4% of a leg chain's area, so this is the expected scale.
*If the mandible share exceeds 15%, the hypothesis is dead and T1's explanation is withdrawn.*

**P2 — inflation buys local fit.**
Spearman(ML/HL, mandible-region residual) is **negative**, with |rho| >= 0.15 at n >= 600.
*A rho that is positive, or |rho| < 0.15, fails P2.*

**CONTROL — the effect must be specific to thin structures.**
Spearman(|log deviation of WL from the corpus median|, mesosoma-region residual) must NOT be
negative at |rho| >= 0.15. The mesosoma is the largest-area part, so it should show no reward for
distortion.
*If the control is also negative, the effect is not about thin structures; T1's explanation is
withdrawn regardless of P1 and P2.*

## Voiding checks — fail closed

- A specimen whose target mesh is missing, unreadable, or has < 1000 vertices is **dropped and
  counted**, never imputed. If fewer than 600 specimens survive, the run is void.
- Region assignment uses the same `dominant` skinning-weight part map as `measure.py`, asserted
  against the 10235-vertex production topology. A vertex-count mismatch voids the run.
- Target and fit live in different coordinate frames; residuals are computed after the same rigid
  alignment used to produce `registration_error.json`, and the recomputed global residual must
  correlate with the stored one at Spearman >= 0.8 or the run is void — the alignment is then wrong
  and every per-region number would be meaningless.
- Residuals are normalised per specimen (divided by that specimen's own centroid size), since the
  corpus has arbitrary per-specimen scale.

## What each outcome licenses

| P1 | P2 | control | conclusion |
|---|---|---|---|
| pass | pass | null | **Named, quantified defect in the data term.** First actionable fix direction for the fitter: region-weighted sampling. |
| pass | pass | also negative | Effect is not thin-structure-specific. T1's explanation **withdrawn**; report the correlation without the mechanism. |
| pass | fail | — | The inversion is likely a **landmark** artefact, not a fitter one. Blocked on the expert-adjudicated mandibular apex (protocol risk R1). |
| fail | — | — | Hypothesis dead on arrival; T1's explanation **withdrawn**. |

No outcome is unpublishable, and the withdraw conditions are written down before the numbers exist.
