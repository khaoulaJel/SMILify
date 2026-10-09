# Pre-registration — A1 basin-map leg_acc curves, old vs corrected sampler (2026-08-25)

Written before `run_basin_map_corr_rho06_20260825.sh` / `plot_A1_basin_map_leg_acc_curves_20260825.py`
have been run against real fits. States what pattern would mean what, ahead of seeing the numbers —
same discipline as B3 and the reason #9's "targets hard cases" over-read got caught before it
shipped.

## The question

The basin-map perturbations (random/proximal/distal/coherent) are noise added ON TOP of each
corpus's own GT poses — the corrected sampler only changes how those underlying GT poses
themselves are generated (AR(1)-coupled vs i.i.d.), not the perturbation-construction logic
(`generate_structured_perturbation_init.py`), which samples its own independent noise per
condition/magnitude regardless of corpus. So the open question is whether the corpus's own
pose-coupling structure changes which perturbation *types* the D1 optimizer finds recoverable.

## Prediction 1 (expected, orthogonality hypothesis)

**The relative ordering and approximate gap sizes between conditions
(coherent > distal > random > proximal at matched magnitude, per the original basin map) will be
preserved under the corrected sampler.** Concretely: no reordering of which condition scores
best/worst at any magnitude, and most old-vs-corrected deltas fall within ±0.03–0.05 leg_acc
(roughly the specimen-to-specimen noise band already visible in the original run's per-specimen
spread).

**Why this is the theoretically expected outcome:** the D1 optimizer's local recoverability
around a given true pose is a property of the *fitter's* loss landscape and the *perturbation's*
structure (finding #3), not of how the corpus's GT was generated. The optimizer has no visibility
into whether the target pose it's fitting toward came from an i.i.d. or AR(1)-coupled sampler —
it only sees the resulting mesh. This prediction is also the one consistent with A1's own
disentangling-probe result already in hand: strong lag-1 coupling (cosine 0.05→0.52) left the
coxa-only representational gap statistically unchanged (50.03° vs 50.46°), i.e. the corrected
sampler measurably does NOT touch the mechanism finding #3 identified. If Prediction 1 holds, it's
the same conclusion from a second, independent angle: the sampler fix (realism) and the basin-map
mechanism (recoverability-by-structure) are orthogonal, not two views of the same thing.

## Prediction 2 (alternative, interaction hypothesis)

If curves shift substantially — any condition's ordering changes relative to the others, or
old-vs-corrected deltas exceed ~0.05–0.1 leg_acc broadly (not just 1–2 specimens, checked via
sign test + Wilcoxon across the 12 specimens per condition, per this investigation's standing
discipline) — that is a genuinely new finding not implied by anything established so far, and
would mean the corpus's own pose-coupling interacts with perturbation recoverability. It would
need its own follow-up before being trusted (e.g.: does `proximal` specifically become less
damaging because AR(1)-coupled GT poses already look more "coherent" at baseline, blurring the
coherent/proximal distinction that made the original result so clean?) — not written up as a
headline result off the first read, per the #9 retraction precedent.

## What to do with either outcome

Prediction 1 (expected): report it plainly as a confirmatory null — strengthens confidence that
A1's sampler fix and the Phase 10 architecture work (A3, B) are addressing genuinely separate
problems, not duplicating effort.

Prediction 2 (surprise): do not fold it into the main A1 write-up as a settled finding. Flag it,
report both the full-sample and outlier-excluded deltas, and treat it as a new open question for
a follow-up probe, not as evidence for or against the corrected sampler's validity.
