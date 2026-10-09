# Pre-registration — G10: does the joint term improve the MORPHOMETRICS?

**Written and committed BEFORE the runs.** 2026-09-01.

## 1. Why this is the experiment that decides everything

G7–G9 measured *joint accuracy*. The project exists to produce *trustworthy measurements*. Those
are not the same thing, and this investigation has been misled by intermediate metrics at least
four times (the stale 0.98 anchor; the synthetic reliability certificate; the unconstrained oracle,
twice). **A 62% joint-error reduction that leaves trait reliability at R ≈ 0 is a hollow win.**

## 2. The circularity that makes the naive design vacuous

`measure.py`'s length traits are **bone lengths**, i.e. distances between adjacent joints. Fitting
joints to ground-truth positions and then asking whether bone lengths match ground truth is
answered by construction — it measures the optimiser's convergence, not the pipeline's value.

**The design must therefore never supervise with the same joints it is scored against.** Two
consequences, both adopted:

- **Supervision is NOISY.** Targets are perturbed joints (the realistic predictor case from G9),
  scored against the *true* joints. The noise breaks the circularity and is simultaneously the
  deployment scenario.
- **σ = 0 is reported as an explicitly labelled circular upper bound**, never as a result.

## 3. The comparator that matters is not the production fit

If a predicted skeleton exists, the trivial alternative is to **read the measurements straight off
it** and not fit at all. So a three-way comparison at every noise level:

| arm | what it is |
|---|---|
| **P — production fit** | today's pipeline |
| **T — targets read directly** | the trivial alternative: measure the predicted skeleton |
| **J — joint-constrained fit** | the proposal |

**J must beat both.** Beating P alone is insufficient: if J ≈ T, the fit adds nothing over
believing the predictor, and the honest recommendation would be to skip the fitting entirely for
morphometric purposes. This is the trait-level form of G9's pass-through test, and it is the
sharpest question available.

## 4. Endpoints

Per block (`mandible`, `leg_prox`, `leg_distal`, `antenna`, `gaster`) and overall, against
human-annotated ground truth, in Mosimann log-shape-ratios exactly as G2 computed them:

- **PRIMARY: R**, the correlation between fitted and true proportion across the 14 specimens.
  G2's production values are the baseline: median R **−0.048**, only mandible above 0.5.
- **SECONDARY: median |bias|**, since G2 found bias unstable at this n and it is what decides
  whether population-level claims survive.

## 5. Decision rule, fixed now

Judged at σ = 15% (a realistic predictor per G9, where the fit already denoises):

- **PASS** — J's median R exceeds production's by **≥ 0.25 in absolute terms** AND exceeds T's.
- **PARTIAL** — J beats production but not T. The fit adds nothing over the predictor; report the
  measurement improvement as attributable to the skeleton, not to the fitting.
- **FAIL** — J does not beat production. The joint term improves joints without improving
  measurements, and G7–G9 describe an intermediate metric rather than the goal.

## 6. Why the fix is principled rather than ad hoc, and why that matters here

`smal_fitter/fitter.py:311` shows the **single-view** fitter already has a joint data term
(`objs["joint"] = w_j2d * mse(rendered_joints, target_joints)`). `fitter_3d/trainer_moonshot.py`
has none. SMPLify and SMALify — the lineage this code is named for — use joint reprojection as
their *primary* data term; the 3D branch replaced image observations with mesh-to-mesh chamfer and
dropped the joint constraint with them.

So this is not a new regulariser being tuned until it works. **It restores the lineage's primary
data term to a branch that lost it**, which is why it is expected to help and why its failure would
be the more surprising outcome. Stating that in advance is what makes a positive result
interpretable rather than a fishing success.

## 7. Out of scope

The predictor itself; structured (correlated) target noise, which is a separate pre-registered
follow-up; any change to the shipped recipe.
