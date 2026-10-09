# Pre-registration — G8: does a joint-position term propagate anatomically, or memorise its targets?

**Written and committed BEFORE the runs.** 2026-09-01. `diagnostics/groundtruth/`.

## 1. The claim under test

G7 showed that adding `λ · joint_target` to the production objective cuts joint error from 38.9% to
27.5% for 19% more chamfer, and to 9.0% at 2.3×. But the residual was measured on **the same joints
supplied as targets**, so it demonstrates *reachability*, not *generalisation*, and it used a
complete human-annotated skeleton, which does not exist at inference.

> **H: constraining a SUBSET of joints improves the joints that were NOT constrained.**

If true, the term pulls the model into a globally correct anatomical configuration and a partial or
approximate skeleton suffices — the fix is practical and a predictor need only be roughly right on
some joints. If false, the term memorises whatever it is given, and a deployed predictor would have
to be near-complete and near-exact, which is a far harder engineering problem than the one G7's
result appears to license.

**The plausible mechanism, which the design also tests:** the 25 betas couple every bone length
simultaneously. Supervising proximal joints could pull betas toward the specimen's true shape, and
the distal joints would then follow *without being supervised at all*. If held-out improvement is
real and is mediated by betas, that is the mechanism.

## 2. Design

Corpus: the 14 annotated specimens. Objective and optimiser exactly as G7 (`production objective +
λ · joint_target` on the supervised subset only, `betas` projected to |z| ≤ 1, `deform_verts` free,
initialised at production). λ ∈ {0.03, 0.1} — G7's knee and its strong-effect point.

**Four partition schemes**, because the answer may depend on *which* joints are supervised:

- **A — anatomical (the deployable case).** Supervise the body axis and proximal leg joints
  (coxa, trochanter, femur); hold out distal (tibia, tarsus, pretarsus) and the entire anterior
  (mandibles, antennae). This is the realistic scenario: proximal joints are large, well-supported
  by geometry, and what a predictor would get right. The held-out set is precisely what G1 showed
  the pipeline fails on.
- **B — random 50%, 5 folds.** Unbiased over which joints happen to be easy.
- **C — sparsity sweep**, 10 / 25 / 50 / 75% supervised, 3 folds each. How many joints are
  actually needed?
- **D — NEGATIVE CONTROL: scrambled targets**, 3 folds. Supervise 50% of joints but with the target
  positions **permuted across joints within the specimen**, so the term is equally strong and
  equally regularising but anatomically meaningless.

## 3. Endpoints

**PRIMARY: median residual on the HELD-OUT joints**, % of mesosoma length, paired against the
production fit's residual on those same joints.

**Reported alongside, and required for the primary to be interpretable:**
- supervised-joint residual (must fall — if it does not, the term is not acting);
- chamfer (the cost);
- `betas` distance from production (the mechanism);
- **scheme D's held-out residual** (see §4).

## 4. Decision rule, fixed now

- **PASS — the term propagates.** Held-out residual improves by **≥ 20% relative** to production in
  scheme A, with ≥ 10/14 specimens improving (sign test p < 0.05), **AND scheme D shows no such
  improvement.**
- **FAIL — the term memorises.** Held-out improvement < 10% relative, or indistinguishable from
  scheme D.
- **PARTIAL** — between those, or scheme A passes while random splits do not.

**Scheme D is voiding.** If scrambled targets improve held-out joints as much as true targets, then
the effect is extra regularisation rather than anatomy, and the primary result is **void** however
large it looks. This is the control that distinguishes "an anatomical constraint propagates through
the kinematic chain and the shape space" from "any additional term stabilises the fit", and without
it the experiment cannot support the claim.

## 5. What each outcome licenses

- **PASS** → a predicted skeleton need only cover a subset of joints and need not be exact. Build
  the predictor (the 14 annotations seed it; OmniTrax-style 3D joint prediction is the precedent),
  and the deployment path from G7 is real.
- **FAIL** → G7's headroom is not reachable without a near-complete, near-exact skeleton. Report
  G7 as a demonstration of what the objective is missing rather than as a fix, and the practical
  route becomes acquiring more annotation rather than better prediction.
- **PARTIAL** → report which joint groups propagate and which do not; that map is itself the
  deliverable, and it tells a predictor which joints are worth getting right.

## 6. Out of scope

Whether this improves the morphometric traits (that is the separate follow-up G7 named), the
predictor itself, and any change to the shipped recipe.
