# Q2: oracle information budget on synthetic scans where correspondence is genuinely hard

Written 2026-10-06, before S2 (Q3b) produced any number, so the regime-selection rule below cannot
be tuned to S2's outcome. Synthetic only (PROTOCOL §4).

## Question

Starting from a scan regime where the learned correspondence fails the way it fails on real scans,
which single piece of anatomical information, restored perfectly, collapses the skeleton error, and
which does not? This decides which Q4 interventions get built (README).

## Regime (fixed rule, applied to S2's output)

Real-scan target: per-point leg accuracy, proxy-corrected median over the 11 JAB specimens = **0.695**
(observed 0.608; Q3a A2b), wrong-number errors dominating wrong-side errors.

R* = the S2 condition (including REAL*, excluding DEG_* alone) whose mean per-point leg accuracy is
closest to 0.695 among conditions with wrong-number >= 2x wrong-side. Ties (within 0.02) go to the
condition with fewer modified factors. If no condition meets the error-type requirement, R* =
REALALL. A fresh 48-specimen set is generated for R* with new seeds (779000+), written as .obj, so Q2
is not scored on the specimens S2 used to select it.

## Arms

All arms start from the same deployed baseline and differ in exactly one restored quantity. The
"restoration" mechanism follows JAB's oracle protocol (`diagnostics/joint_alignment_benchmark/tools/oracle.py`):
from the baseline's final parameters, re-optimise the D1_PROD Stage_3 objective for 800 iterations
(Adam, lr 0.005) with one ground-truth quantity added.

| arm | restored information | mechanism |
|---|---|---|
| B0 | none, no correspondence | D1_PROD fit of R* (reference) |
| B1 | none, deployed correspondence | D1_PROD + CSE (cycle, keep 0.5, all segments), as JAB I_cse; the starting point of every O-arm |
| O0 | control | re-optimisation with nothing added; must not move error (voiding control) |
| O-part | true part membership | CSE retrieval restricted to the template vertices of each scan point's TRUE (leg, segment); within-part choice still from the network; fed as the correspondence term |
| O-corr | true dense correspondence | every template vertex's true position as the correspondence term |
| O-joints | true joint positions | λ · mean squared FK-joint error, λ = 0.03 (JAB O1 value, not re-tuned) |
| O-pose | true articulation | joint_rot set to ground truth, then re-optimised |
| O-shape | true shape | betas + log_beta_scales + betas_trans set to ground truth, then re-optimised |
| O-all | all of the above | |

3 seeds per arm (DEVIATIONS D1.3).

## Endpoints (separate families, PROTOCOL §1)

- **S (primary):** per-specimen median FK joint error / body-axis length L, overall and by region.
- **C:** post-fit leg-level misassignment (C13 scorer) and geodesic correspondence error.
- **G:** chamfer. **M:** HW.

Each O-arm is reported as the fraction of the B1 -> O-all skeleton-error gap it closes, per region,
with a specimen-bootstrap 95% CI; paired comparisons vs O0 use sign + Wilcoxon + paired t.

## Decision rule for Q4 (fixed now)

- **M01 (skeleton-conditioned correspondence)** is built only if O-joints or O-pose closes >= 40% of
  the gap AND that is more than O-part closes. Rationale: M01's premise is that skeletal information
  is what is missing.
- **M05 (articulation canonicalisation)** is built only if O-pose closes >= 40% of the gap.
- **M04 (iterative correspondence-registration refinement)** is built only if O-part or O-corr closes
  >= 40% of the gap (better correspondence is worth having) AND S1 supports H-absorb or H-surfterm
  (how correspondence is consumed matters).
- **M02 (uncertainty-aware objective)** is built only if O-corr closes >= 40% AND S1 supports
  H-absorb (wrong targets hurt, so down-weighting them could help).
- If no O-arm closes >= 40%, the conclusion is that no single restored quantity is sufficient, and
  Q4 is reconsidered from O-all's composition rather than built by default.

## Not claimed

Post-hoc re-optimisation from the deployed fit measures what each quantity is worth *given* the
deployed basin, not from scratch; that limitation is shared with JAB's O1 and stated. Synthetic
only; restoring a quantity perfectly says what it is worth, not that it is obtainable.
