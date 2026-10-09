# V3 — pre-registration. Is the anterior pathology a pose/shape IDENTIFIABILITY degeneracy?

Written 2026-09-07 BEFORE the run. Results appended after.

## Why not the θ/β factorial (experiment A) — a hard blocker, checked first

A requires oracle θ and β on the pathological specimens. **Only 1 of the 69 flagged specimens is
human-annotated** (`Eciton_burchellii_CASENT0744558`; 14 annotated total, 12 expert-corrected).
The factorial would run at n = 1. **A is not runnable and is not attempted.** Obtaining
annotations for flagged specimens is the cheapest thing that would unblock it.

V3 runs the identifiability experiment instead. It needs **no ground truth whatsoever**, and it
tests a deeper claim: not "which of pose or shape is wrong" but "can the surface distinguish them
at all in the anterior".

## The hypothesis

If, restricted to anterior vertices, the surface changes reachable by anterior POSE and those
reachable by SHAPE span nearly the same subspace, then a surface objective cannot tell the two
apart, and the fitter is free to trade one for the other while keeping chamfer flat. That is a
parameter-identifiability problem, not an optimiser problem, and it would explain why violations
concentrate on ML/HL (44) and HW/HL (33) and why they survive removing every per-vertex offset (V2).

## Method

Anterior joints: `b_h, ma_r, ma_l, an_1_r/l, an_2_r/l, an_3_r/l` (9).
**Control region — gaster**: `b_a_1..b_a_5` (5). The control is what makes this a test rather than
a description: a degeneracy present everywhere is uninformative.

At each sampled specimen's **production parameters** (degeneracy is pose-dependent, so it must not
be evaluated at the template), build by central finite differences:

- `S_theta` = ∂V_region / ∂θ_region (3 axes per joint)
- `S_beta`  = ∂V_region / ∂β_k (25 PCs)

restricted to that region's vertices, orthonormalise each, and take the **principal angles**
between the two spans. Principal angles between subspaces are invariant to how each basis vector is
scaled, so the differing units of radians and β do not matter.

**NUMERICAL CHECK, built in:** every quantity is recomputed at two step sizes (1e-3, 1e-2). If the
smallest principal angle moves by more than 2°, the differencing is not converged and the run is
void.

Sample: **all 69 flagged** plus **69 unflagged** matched by drawing from the same corpus.

## Also measured (no ground truth needed)

- **severity distribution** of the 69: how far past the bound, binary vs continuous
- **per-PC anterior table**: each β PC at ±2σ, its effect on ML/HL and HW/HL
- **β- vs θ-induced anterior JOINT displacement**: does changing shape move anterior joints as much
  as changing pose does?
- **violation predictor**: L2 logistic regression, flagged ~ (25 βs + anterior θ), 5-fold
  cross-validated AUC, against a label-permutation null

## Pre-registered bars

**H1 — the anterior is degenerate.** Median smallest principal angle between the anterior pose- and
shape-subspaces is **< 20°**, and is at least **10° smaller** than the gaster control's. PASS =>
pose and shape produce near-interchangeable anterior surface change; the surface objective cannot
identify them, and the fix must be an added observation, not a better optimiser. FAIL => the
anterior is identifiable and the pathology is something else.

**H2 — degeneracy tracks the pathology.** Flagged specimens have smaller minimum principal angles
than unflagged ones, one-sided Mann–Whitney **p < 0.05**. PASS => the degeneracy is not a uniform
property of the model but concentrates in exactly the fits that go wrong.

**H3 — violations are predictable from the parameters.** Cross-validated AUC **>= 0.75** with the
permutation null at ~0.5. PASS => a small set of β/θ directions carries the pathology, giving a
tiny attack surface and a free QC filter for the corpus.

## What cannot be concluded

Principal angles are a **local** (first-order) property at the fitted parameters; they bound what
gradient descent can distinguish nearby, not the global landscape. Degeneracy would explain the
pathology, not prove it caused any individual fit. And there is still no surface ground truth, so
"impossible trait" remains validity, not accuracy.
