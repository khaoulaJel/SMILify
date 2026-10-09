# Q3d: how much identity weight does the fitter need, and how does that depend on target quality?

Written 2026-10-07 before any Q3d fit. From the second review (§4), and S1's dose curve.

## Design (P48, D1_PROD recipe, correspondence term in all stages, 3 seeds)

Weight lambda_corr (`--w_cse_correspondence`, both stages) in {0, 0.1, 0.25, 0.5, 1, 2} crossed with
target quality:

| targets | source |
|---|---|
| correct | S1 `targets_rho0.0` |
| wrong-leg 20% / 40% | S1 `targets_rho0.2` / `targets_rho0.4` (coherent adjacent-leg patches) |
| **non-leg 20%** (new) | 20% of the 24 (leg, segment) units redirected to the true position of the nearest TRUNK vertex (the error type the bad real specimens show: 17-38% of leg targets off the legs) |

Reused, same code state: lambda 0 = S1 `prod`; lambda 1 x {correct, wrong20, wrong40} = S1 `full`.
New: 4 lambdas x 3 qualities + 5 lambdas x non-leg = 17 configurations x 3 seeds = 51 fits.

## Endpoints

S3 median joint error / body-axis length (S); fit-implied identity with true labels and geodesic
error (Q3c code); chamfer (G); free channels.

## Pre-registered readings

- **Optimal weight shifts with quality (primary):** for each target quality, lambda* = the lambda with
  the lowest seed-mean S3 joint error. Predicted: lambda*(correct) >= lambda*(wrong20) >=
  lambda*(wrong40). Supported if the ordering holds AND at each of the two corrupted qualities the
  joint error at lambda* is lower than at lambda = 1 (paired sign test, p < 0.05). Refuted if lambda*
  is the same for all qualities.
- **Useful window:** at wrong20, is there any lambda > 0 beating lambda = 0 (sign p < 0.05)? If yes,
  partially wrong correspondence is still worth injecting at the right weight; if no, S1's
  break-even is not a weight artefact.
- **Error type matters:** at matched rate (20%), compare wrong-leg vs non-leg at each lambda.
  Reported; predicted that non-leg errors are less harmful (they pull legs toward the body, which the
  chamfer term opposes), but not used as a decision.
- **Identity vs geometry:** at each (quality, lambda), the S3 identity and chamfer are reported
  together, to show whether a higher identity weight buys identity at a measurable geometric cost.

All selection on synthetic data; nothing here is ever tuned on JAB.
