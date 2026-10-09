# Q3d results: identity weight x target quality

Rules in PREREGISTRATION.md. 51 new fits + reused S1 arms (lambda 0 = `prod`, lambda 1 = `full`),
3 seeds, P48, final stage. Outputs `out/Q3d_readout.json`, `out/readout.log`.

| targets | lambda 0 | 0.1 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| correct | 2.43 / 0.859 | 1.26 / 0.956 | 1.14 / 0.964 | 1.02 / 0.969 | 0.89 / 0.973 | **0.79 / 0.975** |
| 20% wrong-leg | **2.43** / 0.859 | 3.17 / 0.908 | 3.35 / 0.883 | 3.24 / 0.875 | 3.16 / 0.881 | 2.92 / 0.890 |
| 40% wrong-leg | **2.43** / 0.859 | 5.23 / 0.797 | 5.44 / 0.727 | 4.92 / 0.718 | 4.57 / 0.728 | 4.34 / 0.741 |
| 20% non-leg | 2.43 / 0.859 | 2.29 / 0.944 | 2.25 / 0.951 | 2.03 / 0.957 | 1.96 / 0.960 | **1.92 / 0.962** |

(median joint error % body-axis length / mean fit-implied leg identity)

## Pre-registered readings

- **Primary, SUPPORTED by the letter:** lambda* = 2 (correct), 2 (20% wrong-leg), 0 (40% wrong-leg);
  ordering holds; lambda* beats lambda 1 at both corrupted levels (38/48, p 6e-5; 41/48, p 6e-7).
- **Useful window at 20% wrong-leg: NONE.** No lambda > 0 beats lambda 0 (best 26/48, p 0.67). The
  curve is U-shaped: small weights (0.1-0.5) are worse than both extremes.
- **Error type:** at matched 20%, non-leg errors leave correspondence clearly useful (lambda 2: mean
  2.00 vs 2.96% L, 34/48, p 0.006); wrong-leg errors remove all benefit.

## What this shows

1. **Weighting cannot rescue wrong-leg correspondence.** S1's break-even is not a weight artefact.
   The primary "SUPPORTED" is driven by a binary switch (use strongly vs not at all), and lambda* = 2
   sits on the grid edge for correct and 20%-wrong targets, so it is not a tuned optimum.
2. **The damaging error is specifically leg identity.** Off-leg errors are absorbed; wrong-leg
   errors are not. That is exactly the real CSE error type (Q3a: wrong leg number 0.10-0.37, wrong
   side 0.00-0.05).
3. **Identity and skeleton dissociate again.** With 20% wrong-leg targets the fitted surface is on the
   right leg *more* often than in production (0.875-0.908 vs 0.859) while the joints are worse: a
   minority of wrong-leg patches is enough to drag the pivots.
4. **With correct targets more weight keeps helping** (0.79% L at lambda 2, monotone), so the value
   of correct correspondence is not yet saturated.

## Consequence

The remedy has to act on leg identity itself before or during fitting -- repairing or removing
wrong-leg patches -- not on how much the fitter trusts correspondence overall. Candidates that do
that: re-estimating correspondence from the current fit (M04); a leg-level consistency check that
drops patches whose retrieved leg disagrees with the leg the current skeleton places there; and a
correspondence model whose leg identity survives real articulation (M01 / articulation-matched
training).
