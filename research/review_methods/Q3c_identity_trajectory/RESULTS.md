# Q3c results: identity, geometry and skeleton through the optimisation

Rules in PREREGISTRATION.md (H2/S3 on JAB had been seen; everything else was not). Existing fits only,
3 seeds, seed-mean per specimen. Identity I = fit-implied leg accuracy (synthetic: true labels;
real: expert-skeleton proxy). Outputs `out/Q3c_synthetic.json`, `out/Q3c_real.json`, `out/run.log`.

## Trajectories (mean identity; median joint error)

| arm | H0 | H1 | H2 | S2 | S3 |
|---|---|---|---|---|---|
| syn `prod` (no targets) | 0.762 / 7.07% L | 0.831 / 3.74 | 0.834 / 3.52 | 0.855 / 2.62 | **0.859 / 2.43** |
| syn `full_rho0.0` (correct, all stages) | 0.865 / 5.23 | 0.928 / 1.98 | 0.928 / 1.93 | 0.967 / 1.07 | **0.973 / 0.89** |
| syn `full_rho0.2` | 0.811 / 5.95 | 0.885 / 3.29 | 0.887 / 3.21 | 0.872 / 3.06 | 0.881 / 3.16 |
| syn `full_rho0.4` (40% wrong, all stages) | 0.696 / 6.80 | 0.772 / 5.16 | 0.773 / 4.85 | **0.716** / 4.74 | 0.728 / 4.57 |
| syn `hier_rho0.4` (40% wrong, hier only) | 0.696 / 6.80 | 0.773 / 5.16 | 0.773 / 4.86 | 0.814 / 3.64 | **0.820 / 3.49** |
| real `A_prod` | 0.502 / 43.7% WL | 0.538 / 34.3 | 0.546 / 27.9 | 0.564 / 26.0 | 0.567 / 25.1 |
| real `I_cse` | 0.579 / 33.4 | 0.667 / 23.7 | **0.674 / 21.5** | 0.642 / 25.9 | 0.625 / 25.3 |
| real `J_nooffnorm` | 0.502 / 43.7 | 0.538 / 34.3 | 0.547 / 29.0 | 0.543 / 32.0 | 0.539 / 32.7 |

## Pre-registered tests

| test | result | verdict |
|---|---|---|
| H-erode, synthetic: `full_rho0.4` H2 -> S3 identity falls on a majority while chamfer falls; `full_rho0.0` does not | rho 0.4: dI −0.045, down 32/48 (p 0.029), chamfer down 48/48; rho 0: dI **+0.045**, down 0/48 | **SUPPORTED** |
| H-erode, real (pilot, H2/S3 JAB-seen) | `I_cse` H2 -> S3 dI −0.049, down 8/11 (p 0.23), 8 erosion events; S2 -> S3 down 10/11 (p 0.012). `A_prod` dI +0.021 | same signature; n = 11 underpowered over H2 -> S3 |
| Arm-J prediction: lowest final identity of the three real arms | J lower than `A_prod` on 8/11 (p 0.23), mean −0.028 | **INCONCLUSIVE**, direction as predicted |

## What this shows

1. **Geometric fitting is not what destroys identity.** Every arm without active wrong targets gains
   identity during the chamfer-driven surface stages: synthetic `prod` +0.025 (H2 -> S3, 42/48 up),
   `hier_rho*` +0.017 to +0.046, real `A_prod` +0.021. This contradicts the stronger form of the
   second review's thesis ("geometric optimisation overrides anatomical identity"). On this evidence
   the surface objective *helps* identity when it is not fighting wrong correspondence.
2. **Identity erodes when wrong correspondence stays active in the free-form surface stages.** The
   erosion is concentrated at the transition into Stage_2 (synthetic rho 0.4: −0.057, 34/48,
   p 0.006; rho 0.2: 29 erosion events), with partial recovery in Stage_3. Releasing the same wrong
   targets after H2 (`hier_rho0.4`) turns erosion into recovery: 0.773 -> 0.820 instead of 0.728,
   and joint error 3.49 instead of 4.57% L.
3. **Real CSE fits follow the synthetic wrong-target signature.** Identity peaks at H2 (0.674 vs
   production 0.546) and erodes in the surface stages to 0.625; joint error rises from 21.5 to 25.3%
   WL while production's falls to 25.1. That is what synthetic `full_rho0.2-0.4` does, consistent
   with real targets being ~25-50% wrong on many specimens (Q3a A2b).
4. **Identity and skeleton still dissociate on real scans.** At S3, CSE keeps more identity than
   production (0.625 vs 0.567) with the same joint error. Arm J loses skeleton accuracy H2 -> S3
   (+3.7% WL) with nearly flat identity (−0.008): its "best surface, worse skeleton" effect is
   mostly a skeleton phenomenon, not an identity one (inconclusive at n = 11).

## Consequence for the intervention

The policy that minimises error depends on target quality: correct targets should stay on in every
stage (0.89 vs 1.77% L), wrong targets should be released after skeleton placement (3.49 vs 4.57).
No constant choice is right for both, and real specimens sit on both sides. The intervention family
this points to is a **trust schedule**: correspondence used strongly while the skeleton is placed,
then annealed or re-estimated as the fit improves, and gated by a per-specimen quality estimate.
That is M04 (re-estimation, which can repair targets) and Q3d (constant-weight sweep, which tells us
whether weight alone can do it). Q3d is running.

## Limits

Fit-implied identity is read off the fitted surface, not the optimiser's internal correspondence.
The real trajectories rest on proxy labels (leg-level gate 0.892) and n = 11. Corruption on
synthetic is coherent wrong-leg only; the non-leg type is added in Q3d.
