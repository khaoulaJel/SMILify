# Q3c: does anatomical identity erode during optimisation while geometry improves?

Written 2026-10-07 before any trajectory was computed. Motivated by the second review (§2, §12) and by
Q3a A2, which measured fit-implied leg identity at two stages (H2, S3) on JAB. **Disclosure:** those
two JAB stage values had been seen (CSE identity fell H2 -> S3 on 9/11). The three other stages, the
synthetic trajectories and the arm-J prediction are unseen.

## Quantities, per specimen, per stage (H0_body, H1_legs, H2_joint, Stage_2, Stage_3), seed-mean of 3

- **Identity I(t)**: fit-implied leg accuracy. Points on the target surface carry a label; each is
  matched to the nearest fitted vertex; correct if that vertex's template leg equals the point's leg.
  Synthetic: TRUE labels (4000 points / specimen), plus geodesic error / sqrt(area). Real: expert-
  skeleton proxy labels (gate-validated; tr/fe/ti/ta, pt -> ta, coxa excluded), as in A2.
- **Geometry G(t)**: symmetric chamfer between fitted vertices and target (same code for both).
- **Skeleton S(t)**: median FK joint error (synthetic: / body-axis length; real: % WL, JAB's own
  per-stage scores).

## Data (existing fits only, no new fitting)

Synthetic: Q3b S1 fits on P48, arms prod, full_rho{0, 0.2, 0.4}, hier_rho{0, 0.2, 0.4}.
Real: JAB arms A_prod, I_cse, J_nooffnorm.

## Pre-registered hypotheses and readings

- **H-erode (primary):** in the surface stages (H2 -> S2 -> S3), identity falls while chamfer falls.
  Test per transition and arm: paired sign test over specimens on dI < 0 and on dG < 0; an
  **erosion event** for a specimen is dI < -SD_seed(I) together with dG < 0.
  - SUPPORTED on synthetic if, for full_rho0.4 over H2 -> S3, dI < 0 on a majority (sign p < 0.05)
    with dG < 0 on a majority (sign p < 0.05), AND full_rho0.0 does NOT show it (sign p >= 0.05 or
    dI > 0). The rho = 0 arm is the control: with correct targets, identity should not erode.
  - On real (pilot, JAB-seen at H2/S3): reported with the same test for I_cse and A_prod.
- **Arm-J prediction (unseen):** J_nooffnorm, which has JAB's best chamfer and a worse skeleton, has
  the LOWEST final identity of the three real arms (paired vs A_prod, sign test). If J's identity is
  not lower than A_prod's, "the score can lie" is a skeleton-only phenomenon, not an identity one.
- **Dissociation figure:** I(t) vs G(t) trajectories per specimen; reported, not tested.

## Not claimed

Fit-implied identity is a nearest-vertex reading of the fitted surface, not the correspondence the
optimiser used; on real scans it rests on proxy labels (gate 0.892 leg-level).
