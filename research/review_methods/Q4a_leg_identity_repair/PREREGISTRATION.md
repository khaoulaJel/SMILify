# Q4a: repairing or removing wrong-leg correspondence before it reaches the surface stages

Written 2026-10-08, before any Q4a run and before any Q2 oracle result. Built only if Q2's O-part or
O-corr closes >= 40% of the gap (DEVIATIONS D9); the design is fixed now so that the oracle result
cannot shape it.

## Motivation (all measured)

Wrong-LEG targets are the damaging correspondence error (Q3d), no trust weight rescues them (Q3d),
they erode identity specifically once the free-form surface stages start (Q3c), and the per-specimen
wrong-leg fraction predicts whether deployed CSE helps (Q2x, break-even ~25%). Real CSE targets are
wrong by leg number, rarely by side (Q3a). So the intervention must act on leg identity of the
targets, between skeleton placement and the surface stages.

## Literature basis

- Verification of putative correspondences against the current model estimate is the standard
  remedy for outlier matches in registration: ICP-style rejection, RANSAC-type geometric
  verification, and its learned successors (spectral / consistency-based matching).
- Alternating correspondence and model estimation, with correspondences re-estimated from the
  current fit, is the classical registration loop (ICP; CPD's EM) and the core of LoopReg (NeurIPS
  2020) and Neural ICP (ECCV 2024), where a learned correspondence is refined against the body model.

## Arms (all start from the deployed CSE targets; D1_PROD recipe; 3 seeds)

| arm | what happens after H2 (skeleton placed) |
|---|---|
| B0 | production, no targets (Q2 base fits, reused) |
| B1 | deployed targets kept in all stages (Q2 base fits, reused) |
| HIER | targets dropped after H2 |
| **LCF** (leg-consistency filter, M02 slot) | keep a target only if the nearest vertex of the current H2 fit to the target position belongs to the same leg as the target's template vertex; non-leg template vertices unchanged |
| **REEST** (re-estimation, M04 slot) | recompute targets with the same network, retrieval score = cosine similarity + spatial prior from the current fit: score(p, v) = cos(q_p, k_v) / tau − ||p − x_v(H2)||^2 / (2 sigma^2), tau = network temperature, sigma = 0.05 (normalised units, fixed now, about one leg-segment width); same cycle filter and aggregation as deployed |
| ORACLE-LF | drop exactly the wrong-leg targets using ground truth (upper bound for any filter) |

The surface stages (Stage_2, Stage_3) then run with the arm's targets. LCF and REEST are
annotation-free; ORACLE-LF is not deployable.

## Data and endpoints

The two validated real-articulation synthetic sets (Q2 REALPOSE, REALSCALE; 66 specimens, 11 source
poses x 3, same meshes and base fits as Q2). Final-stage joint error / body-axis length (primary),
fit-implied identity, chamfer.

**Mechanism check (must hold or the arm is void as a test of H-legid):** the arm's target wrong-leg
fraction after the H2 step is lower than deployed (paired over specimens), and its coverage is
reported (a filter that wins by removing everything is not a repair).

## Decision rule

A method PASSES the synthetic gate if, pooled over the 66 specimens, it beats BOTH B0 and B1 on
final joint error (paired sign test p < 0.05, and the cluster-bootstrap 95% CI over the 11 source
poses of the mean paired difference excludes 0), AND its mechanism check holds. Only passing methods
get their single JAB run. ORACLE-LF's gain sets the ceiling against which LCF/REEST are reported
(fraction of the oracle gain recovered).
