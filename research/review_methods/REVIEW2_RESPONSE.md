# Response to the second external review (2026-10-07)

Each point checked against what is on disk before it changes the plan. Verdicts: ADOPT, ADOPT WITH
CHANGE, ALREADY ANSWERED (with evidence), or CORRECTS US.

## Points that correct our own claims

| review point | check | verdict |
|---|---|---|
| §9: P0 is not an "independent real-pose corpus"; its poses are SMILify fits, not ground truth | Correct. "Independent" meant genus/specimen-disjoint from JAB, not independent labels. Fitted poses inherit the fitter's failure modes, and the pilot already showed fitted articulation *under*-estimates expert articulation (REALPOSE fits 25.5 deg vs expert 29.6 deg on the same 11 specimens). Circularity is limited: evaluation stays on expert joints, so a biased pose library can only make training coverage worse, not inflate the test score. | **CORRECTS US.** Renamed "real-scan fitted-pose library". P0's validity gate (fitted vs expert, JAB) is kept and its result travels with every P0 use. Added the review's A/B/C training-distribution design so the library is compared against a broadened synthetic distribution that does not depend on the fitter. |
| §1B: "articulation is the dominant *tested* source", not "the cause" | Correct; S2 verdict is PARTIAL, and S2 is synthetic. | **CORRECTS US.** Wording fixed in all results files. |
| §15: keep leave-one-morphology-out | Our basis for deprioritising it ("real shapes are inside the training range") is measured *in the model's own 25-PC shape space*. Anatomy the model cannot express is invisible to that measure, and REALSHAPE (fitted betas on model surfaces) cannot test it either. So "morphology is not the problem" is only established for **in-model** morphology. | **CORRECTS US.** Claim narrowed to in-model morphology. Out-of-model morphology remains open and is the leading candidate for the unexplained Cephalotes/Discothyrea cycle distances. LOMO moved to Tier 2. A synthetic LOMO also cannot test out-of-model shape, which is stated as its limit. |

## Points already partly answered by data on disk

| review point | evidence | verdict |
|---|---|---|
| §2/§12: track identity through the optimisation; "geometry improves while identity erodes" | Q3a A2 already measured fit-implied leg identity at H2 and S3 on JAB: under CSE it **falls** from H2 to S3 on 9/11 specimens (e.g. Cataglyphis 0.96 → 0.79, Cyphomyrmex 0.87 → 0.72, Discothyrea 0.70 → 0.52), while chamfer falls. Only two stages were measured, on real data, with proxy labels. | **ADOPT, highest priority** as Q3c: all five stages, both real (proxy) and synthetic (true identity, S1 fits), with a pre-registered definition of an "identity-erosion event". No new fitting needed. |
| §13: correspondence-quality threshold, real specimens overlaid | S1 gives the synthetic dose curve; Q3a A2b gives proxy-corrected real target quality. | **ADOPT WITH CHANGE**: the overlay is only valid if the error *type* matches; S1 corrupts with coherent wrong-leg patches, real targets also land off the legs (non-leg 0.17-0.38 on the bad four). The figure will mark that mismatch, and a non-leg corruption arm is added to the next S1 run. |
| §11: deprioritise M02 uncertainty | S1 rho 0.2: hier-only still 2.25 vs prod 2.43 (n.s.), i.e. *where* correspondence is applied matters about as much as how confident it is. | **ADOPT.** M02 becomes an enhancement of M04, not a standalone arm. |

## Points adopted as new experiments

| review point | design change |
|---|---|
| §4: identity-constraint weight sweep (lambda_corr) | Q3d: lambda_corr in {0, 0.1, 0.25, 0.5, 1, 2} x corruption rho in {0, 0.2, 0.4} on P48. Answers "how much identity weight before geometry stops overriding it", and whether the optimum shifts with target quality. Synthetic only; never tuned on JAB. |
| §3: M04 as alternation C0 -> theta1 -> C1 -> ..., testing whether C(t+1) becomes more *anatomically* correct | M04's pre-registration will measure identity of C(t) at each iteration (true labels on synthetic), not just final joint error, so "more geometrically convenient" and "more anatomically correct" are separated. |
| §6/§7: hierarchical correspondence (subsystem -> instance -> position -> vertex), skeleton-relative coordinates | M01 design. Constraint from W1 carried in: normalising segment length/radius destroyed signal; skeleton-relative coordinates must keep absolute units (position along the chain in length units, not fraction). |
| §8: canonicalisation with an oracle arm | M05 design: raw / predicted-skeleton unposed / GT-skeleton unposed, so pose normalisation is tested as a mechanism before it is engineered. |
| §10: training pose distribution A (current) / B (fitted real poses) / C (broadened synthetic) | Adopted as the P0 training experiment; C protects against the circularity of B. |
| §5: oracle gains are not additive | Q2 already has O-all; pairwise combinations (corr+pose, part+pose) added only if O-all differs from the best single arm by more than the seed spread. |

## Priority after this review

Tier 0: Q2 (running), **Q3c identity trajectories (now)**, S2-confirm (running), P0 validity gate.
Tier 1: Q3d lambda sweep, M04 with per-iteration identity, M01 hierarchical skeleton-conditioned.
Tier 2: M05 with oracle arm, P0 A/B/C training distributions, LOMO (in-model only, stated).
Tier 3: M02 as an M04 enhancement, M08 classical baselines (smoke test queued).
