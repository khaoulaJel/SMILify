# Q3b: controlled shifts on synthetic data (tests of Q3a's hypotheses)

Written 2026-10-06, before any shifted specimen was generated. Q3a's hypotheses were formed on the
JAB test specimens; this is where they are tested, on synthetic data only (PROTOCOL §3.6).
One disclosed use of real data: the *articulation level* of the test condition ART-real is set to the
real expert skeletons' median bend (A7), which is a property of the inputs, not of any method's
performance.

## S2: which shift reproduces the real failure signature? (inference only, existing C3 network)

**Real signature to reproduce** (Q3a A2/A4): per-point leg accuracy on tr/fe/ti/ta falls from
~0.95 (synthetic) to 0.23-0.80 (real, median ~0.6); wrong leg NUMBER errors dominate wrong SIDE
errors (A2: side 0.00-0.05, number 0.10-0.37); cycle median rises above the synthetic range.

**Test sets.** 48 new specimens per condition, from the training generator with seeds disjoint from
the training corpus (training used 20260826-20260833; here 777000 + condition index). Each condition
changes ONE factor from the in-distribution baseline C0 (training generator settings).

| condition | factor | levels |
|---|---|---|
| C0 | none (in-distribution) | - |
| ART | articulation: `pose_scale` calibrated so the median leg bend vs rest (A7 statistic) hits a target | 15, 20, 25, 30 (= ART-real, real median 29.6), 35 deg |
| ROT | global rotation about a random axis per specimen | 10, 20, 30 deg |
| SCL | per-joint scale `scale_scale` | 0.2, 0.3 (training 0.10) |
| SHP | shape `shape_scale` | 1.5, 2.0 (training 1.0; real shapes are inside training, so expected small) |
| DEG | scan degradation: Gaussian noise 0.5% of extent; 10% of faces removed in patches; ta+pt removed on 2 random legs | each separately |
| REAL | combined: ART-real + ROT 15 + SCL 0.25 | - |

Input normalisation identical to real scans: (V - mean) / max|V - mean|. Network fed exactly as in
JAB: 8 repeats x 2048 points.

**Metrics** per specimen: per-point leg accuracy (true labels); error-type fractions (wrong side /
wrong leg number / non-leg); cycle-distance median. Geodesic error / sqrt(area) (RULES_AUDIT R7)
as secondary once `geodesic.py` is verified.

**Decision rule for H-artic (fixed now):**
- SUPPORTED if at ART-real: mean leg accuracy <= 0.80 (C0 expected ~0.95), wrong-number fraction
  >= 2x wrong-side fraction, cycle median >= 2x C0; AND the ART drop is larger than the drop of every
  other single factor at its largest level; AND leg accuracy decreases monotonically over the five
  ART levels (Spearman rho <= -0.9).
- NOT SUPPORTED if leg accuracy at ART-real >= 0.90.
- PARTIAL otherwise (articulation contributes but does not reproduce the signature alone; REAL then
  says whether the combination does).

Paired specimen-level statistics are not used here: conditions use different specimens by design.
Effects are reported as condition means with specimen-bootstrap 95% CIs.

## S1: does the fitter turn wrong targets into skeleton error? (H-absorb)

**Corruption of the real type.** Dense ground-truth correspondence targets on P48 (each template
vertex's true position), then for a fraction rho of leg vertices on tr/fe/ti/ta/pt, replace the
target with the true position of the *same-index vertex on the adjacent leg of the same side*
(leg number +-1), the error type A2 measured on real scans. rho in {0, 0.2, 0.4}; 0.4 matches the
median real CSE target leg error (A2: 1 - leg accuracy ~0.3-0.5).

**Arms** (D1_PROD recipe, 3 seeds each): `prod` (no targets); `full` (targets in hierarchical AND
surface stages, as JAB I_cse); `hier` (targets in the hierarchical stage only). `full` and `hier`
at each rho.

**Metrics:** FK joint error vs GT (per joint / synthetic body-axis length L, DEVIATIONS D1.6), by
region, at H2 and S3; free channels (deform_rms, |betas_trans|, |log_beta_scales|, H2->S3 pose
change); post-fit correspondence (C13 scorer).

**Decision rule for H-absorb:**
- SUPPORTED if at rho = 0.4, `full`'s joint error rises from H2 to S3 (paired over specimens, sign
  p < 0.05) while `prod`'s falls or holds, AND `full`'s free channels at S3 exceed `prod`'s on the
  same specimens (sign p < 0.05), AND `hier` at S3 beats `full` at S3 (sign p < 0.05).
- NOT SUPPORTED if `full` at rho = 0.4 does not degrade H2 -> S3.
- rho = 0 is the positive control: `full` must not be worse than `hier` there (otherwise the
  degradation is not about wrong targets). If it is, the result is reported as "the surface-stage
  term hurts even when correct" rather than as support for H-absorb.

## What these do not test

Morphology outside the model's shape space (real anatomy the 25 PCs cannot express) cannot be
generated synthetically from the model; if S2 leaves a residual Cephalotes-type failure unexplained,
that is reported as unexplained, not attributed.
