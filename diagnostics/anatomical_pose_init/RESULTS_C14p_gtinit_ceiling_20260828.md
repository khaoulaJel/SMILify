# C14p result — the coxa is NOT a pose problem, and C14 is ruled out for it

> **CORRECTION (2026-08-28, after this document was written).** This document quotes coxal error
> against an implicit zero point of 0. It should be read against the coxa's **measured floor**:
> `E0_identity` audits the ground-truth mesh *as* the fit — placement exactly correct, no fit run —
> and still scores **0.0908** on `co`, against 0.0003 on `ti` and 0.0010 on `pt`, because adjacent
> coxae sit close enough that nearest-vertex matching misassigns ~9% of coxal points on an exact
> mesh.
>
> **This floor was already known.** `LAB_RECORD_correspondence_20260827.md` §15 measured it as a
> paired ceiling at **0.0949** and correctly computed every share of the *addressable* residual
> against it (co 57.8%, tr 27.7%). E0 is an independent confirmation of that number by a different
> route, not a new discovery, and no percentage in the lab record is affected.
>
> What is wrong is **the framing in this document**, which states the coxa "sits at ~0.41,
> untouched by anything" without subtracting the known floor. Read it as **~0.32 above the coxa's
> own floor**. The deltas between arms are differences and are unaffected.
>
> Also worth stating explicitly here, since this document does not: because a perfect parameter set
> scores 0.09 rather than 0.41, and P48 targets are generated *from* the model, **coxal
> representability is confirmed, not open**. Any reading of this document as evidence of a model
> limitation is wrong.
>
> See `PREREGISTRATION_E1E2_frozen_pose_20260828.md` Control 1, and `make_E0_identity_20260828.py`.

Prerequisite ceiling probe registered in `DESIGN_C14_kinematic_init_20260828.md`. Job 3261013
(`rwth2151`/`c23g`), two arms, n=48 P48, completed 12:39:09 on 2026-08-28. Recipe byte-identical
to C13's `run_arm`; the only differences are `--init_joint_rot_from ground_truth.npz` and, for the
`nocse` arm, the absence of `--cse_correspondence_from`.

## The 2x2, in per-segment leg-level error (lower is better)

| arm | init | corresp. | leg_acc | **co** | tr | fe | ti | ta | pt |
|---|---|---|---|---|---|---|---|---|---|
| `P48_zero`          | zero | — | 0.8859 | **0.4204** | 0.1523 | 0.1588 | 0.1677 | 0.1734 | 0.1717 |
| `C13_uniform`       | zero | CSE | 0.9415 | **0.4115** | 0.0829 | 0.0611 | 0.0396 | 0.0485 | 0.0587 |
| `C14p_gtinit_nocse` | **GT** | — | 0.9468 | **0.4114** | 0.0866 | 0.0504 | 0.0217 | 0.0285 | 0.0140 |
| `C14p_gtinit_cse`   | **GT** | CSE | 0.9459 | **0.4060** | 0.0837 | 0.0609 | 0.0248 | 0.0173 | 0.0135 |

## Result: the coxa does not move, even for a perfect initializer

**Ground-truth pose — handed to the optimizer directly — leaves `co` at 0.4114.** Against
`P48_zero` that is −0.0091, 27/48, **sign p = 0.47** (Wilcoxon p = 0.20): not significant.
Against `C13_uniform`, the GT+CSE arm gives −0.0056, sign p = 0.47.

Every arm ever run on this corpus now sits in a 0.402–0.420 band on `co`: zero init 0.4204,
CSE 0.4115, tolerance-weighted 0.4019, perfect pose 0.4114, perfect pose + CSE 0.4060.

This is not a weak effect. It is a **ceiling argument**: a learned pose initializer's best possible
output *is* the ground-truth pose, and the ground-truth pose does not fix the coxa. **C14 —
or any pose initializer, however good — is therefore ruled out as a fix for coxal placement.**
That includes C14 stage 1, whose whole design was to predict the six `co` rotations.

The measurement is not insensitive, and that is what makes the null informative: the same GT init
transforms the distal chain against `P48_zero` — `pt` 0.1717 → 0.0140 (**12x**, sign p = 0.0015),
`ti` 0.1677 → 0.0217 (**8x**), `ta` 0.1734 → 0.0285, `fe` 0.1588 → 0.0504, all sign p < 0.002,
leg_acc +0.0617. Pose initialization is enormously powerful in this fitter. It simply has no
authority over where the coxa sits.

Combined with C13, the coxa is now excluded from all three mechanisms tried this session:
correspondence (−0.009), gradient reweighting (−0.010, saturating), and pose init (−0.009, n.s.).
**The remaining explanation is shape space** — joint regressor, `log_beta_scales`, `betas_trans`,
or the thorax betas fixing where the coxa attaches to the fitted body. Consistent with the earlier
decomposition (70.7% of the coxal offset common across all six coxae; body per-vertex error
0.0203 ~ coxa 0.0201, i.e. the coxa inherits the body's placement error).

## Second reading: how much headroom C14 has left for anything else

`C14p_gtinit_cse` vs `C13_uniform` is the honest ceiling on a pose initializer built on top of the
correspondence mechanism already shipped:

- leg_acc **+0.0049** (Wilcoxon p = 0.0005) — real, but small.
- Distal gains are genuine (`pt` −0.045, `ta` −0.031, `ti` −0.015) yet sit on already-small numbers.

And the two mechanisms are **near-substitutes, not complements**: GT init *without* correspondence
reaches leg_acc 0.9468, GT init *with* it 0.9459, correspondence alone 0.9415. Correspondence is
already collecting most of what a perfect initializer would deliver — exactly the calibration
stated when C14 was proposed, now measured rather than assumed.

So C14's entire remaining budget is ~+0.005 leg_acc and no coxal movement, against training a
network. On the evidence that is not worth building, and this probe cost one job to establish
instead of a training run plus a fit.

## What this redirects to

A model-expressiveness question, which is a different kind of experiment than anything run this
session: freeze pose at GT and test whether the coxa's position is **representable at all** by the
current shape space for these specimens (joint-regressor / beta / `betas_trans` degrees of freedom).
If it is not representable, no loss, weighting, correspondence, or initializer can ever place it,
and the coxal residual is a model limitation to report rather than an optimization target.

Note this probe reads ground truth. Both arms are ceilings and can never be reported as deployable
results.

## Free follow-up on the existing artifacts (no new run)

First a correction to how this run is described: `--init_joint_rot_from` **seeds** `joint_rot` and
the fitter then optimizes it freely (`optimise_hierarchical.py:394`). `C14p_gtinit_nocse` is a
GT-*initialized* run, not a GT-*frozen* one, so it cannot on its own answer whether the coxa is
representable. It can answer something the frozen version could not.

**1. The fitter walks the coxa away from the GT pose it was handed.** Mean |Δ| between fitted and
seeded axis-angle, per segment (radians): `co` 0.222, `tr` 0.287, `fe` 0.297, `ti` 0.274,
`ta` 0.353, `pt` 0.275. So GT init is not a fixed point — every segment drifts, and the coxa
drifts *least* of the six while still failing completely. The loss's optimum simply is not at the
true pose for the coxa, which is why seeding it there buys nothing.

**2. The only per-specimen variable that predicts coxal error is `log_beta_scales`.** Correlating
per-specimen `co` leg error (n=48, mean 0.4114) against what is already on disk:

| variable | Pearson r | p | Spearman rho | p |
|---|---|---|---|---|
| **`log_beta_scales` error (fit vs GT)** | **+0.503** | **0.0003** | +0.379 | 0.0079 |
| coxal pose drift from GT | +0.216 | 0.14 | +0.197 | 0.18 |
| `betas` error (fit vs GT) | +0.074 | 0.62 | +0.038 | 0.80 |
| specimen size (max extent) | −0.070 | 0.64 | −0.105 | 0.48 |
| `betas_trans` magnitude | −0.018 | 0.90 | +0.166 | 0.26 |

Five tests; p = 0.0003 survives Bonferroni (0.0015). Pose drift does **not** predict coxal error
(p = 0.14) — consistent with the ceiling result. Global `betas` do not either. The single
predictor is the **per-joint scale block**, which is shape space, and specifically not the global
shape vector.

**This is a correlation on 48 specimens, and the direction of causation is not established** —
a badly-placed coxa could equally well *force* `log_beta_scales` to compensate, making the error a
symptom rather than a cause. It is a pointer for where to aim the expressiveness test, not a
finding. But it names a specific parameter block to aim at, which is more than the shape-space
hypothesis had before this run existed.
