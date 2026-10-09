# C13 result — FAIL on the pre-registered bar, and the failure is mechanistic

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

Bar fixed before the run in `PREREGISTRATION_C13_tolerance_weighting_20260828.md`. n=48 (P48),
three arms sharing one correspondence file (`cse_p48_all.npz`) and one recipe, differing only in
the per-vertex weight on the dense term. Job 3253515, all three arms completed 09:26:55.

## Verdict: FAIL

Primary endpoint — paired per-specimen `co` leg-level error vs `C13_uniform`:

| arm | co error | Δ | better | sign p | Wilcoxon p | paired-t p |
|---|---|---|---|---|---|---|
| `C13_invtol`  | 0.4115 → 0.4019 | **−0.0097** | 29/48 | 0.193 | 0.022 | 0.014 |
| `C13_invtol2` | 0.4115 → 0.4030 | **−0.0085** | 29/48 | 0.144 | 0.042 | 0.061 |

PASS needed **−0.05 with sign p < 0.05**. The effect is ~5× too small and the registered gate
(sign test) does not clear. PARTIAL also required sign p < 0.05, so it is not PARTIAL either.
The magnitude-sensitive tests do reach p<0.05 — the direction is real — but a 0.01 shift on a
0.41 error was pre-declared as not worth the complexity, and that judgement stands.

Both guard-rails are violated, so no version of this could have been reported as an improvement:

| | co | tr | fe | ti | ta | pt | leg_acc | seg_acc |
|---|---|---|---|---|---|---|---|---|
| `C13_uniform`  | 0.4115 | 0.0829 | 0.0611 | 0.0396 | 0.0485 | 0.0587 | 0.9418 | 0.8538 |
| `C13_invtol`   | 0.4019 | 0.0859 | 0.0667 | **+0.0232** | 0.0671 | 0.0520 | −0.0026 | +0.0004 |
| `C13_invtol2`  | 0.4030 | 0.0896 | **+0.0245** | **+0.0469** | **+0.0409** | **+0.0335** | −0.0115 | −0.0056 |

Distal cap was +0.02; `ti` breaches it in both arms. Overall `leg_acc` falls in both.

Plumbing control is clean: `C13_uniform` reproduces `P48_cse_all` (leg_acc 0.9418 vs 0.943), so
the weighting code did not perturb the unweighted path.

## Why this is a mechanistic negative, not a null

**The weighting demonstrably worked. The coxa just did not respond to it.**

Weight ratio co:pt is 9.6× in `invtol` and 84× in `invtol2`. The distal segments track that dose
almost perfectly — `ti` degrades +0.023 then +0.047, `pt` +0.000 then +0.034 — which is exactly
what removing gradient from them should do. Gradient really was reallocated.

But the coxa **saturates at ~0.009 and then stops**: `invtol2` weights it 9× harder than `invtol`
and gets a *slightly worse* coxal number (0.4030 vs 0.4019). A gradient-starved parameter does not
behave this way. Dose-response is present everywhere the weight was taken *from* and absent where
it was given *to*.

This is the pre-registered FAIL reading, reached by the route the pre-registration named:

> coxal placement is NOT gradient-starved, and the remaining explanation is that the coxa's rigid
> offset is set by a part of the parameter space the dense term cannot steer (joint regressor /
> thorax betas) even when weighted. That redirects to shape-space.

Per the standing rule, I am not re-running variants of this idea.

## The number that reframes the whole coxal story

`P48_zero` co error is **0.4204**. Every arm ever run on this corpus sits at 0.40–0.42:

| | co |
|---|---|
| `P48_zero` (no correspondence, zero init) | 0.4204 |
| `C13_uniform` (CSE correspondence) | 0.4115 |
| `C13_invtol` (+ tolerance weighting) | 0.4019 |

Correspondence — which moved distal error by **4×** (`ti` 0.168 → 0.040) — moved the coxa by 0.009.
Tolerance weighting moved it another 0.010. **Nothing tried this session has moved the coxa.**
That is now a stronger and more specific claim than "the coxa is the addressable residual": the
coxa appears to be outside the reach of the dense data term entirely, however it is weighted.

## What this does to C14

It raises the stakes on the already-queued ceiling probe (`C14p`, job 3254254,
`DESIGN_C14_kinematic_init_20260828.md`) and makes its reading sharper:

- If **GT-seeded initialization also leaves co near 0.41**, the coxa is not a pose-initialization
  problem at all — it is shape-space (joint regressor / thorax betas), no initializer can reach it,
  and C14's justification must rest entirely on convergence and the distal chain.
- If **GT init drops co sharply**, then coxal placement *is* reachable from pose, just not from
  this loss's gradient — and C14's stage 1, which predicts precisely the six `co` rotations,
  is aimed at the one thing nothing else has moved.

Either way C13 has removed a hypothesis rather than a run, which is the point.

## Provenance note

`interleg_tolerance_20260828.py`, cited in the pre-registration as the source of the measured
tolerances, is **not on disk** (same vanishing-scripts pattern flagged previously). The values
survive hardcoded and documented at `fitter_3d/optimise_hierarchical.py:592-594`
(`co 0.0202 … pt 0.1926`), so C13 itself is sound, but the measurement cannot currently be
reproduced or re-derived on another corpus — which bench50 would require. Regenerate before reuse.

`per_specimen_per_segment_leg_acc` was added to `diagnostics/correspondence_accuracy/run_audit.py`
to compute this endpoint: the audit emitted per-specimen leg accuracy, per-*leg* accuracy, and
leg-conditioned segment accuracy, but not leg-level accuracy split by true segment. It reuses
`leg_confusion`'s existing predictions and changes no pre-existing number.
