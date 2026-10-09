# E1/E2 result — FAIL and SYMPTOM. The coxal-via-pose thread closes.

Bar fixed before the run in `PREREGISTRATION_E1E2_frozen_pose_20260828.md` (committed `8e1c0ad5`,
before job 3264569 was scored). Hier-only, n=48 P48, completed 14:20:22 on 2026-08-28.

| run | leg_acc | **co** | tr | fe | ti | ta | pt |
|---|---|---|---|---|---|---|---|
| `E0_identity` (metric floor) | 0.9926 | **0.0908** | 0.0110 | 0.0043 | 0.0003 | 0.0021 | 0.0010 |
| `C14p_gtinit_nocse_hier_H2` (control: GT-seeded, free) | 0.9413 | **0.4564** | 0.0903 | 0.0520 | 0.0216 | 0.0213 | 0.0192 |
| `E1_frozen_pose` | 0.9382 | **0.4514** | 0.0832 | 0.0452 | 0.0183 | 0.0180 | 0.0117 |
| `E2_frozen_pose_gtscale` | 0.9350 | **0.4674** | 0.0923 | 0.0395 | 0.0175 | 0.0162 | 0.0079 |

## E1 — FAIL

`co` 0.4564 → 0.4514, **Δ = −0.0050**, 25/48, **sign p = 0.885** (Wilcoxon 0.757, t 0.601).

PASS required ≤ 0.20 with sign p < 0.05; PARTIAL required any fall with sign p < 0.05. Neither.
The pre-registered reading applies: **no shape configuration this optimizer reaches places the
coxa, even with pose removed entirely as a confound.**

The freeze is not in doubt — it was verified bit-identical to GT before the run. What the result
says is that pose freedom was never what was driving coxal failure: holding pose at the exact truth
and letting `betas`, `log_beta_scales`, `betas_trans`, `global_rot` and `trans` optimize freely
moves the coxa by half a percent, against a floor 0.36 below.

Guard-rails: distal did not regress (every segment improved slightly). `leg_acc` fell 0.9421 →
0.9389, not significant (p = 0.10) — recorded because the bar said it must not fall, and it did,
though the arm failed on its primary endpoint regardless.

## E2 — SYMPTOM

`co` 0.4514 → 0.4674, **Δ = +0.0160 (worse)**, sign p = 0.111 (t p = 0.025). `leg_acc` falls
0.9389 → 0.9354, Wilcoxon p = 0.0079.

CAUSAL required a further −0.10 with sign p < 0.05. Not only absent — forcing `log_beta_scales` to
its **true** values makes coxal placement and overall leg accuracy *worse*.

So the r = +0.503 correlation between `log_beta_scales` error and coxal error was
**compensation, not cause**: the optimizer was moving the scale block away from truth to absorb a
problem originating elsewhere, and denying it that freedom costs accuracy. The driver of coxal
error remains unidentified, and this line does not find it.

## What closes, and what does not

Per the standing rule and the pre-registration's own final section, **the coxa-via-pose thread
closes here.** Four mechanisms have now been tested and none moves the coxa: correspondence
(−0.009), gradient reweighting (−0.010, saturating), pose initialization (−0.009, n.s.), and now
pose *freezing* at ground truth (−0.005, n.s.) with the scale block ruled out as a cause.

No representability claim is made, as pre-registered: `E0_identity` scores 0.0908 and P48 targets
are generated from the model, so parameters placing the coxa well demonstrably exist. This is a
characterized **optimization** limitation, not a model limitation.

**Scope limit, stated rather than glossed:** E1/E2 carry no correspondence term, deliberately, so
they pair exactly with `C14p_gtinit_nocse`. They therefore say nothing about frozen pose *plus*
correspondence. That would be a fifth variant of the same question and is explicitly not opened.

The live lead is elsewhere and was recorded before these results existed: with correspondence and
*free* pose, the hierarchical stage reaches **co = 0.2564** (`C13_uniform_hier_H2`) — better than
either ground-truth-pose arm here — and D1 then degrades it to 0.4115. See
`OPEN_THREAD_D1_coxal_degradation_20260828.md`. That thread asks which D1 term undoes a position
already found, which is a different question from anything E1/E2 tested.
