# H_A2 result — STILL-DECOUPLED by the registered bar, and the bar's mechanism check was the wrong instrument

Bar fixed in `PREREGISTRATION_HA2_deform_off_20260828.md` (committed `0d0548a1`) before job 3278451
ran. n=48 P48, `scheme: pose` (no `deform_verts`), correspondence carried into D1, same
`C13_uniform_hier/H2_joint.npz` handoff.

| run | leg_acc | **co** | tr | fe | ti | ta | pt | mean\|deform\| |
|---|---|---|---|---|---|---|---|---|
| H2 handoff | 0.9649 | 0.2564 | 0.0539 | 0.0445 | 0.0423 | 0.0534 | 0.0579 | — |
| `C13_uniform` | 0.9415 | 0.4115 | 0.0829 | 0.0611 | 0.0396 | 0.0485 | 0.0587 | 0.001665 |
| `D1_with_cse` | 0.9728 | 0.2142 | 0.0477 | 0.0301 | 0.0413 | 0.0448 | 0.0511 | 0.004179 |
| **`D1_with_cse_nodeform`** | **0.9694** | **0.2226** | 0.0523 | 0.0362 | 0.0404 | 0.0382 | 0.0440 | **0.000000** |

`deform_verts` is **exactly zero**, so the scheme took and the escape hatch is genuinely gone — the
arm is valid by its own void condition.

## Endpoint: met. Registered mechanism check: failed.

co 0.4115 → **0.2226**, 121.8% recovery, 43/48, sign p = 0.0000, `leg_acc` 0.9418 → 0.9690.
The ≤ 0.3030 endpoint is cleared with room.

The registered mechanism check was "coxa `joint_rot` error vs GT must not worsen against 0.3024
rad". It reads **0.3750 rad — worse than `C13_uniform` and worse than H_A's 0.3183.**

**Verdict, per the letter of the bar: STILL-DECOUPLED** — *"co clears 0.3030 but `joint_rot` error
worsens anyway. Then the surface metric is reachable without the skeleton by some other route,
which is a finding about the metric, not a fix."*

Note what this already rules out: **`deform_verts` was never the mechanism.** With it pinned at
exactly zero, the surface metric still improves 122%. H_A's 2.5× deformation inflation was a
correlate, not the cause. That part of H_A's diagnosis was wrong.

## Post-hoc, and labelled as such: the check measured the wrong quantity

Measured **after** seeing the above, therefore **not used to overturn the verdict** — the same rule
applied to F2's threshold, where a true technical reason discovered at the wrong time was declined:

Coxa **surface placement** error (per-leg coxa centroid, fitted vs GT, as a fraction of body
diagonal):

| run | coxa placement error | Δ vs `C13_uniform` | better | sign p |
|---|---|---|---|---|
| `C13_uniform` | 0.01857 | — | — | — |
| `D1_with_cse` | 0.00699 | −0.01158 | 47/48 | 0.0000 |
| `D1_with_cse_nodeform` | **0.00792** | −0.01065 | 45/48 | 0.0000 |

The coxa is placed **2.3× closer to ground truth**, on 45 of 48 specimens, with no free-form
deformation available. That is anatomical improvement by any reasonable reading.

Why the registered check missed it: for the coxa, `joint_rot` orients the *downstream chain*; the
coxa's own position is set by where its joint sits — the joint regressor, `betas`, and global
placement — not by its own rotation. So coxa `joint_rot` error was never a good proxy for coxal
*placement*. It was chosen for H_A because it was the quantity that had moved the wrong way there,
which is not the same as being the right instrument.

## What this means, stated without over-claiming

The registered verdict stands: **STILL-DECOUPLED**, and this arm is not reported as a pass. But the
post-hoc measurement makes the pre-registered *interpretation* of that verdict — "a finding about
the metric, not a fix" — likely wrong in the other direction: both the surface metric and true
coxal placement improved together, which is coupling, not decoupling.

**The honest resolution is a fresh arm with the mechanism check specified correctly in advance**
(coxa surface placement vs GT as the registered check, not `joint_rot`), not a re-reading of this
one. Until that runs, this is: endpoint cleared, registered check failed, and strong post-hoc
evidence that the check was mis-specified.

## Not claimed

- Not a shippable fix, because the registered bar was not met as written.
- Synthetic only. The coxa is ~16.7% of total correspondence error even if this holds.
- The post-hoc placement number is post-hoc, and carries the standing of post-hoc evidence.
