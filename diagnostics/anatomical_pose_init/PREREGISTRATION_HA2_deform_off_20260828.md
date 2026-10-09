# Pre-registration — H_A2: does the D1 correspondence gain survive with `deform_verts` off?

Written before the run, 2026-08-28. New thread, not a continuation of H_A — H_A's diagnosis is
already made; this asks whether the fix is salvageable.

## Why

`D1_with_cse` cleared its bar (co 0.4115 → 0.2142, 127% recovery) and failed its mechanism check:
mean |`deform_verts`| rose 2.5× (0.001665 → 0.004179) while coxa `joint_rot` error vs GT got
**worse** (0.3024 → 0.3183 rad). The correspondence term was satisfied by free-form per-vertex
deformation rather than by moving the skeleton. Removing that escape hatch (`scheme: pose`) tests
whether any of the gain is anatomical.

Arm: `D1_with_cse_nodeform` — the `D1_no_deform.yaml` config (identical weights, `scheme: pose`)
plus `--cse_correspondence_from`, from the same `C13_uniform_hier/H2_joint.npz` handoff. One
variable against `D1_with_cse`.

## Bar

Primary: paired per-specimen `co` leg-level error vs `C13_uniform` (0.4115), n=48, sign + Wilcoxon
+ paired-t. Gap to recover 0.1551; floor is E0's 0.0908.

- **PASS** — co ≤ **0.3030** (≥70% recovery) with sign p < 0.05, **and** the mechanism check passes:
  coxa `joint_rot` error vs GT must **not worsen** relative to `C13_uniform`'s 0.3024 rad.
  Both conditions, not either. This is the shippable outcome.
- **DEFORM-DEPENDENT** — co does not clear 0.3030. Then H_A's entire gain was the escape hatch, and
  carrying correspondence into D1 is not a fix. The coxa thread closes for good.
- **STILL DECOUPLED** — co clears 0.3030 but `joint_rot` error worsens anyway. Then the surface
  metric is reachable without the skeleton by some *other* route, which is a finding about the
  metric, not a fix.

Guard-rails unchanged: distal must not regress > 0.02; `leg_acc` must not fall below 0.9415; any
result better than the H2 handoff's own co = 0.2564 is a suspected bug. `deform_verts` magnitude
reported alongside (expected ≈ 0, since the parameter is not optimised — a non-zero value means
the scheme did not take, and the arm is void).

## Not claimed

Synthetic only. Recovering the coxa is ~16.7% of total correspondence error even if it works.
