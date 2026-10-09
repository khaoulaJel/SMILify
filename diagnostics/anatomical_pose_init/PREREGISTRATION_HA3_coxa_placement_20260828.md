# Pre-registration — H_A3: the coxa fix, with the mechanism check specified correctly

Written 2026-08-28, **before the replicate is run and before any of its numbers exist**.

## Why this arm exists, stated so the sequence is auditable

`H_A2` cleared its endpoint (co 0.4115 → 0.2226, 121.8%, 43/48, sign p = 0.0000) with
`deform_verts` pinned at exactly 0, and **failed** its registered mechanism check (coxa `joint_rot`
error 0.3024 → 0.3750 rad). Verdict recorded as STILL-DECOUPLED and not overturned.

Afterwards I measured coxa **surface placement** (per-leg coxa centroid vs GT / body diagonal) and
got 0.01857 → 0.00792, 45/48. That reasoning — `joint_rot` orients the downstream chain while the
coxa's own position is set by the joint regressor, `betas` and global placement — says the
registered check was the wrong instrument.

**That number is already seen, so re-scoring the same run against a check designed after seeing it
is not confirmation — it is the same evidence relabelled.** Therefore H_A3 runs a **fresh
independent replicate** (`--seed 1`, identical recipe) and takes its numbers as the primary
reading. The existing `D1_with_cse_nodeform` is reported as a supporting, already-seen replicate,
never as the primary.

## Arm

`D1_with_cse_nodeform_s1` — identical to `D1_with_cse_nodeform` (`D1_no_deform.yaml`, `scheme:
pose`, `--cse_correspondence_from`, same `C13_uniform_hier/H2_joint.npz` handoff) except
`--seed 1`.

## Bar — endpoint AND mechanism check, both required

Primary endpoint, paired per-specimen vs `C13_uniform`, n=48, sign + Wilcoxon + paired-t:
- **`co` leg-level error ≤ 0.3030** (≥70% of the 0.1551 gap), sign p < 0.05.

Mechanism check, now the correct quantity, and a **pass condition**:
- **Coxa surface placement error vs GT must IMPROVE** — ≤ 0.0140 (a ≥25% reduction from
  `C13_uniform`'s 0.01857), sign p < 0.05. Not merely "not worsen": the claim being made is that
  the coxa is genuinely better placed, so the check must show it.

Escape-hatch checks — **unmeasured at the time of writing**, and the point of this arm beyond
replication. `joint_rot` was the wrong instrument, but the reasoning that replaced it (position
comes from the joint regressor / `betas` / global placement) names exactly where a *new* escape
hatch would live. All three are reported, and any one of them firing makes the result a
**BOUGHT-ELSEWHERE**, not a pass:
- `betas` error vs GT must not worsen by more than 10% relative to `C13_uniform`.
- `betas_trans` and `log_beta_scales` magnitudes must not grow by more than 2× (the same shape of
  test that caught `deform_verts` at 2.5× in H_A).
- Per-specimen correlation between the coxa placement gain and each of those movements: a strong
  positive correlation (|r| > 0.5, p < 0.05) means the gain is being bought from that block and is
  reported as such.

Guard-rails carried forward: distal must not regress > 0.02; `leg_acc` must not fall below 0.9415;
`mean|deform_verts|` must be exactly 0 or the arm is void; any co better than the H2 handoff's
0.2564 is flagged for inspection rather than celebrated.

## Readings fixed in advance

- **PASS** — endpoint met, placement check met, no escape-hatch check fires. Then carrying
  correspondence into D1 without free-form deformation is a **real, shippable fix** for coxal
  placement, and the coxa thread closes with a positive result.
- **BOUGHT-ELSEWHERE** — endpoint and placement met, but an escape-hatch check fires. The surface
  gain is real and is paid for somewhere else in shape space; reported as a trade, not a fix.
- **NOT REPLICATED** — the fresh seed does not reproduce H_A2's endpoint. Then H_A2 was
  seed-specific and the thread closes negative.
- **FAIL** — endpoint met, placement check not met. The post-hoc reading was wrong and
  STILL-DECOUPLED stands as H_A2 recorded it.

## Not claimed

- Synthetic only; bench50 is untouched by this.
- The coxa is ~16.7% of total correspondence error, so even a PASS is a bounded win.
- A PASS confirms the *mechanism* (correspondence must survive into D1), not that `leg_acc` is
  solved — `leg_acc` here is 0.9690 against an E0 floor of 0.9926.
