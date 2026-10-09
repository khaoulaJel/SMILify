# Pre-registration — H_A4: can the coxa fix be kept without the shape cost?

Written 2026-08-28, before the arm runs. **This is the closing measurement of the coxa thread**, not
a new line of enquiry. The mechanism is already established and replicated; the only open question
is what it costs and whether that cost is controllable.

## Settled before this arm, and not re-tested here

Carrying the correspondence term into D1 — which previously had **no** correspondence term at all —
recovers 119.3% of the coxal regression (co 0.4115 → 0.2265, 43/48, sign p = 0.0000) and places the
coxa **2.3× closer to ground truth** (0.01857 → 0.00794, 45/48, sign p = 0.0000), with
`deform_verts` pinned at exactly 0, replicated across two seeds. `leg_acc` 0.9418 → 0.9682 against
an E0 ceiling of 0.9926. Four prior mechanisms failed on the coxa; this one works.

The single unresolved item: `betas` error vs GT rose **16.3%** relative, firing H_A3's magnitude
escape-hatch check and yielding a BOUGHT-ELSEWHERE verdict.

## The instrument-precedence defect, fixed here in writing

H_A3 registered two escape-hatch instruments and did not say which governs. They disagreed
(magnitude fired at +16.3%; correlation was flat at r = +0.059, p = 0.69). That was a real defect in
my pre-registration. It is resolved now, **before** H_A4 is run, by recognising that the two answer
different questions rather than competing:

- **Magnitude — `betas` error vs GT — is AUTHORITATIVE for PASS/FAIL.** It answers *"is there a
  cost?"*, which is the shippability question. A cost that exists is a cost regardless of what
  causes it.
- **Correlation — per-specimen coxa gain vs `betas` error change — is DIAGNOSTIC ONLY and never
  gates the verdict.** It answers *"is the cost mechanistically tied to the fix?"*: a strong
  positive r means shape is paying for the coxa; a flat r means something else moved independently
  and merely also got worse. It determines which **mechanism story** is reported, not whether the
  arm passes.

This precedence applies to H_A4 and supersedes H_A3's ambiguity going forward. It does **not**
retroactively change H_A3's recorded BOUGHT-ELSEWHERE verdict.

## Arm

`D1_cse_nodeform_betaprior` — identical to `D1_with_cse_nodeform_s1` (`scheme: pose`,
`--cse_correspondence_from`, same `C13_uniform_hier/H2_joint.npz` handoff, `--seed 1`) except
`w_beta_prior: 0.002` in both stages. That value is the project's own established setting
(`A3_priors.yaml`, `B1_best_*.yaml`, `E4_priors.yaml`), not a number chosen for this arm.

One variable. The hypothesis is that the Mahalanobis shape prior holds `betas` near the model's own
distribution while the correspondence term acts on placement.

## Bar

Baseline for the cost check is `C13_uniform` (`betas` err 3.0173); baseline for the endpoint is
`C13_uniform` (co 0.4115). Reference point to preserve is `_s1` (co 0.2265, placement 0.00794).

- **PASS — clean, shippable fix.** All of:
  1. co ≤ 0.3030 with sign p < 0.05 *(the fix survives the prior)*;
  2. coxa placement ≤ 0.0140 with sign p < 0.05 *(placement gain survives)*;
  3. `betas` error ≤ 3.3190 *(within +10% of `C13_uniform`, i.e. the magnitude check does not fire)*;
  4. no distal regression > 0.02, `leg_acc` ≥ 0.9415, `mean|deform_verts|` = 0 or the arm is void.
- **PRIOR TOO WEAK** — 1 and 2 hold, 3 does not. The trade is real and `w_beta_prior = 0.002` does
  not control it. Then the coxa fix ships **with a stated 16%-scale shape cost**, and the thread
  closes on that characterisation.
- **PRIOR TOO STRONG** — 3 holds but 1 or 2 fails. The prior suppresses the fix; the two objectives
  are in genuine conflict and cannot be had together at this weight. Reported as such; **no weight
  sweep** — that would be tuning until something passes.
- **NEITHER** — 1/2 and 3 all fail. Report and close.

Diagnostic, reported in every case and gating nothing: the per-specimen correlation between coxa
placement gain and `betas` error change, plus `betas_trans` and `log_beta_scales` magnitude ratios.

## Not claimed

- Synthetic only; bench50 untouched. A PASS is a fix for the coxa on synthetic data.
- The coxa is ~16.7% of total correspondence error; even a clean PASS is a bounded win.
- `leg_acc` is not solved at 0.9682 against a 0.9926 floor.
- **This closes the coxa thread either way.** No further variant is opened from H_A4's result.
