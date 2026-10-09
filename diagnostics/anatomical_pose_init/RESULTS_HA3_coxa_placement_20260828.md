# H_A3 result — BOUGHT-ELSEWHERE. The coxa fix is real and it costs shape accuracy.

Bar fixed in `PREREGISTRATION_HA3_coxa_placement_20260828.md`, committed before job 3279368 ran.
Fresh independent replicate (`--seed 1`) taken as the **primary** reading; `D1_with_cse_nodeform`
is the already-seen supporting replicate.

| run | leg_acc | **co** | tr | fe | ti | ta | pt |
|---|---|---|---|---|---|---|---|
| `C13_uniform` | 0.9415 | 0.4115 | 0.0829 | 0.0611 | 0.0396 | 0.0485 | 0.0587 |
| `D1_with_cse_nodeform` (seen) | 0.9694 | 0.2226 | 0.0523 | 0.0362 | 0.0404 | 0.0382 | 0.0440 |
| **`_s1` (primary)** | **0.9689** | **0.2265** | 0.0519 | 0.0363 | 0.0389 | 0.0414 | 0.0598 |

## Registered checks, in order

- **Endpoint — MET.** co 0.4115 → **0.2265**, 119.3% recovery, 43/48, sign p = 0.0000
  (bar ≤ 0.3030). The fresh seed reproduces H_A2's 0.2226 closely: **H_A2 was not seed-specific.**
- **Void check — valid.** `mean|deform_verts|` = 0.00000000 exactly.
- **Mechanism check — MET.** Coxa surface placement error 0.01857 → **0.00794**, 45/48,
  sign p = 0.0000 (bar ≤ 0.0140, i.e. ≥25% reduction). The coxa is placed **2.3× closer to ground
  truth** with no free-form deformation available. The post-hoc reading from H_A2 replicates on
  data it was not derived from.
- **Guard-rails — pass.** No distal regression > 0.02 (`ti` −0.0007, `ta` −0.0072, `pt` +0.0011).
  `leg_acc` 0.9418 → **0.9682**, well above the 0.9415 floor.
- **Escape hatches — one FIRES.**

| check | `C13_uniform` | `_s1` | bar | |
|---|---|---|---|---|
| `betas` error vs GT | 3.0173 | 3.5078 | ≤ +10% rel | **FIRES (+16.3%)** |
| \|`betas_trans`\| | 0.016442 | 0.022590 | ≤ 2× | ok (1.37×) |
| \|`log_beta_scales`\| | 0.141992 | 0.210993 | ≤ 2× | ok (1.49×) |
| corr(coxa gain, `betas` error change) | — | r = +0.059, p = 0.69 | \|r\| ≤ 0.5 | ok |

## Verdict: BOUGHT-ELSEWHERE

The pre-registration says any one escape-hatch check firing makes this a trade, not a pass. The
`betas` magnitude check fires. **Verdict stands as written.**

**A defect in my own pre-registration, stated rather than resolved in the convenient direction:**
the two escape-hatch instruments disagree. The *magnitude* check fires (+16.3%) while the
*correlation* check — also registered — comes back flat (r = +0.059, p = 0.69), meaning the `betas`
degradation is **not** predicting which specimens get the coxa gain. If the shape block were paying
for the coxa win, the specimens that gained most should be the ones whose `betas` moved most; they
are not. The pre-registration did not specify precedence between the two instruments, and I am not
choosing the one that yields the nicer answer after seeing both. The conservative reading — the one
that does not claim a win — is BOUGHT-ELSEWHERE, and that is what is recorded.

## What this actually establishes

Three things, none of which depend on resolving the ambiguity above:

1. **The coxa is genuinely fixable, and the fix is known.** Carrying the correspondence term into
   D1 — which previously had none at all — recovers 119% of the coxal regression and places the
   coxa 2.3× closer to truth, replicated across two seeds, with free-form deformation disabled.
   After four mechanisms failed on the coxa (correspondence −0.009, reweighting −0.010, pose init
   −0.009, pose freezing −0.005), this is the first that works.
2. **It costs shape accuracy.** `betas` error rises 16.3%. Whether that trade is acceptable is a
   judgement about the downstream use, not something this experiment settles.
3. **`leg_acc` moves 0.9418 → 0.9682** against an E0 ceiling of 0.9926 — closing ~52% of the
   remaining gap to a perfect fit.

## What would settle it

One arm with precedence between the escape-hatch instruments fixed in advance, and a `beta_prior`
term restored to hold the shape block while the correspondence term acts. That is a **new
pre-registration**, not a re-reading of this one.

## Not claimed

- Synthetic only. bench50 untouched.
- Not shippable on this evidence: the registered verdict is a trade.
- Not a claim that `leg_acc` is solved — 0.9682 against a 0.9926 floor.
