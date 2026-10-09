# H_A4 result — PRIOR TOO WEAK. The coxa thread closes: the fix is real and it costs shape accuracy.

Bar and instrument precedence fixed in `PREREGISTRATION_HA4_betaprior_20260828.md`, committed
before job 3279944 ran. Arm = `D1_with_cse_nodeform_s1` + `w_beta_prior: 0.002`, one variable.

| run | leg_acc | **co** | tr | fe | ti | ta | pt |
|---|---|---|---|---|---|---|---|
| `C13_uniform` | 0.9415 | 0.4115 | 0.0829 | 0.0611 | 0.0396 | 0.0485 | 0.0587 |
| `D1_with_cse_nodeform_s1` | 0.9689 | 0.2265 | 0.0519 | 0.0363 | 0.0389 | 0.0414 | 0.0598 |
| `D1_cse_nodeform_betaprior` | 0.9680 | **0.2311** | 0.0503 | 0.0375 | 0.0384 | 0.0425 | 0.0403 |

| registered condition | result | |
|---|---|---|
| (1) co ≤ 0.3030, sign p < 0.05 | 0.4115 → **0.2311**, 116.3%, 41/48, p = 0.0000 | **MET** |
| (2) coxa placement ≤ 0.0140, sign p < 0.05 | 0.01857 → **0.00833**, 45/48, p = 0.0000 | **MET** |
| (3) `betas` err ≤ 3.3190 (+10%) | 3.0173 → **4.8531 (+60.8%)** | **NOT MET** |
| (4) guards | distal −0.0012, leg_acc 0.9675, deform 0.00000000 | **MET** |

**Verdict: PRIOR TOO WEAK** — conditions 1 and 2 hold, 3 does not.

## The registered label is right; its implied mechanism is wrong

"PRIOR TOO WEAK" was written expecting the prior to under-control the cost. It did the opposite:
`betas` error went from 3.5078 *without* the prior to **4.8531 with it** — adding the Mahalanobis
shape prior made shape accuracy **substantially worse**, not marginally better.

That is explicable and worth recording: the prior pulls `betas` toward the *model's own shape
distribution*, which is not the same direction as the specimens' true `betas`. On a corpus whose
targets were generated with sampled shape parameters, "typical under the model" and "correct for
this specimen" diverge, so the prior actively pulls away from ground truth. `w_beta_prior` is
therefore **not the control for this trade** — it is a second, independent source of shape error.

Per the pre-registration, **no weight sweep**: sweeping `w_beta_prior` until something passes is
tuning, not measurement, and it was ruled out in advance.

Diagnostic (gates nothing, per the registered precedence): corr(coxa gain, `betas` error change)
r = +0.127, p = 0.39 — flat again, as in H_A3. Across both arms the shape degradation does **not**
predict which specimens gain on the coxa. `|betas_trans|` 1.60×, `|log_beta_scales|` 1.63×.

## The coxa thread closes here. What it established.

**The fix, and it is real:** carry the CSE correspondence term into D1, which previously had **no
correspondence term of any kind**, with `deform_verts` disabled.

- `co` leg-level error **0.4115 → 0.2265**, ~119% of the regression recovered, 43/48, p = 0.0000
- coxa surface placement **2.3× closer to ground truth** (0.01857 → 0.00794), 45/48, p = 0.0000
- `leg_acc` **0.9418 → 0.9682**, closing ~52% of the remaining gap to the E0 ceiling of 0.9926
- replicated across two seeds; `deform_verts` pinned at exactly 0, so it is not bought by
  free-form deformation

**The cost, stated rather than buried:** `betas` error vs GT rises **+16.3%** in the shipping
configuration (`D1_with_cse_nodeform_s1`, no prior). It is not controllable by `w_beta_prior`, and
across two arms it is not correlated with the coxa gain — consistent with an independent drift
rather than shape paying for placement, though that remains diagnostic, not established.

**Recommended configuration if this ships:** `D1_with_cse_nodeform_s1` — `scheme: pose`,
`--cse_correspondence_from`, **no** `w_beta_prior`. The prior arm is strictly worse on shape and
marginally worse on the coxa.

**Context for how hard-won this is:** four mechanisms failed on the coxa before this one —
correspondence alone (−0.009), gradient reweighting (−0.010, saturating), pose initialization
(−0.009, n.s.), and pose freezing at ground truth (−0.005, n.s.). The coxa was the single most
resistant problem in the investigation.

Per the pre-registration, **no further variant is opened from this result.**

## Not claimed

- Synthetic only. bench50 is untouched; nothing here is a real-scan claim.
- The coxa is ~16.7% of total correspondence error. A bounded win.
- `leg_acc` is not solved at 0.9682 against a 0.9926 floor.
- The shape cost's *cause* is not established — only its size, and that one candidate control
  makes it worse.
