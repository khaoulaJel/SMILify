# Pre-registration — D1 coxal degradation: which stage, and why

Written **before** any ablation is run, 2026-08-28. Step 1 (free, existing artifacts) is complete
and reported below because it localises the target; the bar for step 2 is fixed here in advance.

## Step 1 result — the regression is one transition, and it is real

Full stage trajectory of `C13_uniform` (CSE correspondence, zero pose init), auditing every
existing npz. Nothing was re-run.

| stage | leg_acc | **co** | tr | fe | ti | ta | pt |
|---|---|---|---|---|---|---|---|
| H0_body | 0.9430 | 0.3345 | 0.0888 | 0.0901 | 0.0927 | 0.1151 | 0.0966 |
| H1_legs | 0.9642 | 0.2621 | 0.0547 | 0.0473 | 0.0446 | 0.0530 | 0.0621 |
| H2_joint | 0.9649 | **0.2564** | 0.0539 | 0.0445 | 0.0423 | 0.0534 | 0.0579 |
| H3_deform | 0.9654 | 0.2602 | 0.0563 | 0.0441 | 0.0442 | 0.0508 | 0.0523 |
| **D1 coarse** | 0.9444 | **0.4015** | 0.0838 | 0.0623 | 0.0375 | 0.0434 | 0.0522 |
| D1 fine | 0.9415 | 0.4115 | 0.0829 | 0.0611 | 0.0396 | 0.0485 | 0.0587 |
| *floor (E0)* | *0.9926* | *0.0908* | *0.0110* | *0.0043* | *0.0003* | *0.0021* | *0.0010* |

Paired per-specimen, stage to stage:

| transition | Δ co | worse | sign p |
|---|---|---|---|
| H0 → H1 | −0.0724 | 7/48 | 0.0000 |
| H1 → H2 | −0.0057 | 21/48 | 0.47 |
| H2 → H3 | +0.0039 | 28/48 | 0.31 |
| **H3 → D1 coarse** | **+0.1412** | **42/48** | **0.0000** |
| D1 coarse → D1 fine | +0.0101 | 32/48 | 0.029 |

**91% of the loss happens in the single H3 → D1-coarse transition**, on 42 of 48 specimens
(H2 → D1 fine overall: worse 42/48, Wilcoxon p = 1.25e-09). The 0.2564 is stable, not a logging
artifact — it is flat across H1/H2/H3, and its best specimen reaches **0.091**, i.e. the E0 floor
exactly. The hierarchical stage genuinely solves the coxa on some specimens.

## The hypothesis this changes

The thread was opened expecting the culprit to be one of D1's extra loss terms (edge, laplacian,
symmetry, offset, midline) pulling the coxa away. Probing the code first says that is probably the
wrong hypothesis:

**`optimise_moonshot.py` and `trainer_moonshot.py` contain no correspondence term at all.**
`grep -n "cse\|dense_gt"` returns nothing in either file. The CSE correspondence term
(`w_cse_corr`) exists only in the hierarchical trainer. D1 also enables `deform_verts`
(`scheme: all`), which the hier recipe disabled via `--deform_its 0`.

So the leading explanation is not that a D1 term *pushes* the coxa away, but that D1 **drops the
term that was holding it there**, and the solution relaxes back toward the chamfer optimum — which
is exactly where the coxa was before correspondence was introduced (`P48_zero` co = 0.4204;
D1 fine = 0.4115).

**A natural experiment already on disk corroborates this and was recorded before this document.**
`C14p_gtinit_nocse` has no correspondence anywhere, so it has nothing to lose in D1 — and its coxa
*improves* across D1 (0.4564 → 0.4114), the opposite direction to `C13_uniform` (0.2564 → 0.4115).
That differential is predicted by "D1 drops the holding term" and not by "a D1 term pushes it away".

## Arms and the bar, fixed before running

Two hypotheses, tested against the same endpoint.

**H_A — D1 drops the correspondence term (primary).** Requires a code change: carry `w_cse_corr`
into `trainer_moonshot`. Arm: `D1_with_cse`, standard D1_low recipe plus the correspondence term at
the hier stage's weight, initialised from the same `C13_uniform_hier/H2_joint.npz`.

**H_B — an existing D1 term pushes the coxa away (alternative).** Config-only, no code change.
Arms, each zeroing exactly one weight in `Stage_2_deform_coarse` and `Stage_3_deform_fine` of
`D1_low.yaml`, everything else identical: `D1_no_edge`, `D1_no_sym`, `D1_no_offset`,
`D1_no_midline`, `D1_no_lap`. Plus `D1_no_deform` (`scheme: pose`, removing `deform_verts`), which
is the other structural difference from the hier recipe.

Primary endpoint: paired per-specimen `co` leg-level error against `C13_uniform` (D1 fine, 0.4115),
n=48, sign test + Wilcoxon + paired-t. The gap to recover is **0.4115 − 0.2564 = 0.1551**; read
against the E0 floor of 0.0908, never against 0.

- **H_A CONFIRMED** — `D1_with_cse` recovers **≥ 70%** of the gap (co ≤ **0.3030**) with
  sign p < 0.05. Then the fix is to carry the correspondence term through D1, and it is a real,
  shippable change rather than another characterization.
- **H_B CONFIRMED** — some single-term removal recovers ≥ 70% of the gap with sign p < 0.05. The
  term named must be **fixed in advance of looking**: on the reasoning above I predict **none of
  the H_B arms clears the bar**, and I am recording that prediction now so a post-hoc "it was
  `w_offset` all along" cannot be told as if it were expected.
- **PARTIAL** — recovery ≥ 0.05 but < 70%, sign p < 0.05. Report as bounded; combine only if both
  hypotheses show partial effects.
- **FAIL / NEITHER** — nothing clears 0.05. Then the degradation is a property of D1's
  re-optimization as a whole, not any single removable component, and the thread closes with that
  as the finding.

Guard-rails, fixed now:
- **Distal must not regress** by > 0.02 (`ti`/`ta`/`pt`), or a coxal gain is a trade.
- **`leg_acc` must not fall** below `C13_uniform`'s 0.9415.
- Any arm that beats **H2_joint's own 0.2564** must be treated as suspicious and re-checked before
  being believed — D1 adding information it does not have would indicate a bug, not a win.

## What will NOT be claimed

- No real-scan claim. P48 is synthetic and model-generated; bench50 is separate.
- No claim that this fixes `leg_acc` in general — this targets a regression, i.e. recovering ground
  already held, not new ground.
- If H_A confirms, the shippable change is "carry correspondence into D1", **not** "the coxa is
  solved": 0.2564 is still far above the 0.0908 floor.
