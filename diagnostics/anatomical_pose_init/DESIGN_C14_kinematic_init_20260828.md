# Design note — C14: kinematics-aware initializer on the CSE backbone

**Not a pre-registration.** The bar cannot be fixed until C13 lands, because C13's result decides
what this has left to target. Written 2026-08-28 while C13 (job 3253515) is on arm 1 of 3.

## Facts probed before designing (repo, not recollection)

1. **The training corpus is already pose-labelled.** `diagnostics/moonshot/synth_b2_train_corrb06.npz`
   carries `verts (4000, 10235, 3)` **and `joint_rot (4000, 54, 3)`**, plus `names`. This is the
   same corpus that trained B2 and warm-started the C3 CSE head.
   *Consequence:* the "only 66 real training rows exist" objection does not bind here. A pose head
   trains on 4000 rows of exact GT pose, on the identical corpus, with no new data generation and
   no new sampler. What 66 rows ruled out was a pose predictor supervised by *real* fits; this is
   supervised by the synthetic generator that already produced every correspondence result.

2. **The kinematic tree is six clean 6-joint chains.** `J_names`/`kintree_table` (55 joints,
   `N_POSE=54`, `joint_rot[:, k]` ↔ joint `k+1` — root's 3 params are prepended separately in
   `fitter_3d/trainer.py:335-344`):

   | leg | co | tr | fe | ti | ta | pt | `joint_rot` slice |
   |---|---|---|---|---|---|---|---|
   | l_1_r | 6 | 7 | 8 | 9 | 10 | 11 | `[5:11]` |
   | l_2_r | 12 | … | | | | 17 | `[11:17]` |
   | l_3_r | 18 | | | | | 23 | `[17:23]` |
   | l_1_l | 26 | | | | | 31 | `[25:31]` |
   | l_2_l | 32 | | | | | 37 | `[31:37]` |
   | l_3_l | 38 | | | | | 43 | `[37:43]` |

   Every `co` joint's parent is the root (0). So **"one rigid transform per leg" is not an
   approximation that has to be projected onto the parameterisation — it is exactly `joint_rot`
   at the six `co` indices.** Stage 1 predicts 6×3 = 18 numbers that are natively fitter-legal;
   stage 2's chain residuals are the remaining 6×5×3 = 90. No IK, no retargeting, no least-squares
   fit of a transform onto angles — which is where `geom_leg_init.py`'s chain-IK broke.

3. **The hook is a plain npz contract.** `fitter_3d/optimise_hierarchical.py:377-396` requires
   exactly `names` (specimen stems) and `joint_rot` of shape `(N, 54, 3)`, seeds
   `smal.joint_rot`, and touches nothing else. Confirmed unchanged. `infer_leg_pose_init.py`
   already writes this format and is the template to copy.

## What this means for the build

The design in the plan survives probing, and gets *simpler* than described:

- **Stage 1 (leg-level).** Pool the frozen backbone's per-point features over each leg's predicted
  point set → predict the 6 `co` axis-angles directly. Fixes the collapsed-global-vector flaw of
  the original `smil_pointnet.py` while writing into the parameterisation natively.
- **Stage 2 (chain residuals).** Predict `tr…pt` in chain order, each conditioned on the running
  product of its ancestors. Manufactures the coherence the basin map found to be the recoverable
  error mode.
- **Stage 3 (twist).** Explicit per-joint twist component, gated by the measured `circum_gap`
  (`tr`/`fe` 0.38–0.44 well-conditioned; near-cylindrical joints are not) — predict twist only
  where the geometry supports it, leave it at the prior elsewhere. `circum_gap` is defined in
  `verify_metric_on_toy.py:56` as `(w1 − w2)/w1`; it is **not** yet computed per-joint on the real
  template, which is a prerequisite probe, not part of the model.

## The dependency on C13, stated concretely

C13 asks whether the coxa's residual is gradient-starved. Note the collision: **stage 1 of this
initializer is, precisely, a predictor of the six coxa rotations.**

- If C13 **PASSES** (weighting fixes the coxa), the coxal share of the residual is being taken by
  the loss, and C14's justification narrows to convergence speed/reliability and the distal chain.
- If C13 **FAILS** (coxa is not gradient-starved, and the offset lives in a part of the space the
  dense term cannot steer), then a *better coxal starting point* is one of the few remaining levers
  the dense term never had to steer at all — and C14's primary endpoint should be the same paired
  per-specimen `co` leg-level error C13 used, making the two directly comparable on one metric.

Either way the bar is pre-registered before training, per the standing discipline, and the honest
prior stays as stated: correspondence alone already moved `leg_acc` +0.058 at n=48, and C11/C12
twice showed a component improving in isolation without the fitter benefiting.

## Prerequisite probes (do not depend on C13; safe to run first)

1. Per-joint `circum_gap` on the real template, per leg segment — the stage-3 gate is currently a
   number measured on `tr`/`fe` only.
2. Ceiling check: seed `--init_joint_rot_from` with the corpus's own GT `joint_rot` on P48 and read
   `co` leg error. This measures what a *perfect* initializer buys under the current recipe, and
   costs one fit. If the GT-seeded ceiling is small, C14 is not worth training regardless of C13.

### Note on probe 2 and prior art

A GT-seeded arm exists (`gen_sweep_sbatch.sh:74`, `--init_joint_rot_from $D/ground_truth.npz`) and
the `ACI_noise5…noise30` sweep is its noise ladder — but those ran on the **n=12 ceiling corpus,
before the CSE correspondence term existed**. Both facts matter: n=12 is the exact power regime
that produced the false "leg_acc never moves" claim, and the ceiling of a better *initializer* is
conditional on the loss it initialises. So probe 2 is a re-measurement at n=48 on P48 under the
current `--cse_correspondence_from` recipe, not a repeat.
