# Phase 10 Design: Correspondence-Supervised Network + Fitter Integration

**Status: design only, no code written against this doc yet.** Written for review before any
implementation starts, per the scale of the commitment (a new data generator, a new architecture,
and new engineering inside the mature D1 chain).

## 0. Why this, why now (the evidence this design is built from, not the literature sweep alone)

Every geometry-only candidate this investigation tried failed for the same diagnosed reason, and
there is no more real data to fix it with:

- The disentangling oracle (`coherent_multi_estimator_oracle_PROBE.py`, 2026-08-25) proved real leg
  poses distribute ~50 deg of bend across tr/fe/ti/ta as well as the coxa (`coxa_only_gap`
  mean=50.03 deg, median=49.82 deg) -- so no coxa-only estimator, however accurate its direction
  estimate, can recover the true per-joint decomposition. This is a structural limit, not an
  estimation-quality gap.
- Three independently-built direction estimators (PCA, cluster-axis, tip-direction) confirmed this
  empirically: all three landed at 0.804-0.832 aggregate leg_acc, all below zero-init's 0.867-0.872,
  despite being mechanistically distinct (median cross-estimator agreement of only 10-16 deg on real
  scans, confirming they are not just copies of each other).
- Pooling all 5 candidates (zero/IK/PCA/cluster/tipdir) moved the oracle ceiling by only +0.009
  (0.906->0.915) while the GT-free (fscore@0.02) selector's own correctness rate *dropped* from 9/12
  to 7/12 -- more options did not buy a meaningfully higher ceiling, and made the practical selector
  less trustworthy, not more.
- A learned linear pose-subspace (leave-one-out PCA over true [co,tr,fe,ti,ta] rotations, this
  session) also failed: explained-variance ratios were nearly flat (0.14, 0.12, 0.10, 0.09, 0.08...)
  rather than concentrated in a few modes, and every mode count tested performed at or worse than
  raw box-constrained IK. Checked directly: there is no larger corpus of real, ground-truthed leg
  poses anywhere in this project to fix this with. `joint_limits.py`'s per-axis bounds are
  hand-authored in Blender by an artist (`docs/joint_limits_user_guide.md`), not statistically
  calibrated from digitized poses. The AntScan real-specimen pipeline (`custom_processing/`) has no
  per-joint rotation ground truth at all -- extracting it would require solving the correspondence
  problem this investigation exists to fix, which is circular. The 12 `synth_clean` specimens are
  the only source of ground-truth joint rotations in the repo.

**The reframe:** stop trying to recover *pose* (54 joint angles) from geometry that structurally
underdetermines it. Recover *correspondence* (per-point: which template vertex is this) instead --
a better-posed problem that does not require resolving twist ambiguity or the whole chain's bend to
be recoverable from a single estimate. Pose becomes something derived from correspondence
afterward (the same way D1's own optimizer already recovers pose once its data term converges),
not something regressed directly.

## 1. Fix the training-data generator

**What's actually broken, verified in the code, not assumed:** `sample_smil_model.py`'s
`generate_random_parameters` samples `random_joint_rot = pose_scale * get_random_values((batch,
N_POSE, 3))` -- i.i.d. Gaussian (or uniform) noise, independently per joint, per axis, no coupling
between co/tr/fe/ti/ta at all (`sample_smil_model.py:82`). This is not a hypothetical training-data
concern: `diagnostics/moonshot/make_synth_corpus.py` imports this exact function
(`make_synth_corpus.py:34-38`) with `pose_scale=0.25` to generate `synth_clean` itself. **Every
leg_acc number in this entire investigation, including the disentangling probe's own ground
truth, comes from this i.i.d.-noise generator.** That has a real consequence beyond training data:
it means the coxa_only_gap and coherent-family failure numbers measured so far might themselves be
partly an artifact of unnaturally decorrelated ground-truth poses, not purely a fact about real ant
articulation -- worth re-measuring once a better generator exists, not just assumed to transfer.

**The fix:** build a new sampler that generates correlated, whole-chain-coherent bends, informed
directly by what the basin map already measured about which error structures are anatomically
plausible: `coherent` (root-only rotation, downstream at rest) was the least damaging structured
perturbation at every tested magnitude; `proximal` (co/tr/fe-concentrated) was the most damaging.
A realistic sampler should draw from a distribution where per-joint rotations are *coupled* (e.g. a
shared per-leg "bend budget" split across joints with smoothly decaying influence toward the tip,
or an explicit low-order Markov chain along the joint sequence) rather than treating all 5 joints
as independent draws. This is now a specified fix to a measured problem, not a guess.

## 2. Fix the supervision target

Train on dense per-point -> template-vertex correspondence, not joint angles and not chamfer
distance. Synthetic generation gives this for free: every sampled point comes from a known face of
a known-topology mesh, so its ground-truth template vertex (or barycentric-nearest vertex) is
already determined at generation time -- no extra labeling cost.

This directly kills the first architectural flaw found in `smil_pointnet.py`, independent of
anything else in this design: chamfer distance (what `SMILPointNet`/`SMILPointNet2` are trained
against today) cannot see identity -- two points that are Euclidean-close but anatomically
unrelated (e.g. a mesothoracic coxa point and a neighboring leg's point, the exact
`assign_points_to_legs_coxa` starvation defect found 2026-08-20) score as if correct. A
correspondence label makes that failure directly visible in the loss, by construction, rather than
averaged away inside a distance metric.

## 3. Fix the architecture: decode before collapse

**Verified, not theoretical:** `smil_pointnet.py`'s `SMILPointNet2.forward` runs `sa1` (512 local
groups, multi-scale) -> `sa2` (128 local groups) -> `sa3 = PointNetSetAbstraction(..., group_all=True)`,
whose output is explicitly "a global feature of 1024 channels" (`smil_pointnet.py:380-381`), and
`ALL` output parameters (global_rot, joint_rot, betas, trans, scales) are regressed from that one
pooled vector (`smil_pointnet.py:454-457`). Per-region spatial identity that `sa1`/`sa2` still
carry is thrown away before any parameter is decoded -- the same collapse-before-decode failure
mode literature on human pose estimation (PARE) diagnoses for exactly this class of architecture.

A correspondence head must decode from features that are still spatially localized -- read from
`sa1`/`sa2`-level per-region features (or an explicit feature-propagation/upsampling path back to
per-point resolution, as PointNet++ segmentation heads already do), not from `sa3`'s single pooled
vector. This is architecturally close to GeoTransformer / D3Feat / REGTR / TANet (named in
`LITERATURE_SWEEP_20260825.md`), now assigned a specific, evidence-justified role -- the correct
answer to a diagnosed architectural bug in this project's own existing network, not a generic
citation.

## 4. New fitter hook: dense correspondence as a data term, not just an init

This is the one piece of new engineering that touches the mature, heavily-tested D1 chain, and
should be scoped and reviewed as such, separately from the network/data-generator work above.

**What exists today, confirmed by reading the code:** `optimise_hierarchical.py --init_joint_rot_from`
loads a `joint_rot` array and seeds it as the STARTING value before any hierarchical stage begins
iterating (`optimise_hierarchical.py:317-336`) -- it has no further effect once optimization starts.
The actual per-iteration data term lives in `trainer_hierarchical.py`'s `TargetPartition`
(`trainer_hierarchical.py:205`), which assigns each target point to a model group via nearest-
fitted-vertex `knn_points` search (`trainer_hierarchical.py:257,265`), consumed by
`_partitioned_chamfer` (`trainer_hierarchical.py:430`) to compute the grouped/partitioned chamfer
loss every iteration (`trainer_hierarchical.py:491,496`).

**The honest version of this idea** does not stop at a better starting pose -- it gives the
optimizer a learned, per-point correspondence prior it can consult every iteration, not just once
at t=0. Two concrete integration shapes to choose between during implementation (not decided here):
  (a) bias `TargetPartition`'s own nearest-fitted-vertex assignment toward the network's predicted
      correspondence when the two disagree (soft prior on group membership), or
  (b) add a new, separate loss term alongside `_partitioned_chamfer` that directly penalizes the
      fitted mesh's per-vertex position against the network's predicted corresponding target point,
      independent of the current pose's own nearest-neighbor guess.
(b) is more surgical (additive term, does not change `TargetPartition`'s existing behavior when the
network is absent or low-confidence) and is the safer default to prototype first; (a) is the
stronger claim (actually replaces the current correspondence mechanism) and should only be
attempted after (b) is validated. Either way: new loss term(s), a new CLI flag analogous to
`--init_joint_rot_from` but data-term-facing (e.g. `--correspondence_prior_from`), and an ablation
against the existing `TargetPartition`-only baseline on the same corpora already used throughout
this investigation, before this is ever turned on by default.

## 5. The real, unresolved risk: sim-to-real transfer

Nothing in sections 1-4 tells us whether a network trained purely on synthetic correspondence
generalizes to real, damaged, noisy bench50 scans. This is the same open question this project's
own `H1`/pf_init history never fully resolved (`FINAL_REPORT.md` §7.9) -- now for a harder target.
It should be named as the central research risk of this whole plan, not discovered mid-training.

The one lever actually available: bench50's real scans exist without joint-rotation ground truth,
but D1's own converged output on them (after any of the existing GT-free proxies pass a quality
bar) can serve as pseudo-labels for a self-training/fine-tuning pass -- using the optimizer's own
results to adapt the network to real geometry, rather than assuming synthetic-only training
transfers by default. This should be planned as an explicit phase of the work (train on synthetic
-> evaluate the sim-to-real gap directly -> fine-tune on bench50 pseudo-labels if the gap is large),
not an afterthought if synthetic-only training underperforms.

## 5b. A second, independent gap: even PERFECT correspondence has a ceiling well short of 1.0

Confirmed empirically, not assumed (`RESULTS_correspondence_oracle_20260825.md`): a dense,
per-vertex, ground-truth correspondence oracle (`Dense_GT_oracle`) — perfect information, fed
throughout the entire D1 fit — raises `seg_acc` (within-part correctness) from 0.831 to only 0.874.
Before treating that as a shortfall, the metric's own ceiling was measured directly: feeding the
ground-truth mesh itself through the identical evaluation (as if it were the fitted output) scores
0.974, not 1.0, due to inherent near-boundary point-sampling ambiguity. So the real residual gap is
~0.10 (0.874 -> 0.974), not ~0.17 (0.874 -> 1.0) — smaller than it first looks, but still real and
substantial.

Two explanations were distinguished, not assumed: a weight-dominance check (`w_dense_gt` raised
50x, confirmed to actually dominate the objective via a sharp rise in edge distortion, 0.05->0.36-
0.80) left `seg_acc` unchanged (0.870 vs 0.874, not significant). **The gap is not a matter of the
correspondence signal being under-weighted relative to smoothness/edge terms** -- it persists even
when that signal completely dominates every other term. Something else -- shape-space capacity,
mesh topology/resolution, or the model's own deformation limits -- caps how well the fitted mesh
can realize even a maximally-weighted, perfectly-correct correspondence signal.

**Consequence for this design:** the eventual network's realistic ceiling is not 1.0, and not even
0.974 -- it is bounded above by ~0.874 (the PERFECT-information ceiling), and a learned, imperfect
correspondence predictor will land somewhere below that, not somewhere below 1.0. This is a
different, independent risk from section 5's sim-to-real gap (that one is about train/test
distribution shift; this one is about the fitter's own capacity to realize a correct signal once
given one) and should be sized and reported separately, not folded together. Before committing
further engineering to closing THIS specific gap (as opposed to the sim-to-real one), it is worth
identifying which of shape-space capacity / mesh resolution / regularization structure is
responsible -- that diagnosis is itself a cheap, valuable next probe, not a reason to abandon the
correspondence direction (the broad, 12/12-specimen `seg_acc` gain this oracle also produced is
real and still the strongest validated signal in this investigation).

## 6. Validation plan (reuse, don't rebuild)

- **Synthetic correspondence accuracy:** the same `leg_acc`/`within_leg_segment_confusion`/
  `correspondence_confusion.json` pipeline (`diagnostics/correspondence_accuracy/run_audit.py`)
  already used for every candidate in this investigation -- cheap and near-infinite supply once
  the new sampler (section 1) is in place, since ground truth is generated, not measured.
- **Real (bench50) evaluation, GT-free:** the same `fscore@0.02`-style surface-fit proxies already
  used for candidate selection, plus population-level correspondence consistency once `probe_19`
  is recovered (currently missing from this WSL checkout, methodology documented in
  `diagnostics/moonshot/REPORT.md` §6.4 -- same cluster-migration loss as `labels.py`/`confusion.py`/
  `geodesic.py` earlier this investigation, recoverable the same way).
- No new evaluation machinery is proposed here -- this design deliberately reuses the discipline
  this investigation already earned rather than building a parallel one.

## 7. Open scope questions for review before implementation starts

1. Confidence-weighting: should the section 4 data term be unconditionally trusted, or gated by a
   per-point network confidence score (analogous to how `TargetPartition` already gates on
   distance)? Not decided here.
2. Correlated-pose sampler (section 1) parameters (bend-coupling strength, per-leg-position
   differences) should be validated against the basin map's own coherent/proximal/distal numbers
   as a sanity check before being trusted as "realistic," not assumed correct on first pass.
3. Section 4's two integration shapes (additive term vs. replacing `TargetPartition`'s own
   assignment) should be an early, cheap ablation, not a committed choice at design time.
4. Section 5's fine-tuning phase needs a concrete trigger condition (how large a measured sim-to-
   real gap justifies it) defined before training starts, not decided reactively.

## 8. Non-goals

This design does not propose replacing D1's optimizer, deform stages, or shape/beta fitting -- only
the leg-pose correspondence mechanism feeding into the existing hierarchical/moonshot pipeline.
`smil_pointnet.py`'s existing chamfer-supervised regressor is not modified in place; this is a new,
separate correspondence-supervised network, evaluated head-to-head against it and against every
existing candidate in this investigation before any claim of improvement.
