# Full Reasoning Chain: Learned Leg-Pose Initializer Investigation (2026-08-20 → 21)

Preserved narrative of the whole investigation, in order, so nothing gets lost as this thread
gets long. Each `RESULTS_*.md` file in this directory is the clinical record of one stage; this
file is the connective reasoning between them — why each step happened, what it ruled in/out,
and what it changed about the next step.

## 0. Starting point (inherited from a prior session)

Prior work: `diagnostics/anatomical_pose_init/` already contained a FAILED experiment
(`out_ceiling_20260820/RESULTS.md`, the "Anatomical Initialization Ceiling Test") — a
hand-designed, correspondence-free leg-pose initializer (`cheap_anatomical_init.py`, Arm B)
scored *worse* than zero-init on almost every metric, despite being anatomically motivated. That
experiment's own conclusion: coarse geometric heuristics aren't good enough to be worth learning
from yet; a learned regressor trained directly against GT poses might do better. This session
picked up that thread: train a PointNet-style regressor on synthetic data to predict leg pose
from a point cloud, see if it beats zero-init.

## 1. Model path + a real bug in the training script (first correction round)

User pointed out two things needing fixing before any training:
1. The 5000-specimen training corpus should be generated with `SMIL_OmniAnt.pkl` (10229 verts,
   55 joints) — the model actually used to generate it.
2. `train_leg_pose_regressor.py` had `self.joint_rot = gt["joint_rot"]` (matrices, `(N,54,3,3)`)
   feeding straight into `axis_angle_to_matrix()` (which expects axis-angle, `(...,3)`) — a real
   shape-mismatch bug. Fix: use `gt["joint_rot_aa"]` instead (the axis-angle field already in the
   corpus). Also flagged the same pattern in `infer_leg_pose_init.py`.

I verified both independently (probed the actual `.npz` shapes before editing, per this project's
"probe before editing" rule), fixed both, ran a 2-epoch smoke test (passed, no dimension errors),
then a full 200-epoch training run on Slurm. **Result**: val loss bottomed at epoch ~30 (44.503°)
and monotonically got worse through epoch 199 (46.353°) — clear overfitting, checkpoint saving
worked correctly (kept the epoch-30 model), but 170 epochs of GPU time were wasted.

## 2. The account/compute question (side thread, resolved)

Discovered mid-session that jobs were running under `--account=default` (personal quota) instead
of a project account, despite being listed as a member of the "SMILify" project. Investigated via
`sacctmgr` — confirmed no project Slurm account existed for the user yet. Fabian (project owner)
added the user to Slurm account `rwth2151` mid-session; verified it worked via a dry-run submit
and `sacctmgr show associations`, then switched all subsequent sbatch scripts to
`--account=rwth2151`. (This is unrelated to the science but explains why job scripts reference
that account throughout.)

## 3. The real topology bug (this is the pivot point of the whole session)

Before running the 200-epoch job "for real," ran `infer_leg_pose_init.py` against the held-out
`synth_clean` corpus (the actual 12-specimen benchmark with the historical 23.1°/28.5°/32.0°
numbers) — it crashed: `KeyError: 'joint_rot_aa'`.

Investigated rather than patched blindly: **`synth_clean` was generated with a completely
different model than the training corpus.** `synth_large` (training) used `SMIL_OmniAnt.pkl`
(10229 verts, 13 shape PCs). `synth_clean` (the actual eval benchmark) and the real fitter both
use `OmniAnt_25PCs_joint_limited.pkl` (10235 verts, 25 shape PCs). The 200-epoch checkpoint just
trained was topologically invalid for the benchmark it was supposed to be compared against.

Root cause, traced through `config.py`: `SMAL_FILE`/`N_BETAS` are resolved from the
`SMILIFY_SMAL_FILE` env var **at `import config` time**, before any argparse runs. So a `--model`
CLI flag on the generator script can't actually switch the model — by the time `main()` reads
`args.model`, `N_BETAS` is already locked in from whatever the env var said at import. Fixed
`generate_synth_large.py` to assert `--model` matches the resolved `config.SMAL_FILE` and fail
loudly instead of silently generating the wrong topology (this is the guard that catches this
exact mistake if it happens again).

Regenerated the corpus (`synth_large_25pc`, correct topology, validated: shapes, orthogonality
error ~6e-7), retrained (50 epochs this time, not 200 — the overfitting point was already known).
**Result**: `infer_leg_pose_init.py` against `synth_clean` now worked: mean pose error 22.79-
22.862° (two runs, minor seed variance) — landing right next to zero-init's own 23.09°/23.1°.

## 4. Correcting the "23.1° = best existing initializer" misconception

At this point the framing (inherited from earlier in the conversation) was "learned (22.86°) beats
best existing initializer (23.1°)." User caught that this was wrong: read
`out_ceiling_20260820/RESULTS.md` and found **23.1° is zero-init's own pre-optimization error**,
not a separate hand-designed initializer. The one real hand-designed initializer that exists
(`cheap_anatomical_init.py`, Arm B) scored *worse* pre-opt (28.5°) and *catastrophically* worse
post-fit (leg_acc 0.732 vs zero-init's 0.857) — the FAILURE result from stage 0. So "beating 23.1°"
was never a meaningful target; the real comparison needed to be against zero-init directly, using
this project's own established post-fit metrics, not just a pose-regression number.

## 5. Arms A/B/C: the real, apples-to-apples comparison (`RESULTS_ABC_DEF.md`, part 1)

Re-ran the full comparison using this project's own evaluation code (`run_audit.py` for
`leg_acc`/`all_feature_R`, `eval_run.py` for Chamfer/F-score), fitting all three arms through the
identical D1 recipe:

| Arm | Init error | leg_acc | all_feature_R | Chamfer | F@0.01 |
|---|---:|---:|---:|---:|---:|
| A zero | 23.09° | 0.857 | 0.745 | 0.000264 | 0.6951 |
| B cheap_anatomical | 28.52° | 0.732 | 0.518 | 0.000417 | 0.5948 |
| **C learned** | 22.79° | **0.901** | **0.761** | **0.000226** | **0.7153** |

C wins every metric. At near-identical scalar pose error to zero-init, it produces a measurably
better final fit. This raised the real question: *why* — since scalar pose error alone doesn't
explain it (B had worse scalar error too, but catastrophically worse fit — so error magnitude
isn't the whole story even before this point).

## 6. Arms D/E/F: the mechanism (`RESULTS_ABC_DEF.md`, part 2) — the strongest single result

Hypothesis: it's not the magnitude of the error, it's *where* the error is (proximal vs. distal
in the kinematic chain) that determines whether the optimizer converges well. Tested directly:
constructed `theta_0 = theta_GT + epsilon` with `epsilon` at matched magnitude (~23°, verified
per-specimen band 22-24°) but different STRUCTURE — random (D), proximal-concentrated (E,
coxa/trochanter/femur), distal-concentrated (F, tibia/tarsus/pretarsus). Perturbation
construction grounded in the literature before implementing (axis uniform on S², angle magnitude
fixed directly — avoids the well-known small-angle bias of naive angle~Uniform sampling; Kuffner
2004, Yershova & LaValle 2004).

**Result, consistent across every metric family independently:**

| Arm | leg_acc | all_feature_R | Chamfer | F@0.01 |
|---|---:|---:|---:|---:|
| D random | 0.883 | 0.783 | 0.000228 | 0.7218 |
| **E proximal** | **0.768** | **0.648** | **0.000393** | **0.6443** |
| **F distal** | **0.939** | **0.816** | **0.000218** | **0.7329** |

At essentially identical scalar error, proximal-concentrated error collapses the fit
(comparable to the earlier FAILED Arm B), distal-concentrated error is nearly harmless. Consistent
with forward-kinematic error propagation: a proximal joint's rotation error displaces the entire
downstream chain; a distal joint's error stays local. **This explains why the learned
initializer (C) beat zero-init (A)**: a joint-wise error breakdown (A vs C) showed C's modest
improvement over A was concentrated on proximal joints (coxa, trochanter) — small correction,
large leverage, exactly where D/E/F shows it matters most. One important correction from the user
here: don't say "F beats gt-init" just because its post-fit numbers were numerically slightly
higher on this n=12 run — that's "F slightly exceeds the gt-init reference on some post-fit
metrics in this run," not "F is a better initialization than ground truth." Scientific precision
in the write-up mattered more than the flashier claim.

## 7. Stage 4A/B: does this transfer to real, correspondence-free scans?

The learned initializer's whole input pipeline (`vertex_labels`, `body_core_canonicalize`) is
indexed by TEMPLATE vertex ID — meaningful only because synthetic corpora are deformations of the
template itself with exact vertex correspondence preserved. A real scan has no such
correspondence. Investigated (before building anything new) what the existing codebase already
had for scan→anatomy association: found `cheap_anatomical_init.py`'s `estimate_global_pose` (PCA
rigid alignment, correspondence-free) and `geom_leg_init.py`'s `analytic_coxa_anchors`/
`assign_points_to_legs` (6 coxa positions from a rigid alignment + template rest geometry, then
nearest-anchor point assignment) — most of a correspondence-free pipeline already existed.

**Validating it against `synth_clean`'s known ground-truth vertex labels (before trusting it)
found a real, previously-undiscovered defect**: `assign_points_to_legs` (single-coxa-point
nearest-anchor assignment) systematically starves the middle (mesothoracic) legs — ML/MR recall
only 0.349 pooled, reproduced on data with KNOWN correspondence, not scan noise. This also revises
the historical Arm B (`cheap_anatomical_init.py`) verdict: it used this same defective assignment
internally, so its FAILURE result stands, but its attribution to "the IK heuristic being bad" is
now unconfirmed — the upstream point-to-leg assignment may have contributed.

**Fix, grounded in the literature first** (RigNet/classical skeletal rigging: distance to nearest
*bone segment*, not nearest joint, is the standard primitive): implemented
`assign_points_to_legs_chain` (nearest-articulated-rest-pose-polyline, not nearest-coxa-point) in
the shared `fitter_3d/geom_leg_init.py`, preserving the old function
(`assign_points_to_legs_coxa`) unchanged for historical auditability. **Validated**: macro-F1
0.656→0.831, ML/MR recall 0.349→0.768, wins 11/12 specimens. Not a complete fix (one residual
hard case, `synth_011`'s `l2_r`, still poor under both hard and soft assignment — traced to
genuine chain-geometry ambiguity, not an argmin-destroys-information problem, via a GT-posed-chain
ceiling probe and a temperature-sweep soft-assignment diagnostic). Found and fixed the *same*
single-point-for-an-extended-region defect a second time, independently, for the body-core class
(core recall 0.366→0.663 with a body-core *chain*, head→thorax→gaster, instead of a single root
point) — see `RESULTS_correspondence_free_representation.md` for the full validation detail
(confusion matrices, per-temperature diagnostics, the body-core threshold sweep that motivated
not using an arbitrary threshold).

## 8. Real bench50 test — first negative, then explained (`RESULTS_bench50_G1_vs_G3.md`)

Built the full raw-scan pipeline (normalize scale — verified empirically matches
`fitter_3d.utils.load_meshes`'s convention, not assumed — actual PCA global rotation, not the
identity shortcut synth-only code uses, 7-way chain assignment, existing trained checkpoint).
Before fitting anything, checked predictions were even plausible: stable across resamples (0.48°
mean variance), sane magnitude (11.5°), consistent front>hind>middle leg-position structure across
50 taxonomically diverse real specimens never seen in training. Cleared that gate, then ran real
fits (G1 = this learned init, G3 = zero-init, both identity global orientation — deliberately
isolated from the separate, not-yet-built global-orientation question).

**Result: G1 (learned) measurably LOST to G3 (zero) on real data** — chamfer 0.000697 vs 0.000637,
F@0.01 0.4762 vs 0.4983, won only 20/50 specimens. Opposite of the synthetic result. Ruled out one
candidate explanation (PCA rotation-magnitude misalignment — no correlation with win/loss).
Best-supported remaining hypothesis: **domain gap in training pose-noise magnitude** — the network
was trained on synthetic data with large injected pose noise (~23° mean), but real pinned museum
specimens are plausibly much closer to rest pose, making zero-init an unfairly strong baseline on
real data specifically (it's closer to correct a priori there) while making the network
over-correct.

**Tested this directly rather than leaving it as a hypothesis** (three full retrain+refit cycles
overnight, each: regenerate 5000-specimen corpus at a lower `--pose-scale`, retrain 50 epochs,
regenerate bench50 init, refit all 50 specimens, eval):

| Training pose-noise | Chamfer L2 | F@0.01 | Wins vs G3 | Paired p vs G3 |
|---|---:|---:|---:|---:|
| G1: 23° (original) | 0.000697 | 0.4762 | 20/50 | **0.013** (significantly worse) |
| G1b: 13.8° | 0.000649 | 0.4972 | 19/50 | 0.850 (parity) |
| **G1c: 8.85°** | **0.000637** | **0.4974** | **26/50** | 0.855 (parity, but chamfer now tied, majority win-rate) |

Clean, monotonic dose-response across three independent retrains: as training pose-noise magnitude
decreases toward a "near rest pose" assumption, real-data transfer improves monotonically from
significantly-worse to statistically-tied-with-majority-win-rate. This is a real, controlled,
mechanistic confirmation — not just a plausible story. **Currently running**: G1d at pose-scale
0.05 (~ meant to bracket further below the 8.85° point that already achieved parity+majority
win-rate, to see if the trend continues into a clear, significant win) — job 3102681, queued as of
this writing (2026-08-21 morning), waiting on cluster priority.

## Where this stands right now

- **Mechanism (D/E/F)**: solid, validated against ground truth, the strongest single result of the
  session — proximal vs. distal error location matters far more than scalar magnitude for this
  optimizer's basin of attraction.
- **Synthetic learned-initializer result (A/B/C)**: solid, validated against this project's own
  established metrics, explained mechanistically by (6).
- **Correspondence-free representation for real scans**: a real defect found and fixed at the
  shared-primitive level (not a one-off patch), validated against ground truth twice
  independently (legs, then body-core), residual limitations honestly characterized, not hidden.
- **Real-scan transfer (bench50)**: went from a clean negative result to a controlled,
  mechanistically-explained recovery to statistical parity via a training-distribution fix,
  currently being pushed further (G1d, in queue). Not yet a confirmed win on real data — parity
  is the current honest status, with one more data point in flight.
- **Not yet done, explicitly deferred on purpose** (to avoid confounding hypotheses): the
  global-orientation question (G2/G4 — does seeding the optimizer's global rotation from the PCA
  estimate help, on top of the leg-pose initializer). `optimise_hierarchical.py` currently has no
  mechanism to accept a global-orientation init at all; this would need a real fitter-side code
  change, deliberately not made yet.
- **Side thread (unrelated to the science)**: a separate, pre-existing large CPU array-job
  campaign (`issue95_fullrun`) was found running under personal Slurm quota; investigated whether
  it could move to the project account (`rwth2151`) — it can't (no CPU partition access on that
  account) — and its auto-chaining dispatcher was inadvertently stopped during that investigation,
  then fully recovered (current chunk running correctly, all completed specimens intact, next
  chunk pre-staged) with one manual step left for the user to run from an actual terminal to
  resume full automation. Documented separately; not part of the main scientific thread.

## Files to read for full detail, in dependency order

1. `out_ceiling_20260820/RESULTS.md` — the inherited FAILURE result this session responded to.
2. `out_ceiling_20260820_25pc/RESULTS_ABC_DEF.md` — topology fix, Arms A/B/C, Arms D/E/F, the
   mechanism.
3. `out_ceiling_20260820_25pc/RESULTS_correspondence_free_representation.md` — the
   `assign_points_to_legs` defect, its fix, body-core, the raw-scan pipeline, the pre-fit sanity
   checks.
4. `out_ceiling_20260820_25pc/RESULTS_bench50_G1_vs_G3.md` — the real-data test, the negative
   result, the dose-response confirmation (G1/G1b/G1c, G1d pending).
