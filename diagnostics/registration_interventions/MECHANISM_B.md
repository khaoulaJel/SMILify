# Intervention B — geometry-only leg pose initialisation: design record

## Why not `--pf_init`

Investigated first (see `REPORT.md` §4 and the chat record): `fitter_3d/part_anchor_init.py`
(the code `--pf_init` needs) does not exist on `to-ship` — only on `feature/registration_moonshot`,
read via `git show`, never merged. On that branch, after a serious bias in it was fixed
(barycentric-area-weighted moments — the unweighted version literally drove chamfer 105x worse,
rotating mandibles 52-94°, chasing a sampling artefact, not a pose), the *fixed* version bought
`edge_logratio +5.1%` and **"nothing else"** (`moonshot/REPORT.md` §5.3) — and even that result was
confounded with a separately, decisively bad frozen learned-part-field partition used for the rest
of the fit (chamfer −47.6%, `part_leg_distal within_tau` −18.5%), so it never isolated the
init-alone effect. Independently, `PartAnchorInit`'s objective (match per-part centroid + second
moment) is close to blind to *articulation*: a bent and a straight leg of the same length/radius
have similar moments, so the objective has little gradient telling it *where* a bend happened —
exactly the information Task 6 D5 needed (the actual GT pose) to recover `leg_distal` reliability.

## Design requirement (user-specified, verbatim intent preserved)

Use only target geometry + known anatomical template/anchor information. No GT vertex/joint
labels, no GT rotations, no fitted correspondences, no learned part fields, no manually selected
specimen-specific correspondences. Validate independently against GT (evaluation only) before any
fitting run. Do not tune hyperparameters against the four real test conditions.

## What the initialiser does (`fitter_3d/geom_leg_init.py`)

1. Reuses H0's fitted `global_rot`/`trans` (shared, identical, precondition for BOTH arms —
   the only thing baseline and geometric-init have in common besides everything else).
2. Computes 6 coxa anchor POSITIONS purely algebraically from H0's rigid params + the template's
   known rest-pose kinematic tree (`Jr @ v_template`) — no target point cloud involved yet.
3. Assigns each target point to its nearest of the 6 ANALYTIC anchors (not `TargetPartition`,
   not any mesh-based nearest-fitted-vertex correspondence — that mechanism is what Intervention
   A already showed is unreliable for distal segments, so it is deliberately excluded here to
   avoid bootstrapping the hard part of the problem from an estimate known not to solve it).
4. Within a leg, bins its points by GRAPH (geodesic-like, k-NN + Dijkstra) distance from the coxa
   anchor against the template's cumulative rest bone length, takes each band's centroid as an
   estimated joint position.
5. Converts the estimated curve into `joint_rot` via closed-form forward-kinematics inversion
   (chain "aim" IK) — a single algebraic pass, not an optimisation loop.
Falls back to zero local rotation (= today's default) per-segment wherever a band has too few
points. Full audit of what is and is not used: `geom_leg_init.py`'s own "LEAKAGE AUDIT" docstring
section.

## Two bugs found and fixed during PRE-REGISTERED validation (before touching real target data)

The validation discipline (build a synthetic rest-pose self-consistency check — a target
literally built from the model's own rest pose, zero articulation, must recover ~zero rotation —
BEFORE running on any real corpus or GT) caught two real design errors, neither visible from
code review alone:

1. **FK root-pivot bug.** First version computed a coxa's world position as
   `trans + R(global_rot) @ (rest_coxa - rest_root)`, dropping the fact that
   `batch_global_rigid_transformation`'s root transform pivots the WHOLE skeleton about the
   root's own rest position (not the coordinate origin) before translating — i.e. the correct
   formula is `trans + rest_root + R(global_rot) @ (rest_coxa - rest_root)`. Confirmed by direct
   comparison against `smal_model.batch_lbs`'s own code, not guessed. The rest-pose self-test
   caught this immediately (max rotation 1.3-2.3 rad on a target that should recover ~0).
2. **Distance-metric bug.** First (and second) versions used Euclidean distance from a fixed or
   walking anchor as a proxy for "distance along the chain." Both failed the rest-pose
   self-test — ant leg segments are not collinear even at rest (coxa/trochanter/femur already
   bend in the template), so straight-line distance from one point conflates points from
   different segments even with zero articulation. Fixed by replacing it with GRAPH distance
   (k-NN graph + Dijkstra shortest path), a standard Isomap-style geodesic estimate that follows
   the point cloud's own local connectivity instead of straight-line distance.

After both fixes, the rest-pose self-test gives median |rotation| ≈ 0.08 rad across all 30 leg
joints (co/tr/fe/ti/ta x 6 legs), most near-exactly zero, with residual error concentrated on a
few femur joints (legs 1 and 3) even at rest — an honestly-reported, NOT tuned-away, remaining
limitation of a purely geometric heuristic. No further architecture changes were made after this
point specifically to chase that residual to zero, since doing so on a synthetic self-test that
is itself a design proxy (not one of the four real evaluation conditions) risks exactly the
outcome-chasing the validation discipline exists to prevent. The REAL evidence is the
pre-registered `probe_B0_geom_init_validation.py` run against actual target point clouds and
ground truth on all four conditions (§ below), not this synthetic self-test, which only gates
whether the mechanism is implemented correctly enough to be worth evaluating for real.

## Fixed hyperparameters (declared before any of the 4 conditions were examined)

- `min_pts_per_band = 4` (redundancy floor for a 3D centroid, same convention as the unrelated
  moonshot `part_moments` code).
- band half-width = 0.5x that band's own rest segment length (structural: guarantees adjacent
  bands tile the rest chain with no gap at zero padding, gives one full segment-length of slack
  for a genuine bend).
- `KNN_K = 6` for the geodesic graph (standard Isomap/geodesic-embedding default).
None of these were adjusted after seeing the pose0/pose25/drop30/drop60 results.

## Validation results (pre-registered, evaluation-only use of ground truth)

`diagnostics/registration_interventions/probe_B0_geom_init_validation.py`, SLURM job 3053017,
COMPLETED. **REJECTED**: worse than zero-init at every leg joint in every condition (leg_distal
2-16x worse depending on condition, worst at pose0 where zero-init is already near-correct).
Full numbers, leading hypothesis for the cause (geodesic short-circuiting on area-sampled surface
width, not verified), and the verdict are in `REPORT.md` §4. Not wired into any fitting run.
