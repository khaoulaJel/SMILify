# Everything Tried Today (2026-08-25): Ideas, Findings, What They Mean

One document covering the full day's investigation into leg-pose initialization and, later,
correspondence-based fitting for SMILify. Organized as a sequence of attempts, each with what was
tried, what was found (with numbers), and what it means for the project. Detailed
methodology/derivations live in the referenced files; this is the synthesis.

---

## 1. Basin map: why some pose errors are far more damaging than others

**Idea.** Perturb ground-truth leg pose in structured ways (random / proximal / distal / coherent
/ swap / mirror) at controlled magnitudes, fit each via the standard D1 recipe, and measure which
error *structure* — not just magnitude — the optimizer can recover from.

**Finding.** At matched ~30° magnitude, `coherent` (a single rigid rotation at the coxa only,
downstream joints exactly at rest) reaches leg_acc **0.839**, vs `proximal` (independent noise
concentrated at co/tr/fe) at only **0.622** — a 0.217 gap from structure alone, not magnitude.
`distal` stays flat/high (0.937-0.940) regardless of magnitude; `swap` (a fully valid donor-
specimen pose) is nearly as damaging as bad proximal noise (0.834), evidence of a basin-*selection*
problem, not an anatomical-implausibility one. Full detail:
`out_basin_map_20260825/RESULTS_basin_map_20260825.md`.

**What it means.** Error *coherence* — whether a bend is a single clean rotation or scattered
independent noise — matters more than how big the error is. This became the design premise for
section 2 below, and it's also the finding G1d's own dose-response curve was later shown to be
explained by (same proximal-skewed/incoherent signature at every training scale).

**Adjacent probes run the same day, same investigation:**
- IK oracle (given the TRUE tip position, the best input this method could ever get): mean
  per-joint error only closes to **~21°**, a genuinely redundant problem (5 joints × 3 axes = 15
  unknowns, 3 position constraints), confirmed via a disambiguation test (GT-seeded vs zero-seeded
  vs weak-regularizer runs all converge to ~21-23°, ruling out "regularizer artifact").
- Constraint-count sweep (k=1..5 true positions along the chain): 21.15° → 19.34° → 17.48° →
  16.05° → 14.23° — closes gradually, never vanishes. Some of the redundancy is fundamental
  (twist ambiguity: a position tells you which way a bone points, never its rotation about its
  own axis).
- `swap`'s failure was tested against donor-target pose distance: flat correlation at two levels
  of decomposition — swap's damage is not predictable from how far the donor pose is.
- A pose-mirroring transform (`Ref = diag(1,-1,1)`, conjugation `R_mirrored = Ref @ R @ Ref`) was
  built and verified via forward-kinematics round-trip before use.

---

## 2. Multi-start coherent candidates (PCA / cluster-axis / tip-direction)

**Idea.** Since `coherent` beat `proximal` this convincingly, build several *independently
estimated* coherent (coxa-only) candidates from real scan geometry — not perturbed ground truth —
and let a GT-free selector (`fscore@0.02`) pick the best one per specimen. Rationale: PCA-coherent
alone (built earlier the same day) lost to zero-init in aggregate (0.804 vs 0.867-0.872) despite
being coherent by construction — the hypothesis was that this was an *estimation* problem, fixable
by diversifying the estimator, not a mechanism problem.

**Built** (`fitter_3d/geom_leg_init.py`): `estimate_leg_cluster_axis_direction` (near/far graph-
distance-split centroid direction) and `estimate_leg_tip_direction` (reuses `estimate_leg_tip`'s
furthest-point estimate as a pure direction), sharing a new `init_joint_rot_for_specimen_coherent`
skeleton with the existing PCA-coherent estimator.

**Oracle validation first** (`coherent_multi_estimator_oracle_PROBE.py`), on TRUE (clean, noise-
free) points: all three estimators agree with the TRUE overall (coxa→tip) direction to within
3.95° (PCA) / 5.37° (cluster) / ~0° (tipdir, tautologically, on clean data) — the estimator math is
sound. But their implied *coxa rotation* differs from the TRUE coxa rotation by ~50-55° on
average. A disentangling check explains this precisely: the TRUE coxa rotation's own implied
direction differs from the TRUE overall direction by a mean of **50.03°** — i.e., real leg poses
distribute roughly half of their bend across tr/fe/ti/ta as well as the coxa. **No coxa-only
estimator, however accurate, can recover the true per-joint decomposition** — this is the
structural fact the rest of this thread is built on.

**Real-scan generation and D1 results** (`generate_multistart_coherent_init.py`, run scripts
`run_multistart_coherent_20260825.sh`): raw pose error 33.34° (cluster) / 31.30° (tipdir),
comparable to PCA-coherent's 32.89°. D1 aggregate leg_acc: **Cluster_coherent 0.826-0.831,
Tipdir_coherent 0.832-0.835** — both below zero-init (0.867-0.872), the *same* qualitative loss
PCA-coherent already showed (0.804-0.808). Three independently-built estimators, three losses to
zero-init, for the same diagnosed reason.

**Cross-estimator spread as a confidence signal (Option C) — tested and closed.** Hypothesis: legs
where the 3 estimators disagree a lot are the ones a selector should distrust. On real scan data,
spread is well above the clean-data floor (median 10-16°, but heavy-tailed to 122-148°) — a real,
measured signal — but correlation with actual per-leg outcome was weak throughout: 2-estimator
spread r=-0.12 to -0.16 (never significant); with all 3 estimators, r=-0.22 to -0.24 (borderline,
Spearman not significant). Per an explicit user instruction, this was closed as a non-actionable
signal rather than built on further.

**5-candidate ceiling/selector test** (`analyze_multistart_selection_20260825.py`): pooling zero /
IK / PCA / cluster / tipdir moved the oracle ceiling only **+0.009** (0.906→0.915) — far smaller
than the earlier 3rd-candidate jump — while the GT-free selector's own correctness *dropped* from
9/12 (75%) to 7/12 (58%), with mean selected leg_acc staying flat at 0.901 by coincidence, not
because the selector held up. Margin still doesn't predict correctness (r=-0.037, p=0.91).

**What it means.** Adding more geometrically-diverse coxa-only candidates does not scale: it barely
raises the achievable ceiling and measurably degrades the practical (GT-free) selector's
reliability. This is the point at which "try more coherent estimators" was correctly abandoned as
a direction, in favor of asking whether a *data-driven* pose prior could do what hand-built
geometric estimators couldn't (section 3).

---

## 3. Learned pose-subspace IK — tried, and definitively closed by a data-availability check

**Idea.** Since real poses couple joints in *some* structured way (even if coxa-only estimators
can't capture it), fit a low-dimensional PCA subspace over real (ground-truth) whole-chain
[co,tr,fe,ti,ta] rotations, leave-one-out across the 12 `synth_clean` specimens, and do IK *within*
that subspace instead of raw 15-D box-constrained IK.

**Oracle result** (`pose_subspace_ik_oracle_PROBE.py`): every mode count tested (2, 3, 5, 8, 15)
performed at or worse than raw IK's own oracle (21.15°/19.34° tip-only/tip+waypoint). Explained-
variance ratios were nearly flat (0.14, 0.12, 0.10, 0.09, 0.08, ...) instead of concentrated in a
few leading modes — with only 66 pooled training rows (11 held-out specimens × 6 legs), there is no
detectable low-dimensional linear structure to exploit.

**Checked for a bigger real corpus before concluding anything** (this was the key, decisive step):
`joint_limits.py`'s per-axis bounds are confirmed **hand-authored in Blender by an artist**
(`docs/joint_limits_user_guide.md`), not statistically calibrated from any digitized-pose corpus.
The only other real-specimen pipeline in the repo, AntScan (`custom_processing/antscan_*`), holds
individual pinned museum scans with **no per-joint rotation ground truth at all** — extracting it
would require solving the correspondence problem this whole investigation exists to fix, so it
doesn't qualify regardless of specimen count. **The 12 `synth_clean` specimens are the only source
of ground-truth joint rotations anywhere in this project.**

**What it means.** This isn't a "needs more tuning" result — it's "this corpus cannot support a
learned linear pose-subspace prior, at any mode count, full stop," because there is no larger real-
or-synthetic-with-natural-coupling corpus to fit one from. Re-slicing the same 12 specimens by leg
position would make the data problem worse (11 samples/fold), not better. This closed the entire
"learn a better pose prior from available data" family of ideas, alongside section 2's closure of
"hand-build a better geometric estimator" — both for the same underlying reason (see section 4's
reframe).

---

## 4. The reframe: predict correspondence, not pose (Phase 10 design)

**Idea.** Every closed door above was closed by trying to recover *pose* (joint angles) from
geometry that structurally underdetermines it (section 2's disentangling finding) with too little
data to learn a substitute (section 3). The literature sweep's own SOTA pattern (GeoTransformer /
D3Feat / REGTR / TANet) suggests a better-posed target: **for each scan point, which template
vertex does it correspond to?** Pose becomes something derived from correspondence, not regressed.

**Full design, grounded in verified code, not just described:**
`PHASE10_DESIGN_correspondence_network_20260825.md`. Five decisions, each tied to a specific,
confirmed fact in the codebase:
1. **Fix the training-data generator.** `sample_smil_model.py`'s `generate_random_parameters`
   samples every joint's axis-angle i.i.d. (`pose_scale * randn(...)`, no coupling) — confirmed at
   the line level, and confirmed that `make_synth_corpus.py` builds `synth_clean` itself with this
   exact function. Every leg_acc number in this whole investigation, including the disentangling
   probe's own ground truth, comes from this decorrelated sampler.
2. **Supervise on dense per-point correspondence**, free from synthetic generation, not chamfer
   (which "cannot see identity" — two anatomically unrelated points that happen to be
   Euclidean-close score as correct).
3. **Fix the architecture's collapse-before-decode.** Confirmed at the line level in
   `smil_pointnet.py`: `sa3` is `group_all=True`, producing one pooled 1024-d vector that every
   output parameter (pose, shape, trans) is regressed from — discarding `sa1`/`sa2`'s still-
   localized per-region features before any parameter is decoded.
4. **A new fitter hook, not just an init.** Confirmed `--init_joint_rot_from` only seeds the
   starting value (`optimise_hierarchical.py`); the real per-iteration mechanism is
   `TargetPartition`'s `knn_points`-based nearest-fitted-vertex assignment
   (`trainer_hierarchical.py`). This is the one piece of genuinely new engineering, scoped as an
   additive term first (safer) before considering a full replacement of `TargetPartition` itself.
5. **Name the sim-to-real risk plainly** (does synthetic-trained correspondence transfer to real,
   noisy bench50 scans) rather than assume it away; propose D1's own converged bench50 output as
   pseudo-labels for a fine-tuning pass if the gap proves large.

**What it means.** A concrete, evidence-grounded plan exists, but per its own explicit sequencing,
nothing in it should be built (sampler, architecture, training pipeline) until the cheapest,
highest-leverage question is answered first: is the correspondence lever worth anything at all,
tested with perfect information, before anything estimates it. That's section 5.

---

## 5. The decisive test: correspondence-oracle results

**Step 1, prioritized deliberately over the sampler** (explicit sequencing decision: the sampler
only calibrates data realism; this test determines whether the entire Phase 10 premise has
headroom at all). Full numbers, both falsifiable checks, and the metric-ceiling/weight-dominance
resolution: `RESULTS_correspondence_oracle_20260825.md`.

### 5a. Group-level oracle (`Oracle_GT_partition`)
New `--oracle_gt_partition_from` flag in `optimise_hierarchical.py`, reusing `PartFieldPartition`
**unmodified** (the same class `--part_field` already uses for a *learned* partition), fed ground-
truth leg labels instead. leg_acc **0.937** vs zero-init's 0.867.

- **Self-correction applied:** an initial read ("targets hard cases specifically") did not survive
  a sign test (p=0.387) or Wilcoxon (p=0.076-0.117) on the full 12 specimens; excluding the two big
  movers (synth_003, synth_005), the remaining 10 are genuinely flat (mean delta +0.0075, split
  5/5) — retracted rather than kept.
- **Traced to `FINAL_REPORT.md`'s own headline number:** `why_partitions_null.py`'s `--gt_groups 7`
  uses the *same* grouping this project's `leg_acc` uses, and established that 83.3% of total
  correspondence error is **within-part** — a slice no partition, however fine, can ever reach.
  `Oracle_GT_partition` only ever had access to the other 16.7%. Reframed honestly as a **floor**
  under the effect, not a ceiling.

### 5b. Dense per-vertex oracle (`Dense_GT_oracle`) — the one that reaches the 83.3% slice
New additive `w_dense_gt` loss term in `trainer_hierarchical.py`'s `forward()` (default 0.0, gated
identically to every other optional term — `w_edge`, `w_midline` — so byte-identical when off,
same discipline as `--init_joint_rot_from`), matching each resampled target point directly to the
fitted mesh's own vertex at its TRUE index.

**Pre-registered before looking at the result:** metrics (leg_acc *and* seg_acc), all three tests
(t/sign/Wilcoxon) reported regardless of which looked best, explicit outlier check, and a concrete
bar for "real" vs "still capped" stated in advance.

**Result: seg_acc +0.057, 12/12 specimens positive** (paired t p=0.011, sign test p=0.0002,
Wilcoxon p=0.0002) — the first broad, non-outlier-driven effect in the entire day's investigation.
leg_acc (0.936) is statistically indistinguishable from the group oracle's own 0.937, exactly as
expected (leg_acc can't see within-part correspondence at all).

**Two more falsifiable predictions, both resolved before the causal story was written up:**
- **Check A:** synth_003/005's dense-oracle gain is 0.99× their group-oracle gain — essentially
  identical. So the "these two specimens are special" pattern is about *any* correspondence fix,
  not dense specifically. The real within-part story is elsewhere: synth_002 (never previously
  flagged) shows the second-largest seg_acc gain (+0.126).
- **Check B:** l1/l2/l3 seg_acc deltas are nearly identical (+0.060/+0.062/+0.068) — inconclusive
  on whether the documented mesothoracic-starvation defect (`geom_leg_init.py`) limits correspondence
  recovery specifically at l2. Reported as a null result, not rounded toward either story.

**The residual gap, resolved rather than assumed:** `Dense_GT_oracle`'s seg_acc (0.874) is short of
1.0 — but so is ground truth itself. Feeding the ground-truth mesh through the identical evaluation
(`GT_as_fitted_ceiling`) scores only **0.974** seg_acc / 0.991 leg_acc, from inherent near-boundary
point-sampling ambiguity in the metric. So the real gap is ~0.10, not ~0.17. Two explanations for
that ~0.10 were distinguished, not assumed: a weight-dominance run
(`Dense_GT_oracle_dominant`, `w_dense_gt_correspondence=50.0`, confirmed to actually dominate the
objective via a sharp rise in edge distortion, 0.05→0.36-0.80) left seg_acc **unchanged** (0.870 vs
0.874, p=0.245, only 5/12 specimens even improved). **The weight-balance explanation is falsified.**
The gap is a genuine fitter-capacity/geometric limit — shape-space capacity, mesh
topology/resolution, or the model's own deformation limits — independent of how strongly the
correspondence signal is weighted.

**What it means.** Correspondence is validated as a real, broad, addressable lever — the strongest,
least outlier-dependent finding of the entire day. But there are now two *independent*, correctly-
separated risks for the eventual network, not one: (i) the already-known gap between a perfect
oracle and a *learned*, imperfect predictor (sim-to-real / oracle-vs-realistic, section 4 point 5),
and (ii) a newly-earned fitter-capacity ceiling (~0.874, not 1.0) that exists even under perfect
information, unrelated to how good the correspondence estimate is. Both are now written into
`PHASE10_DESIGN_correspondence_network_20260825.md` (§5, §5b) rather than conflated into one.

---

## 6. Where this leaves the project

**Closed, with evidence, not by default:**
- Hand-built geometric coxa-only candidates (PCA / cluster-axis / tip-direction), individually or
  pooled — capped by a shared structural limit (section 1-2), diminishing returns on pooling.
- A learned linear pose-subspace prior — no corpus in this project can support one (section 3).
- "Cross-estimator disagreement predicts which legs are wrong" — tested twice, never significant.

**Open and validated, with a precisely scoped remaining risk:**
- Correspondence-as-target (Phase 10) is real: a broad, 12/12-specimen effect exists and survives
  every falsification attempt thrown at it today. What's NOT yet known is how much of the ~0.10
  fitter-capacity gap (section 5b) can be closed, and by what (shape-space capacity vs. mesh
  resolution vs. regularization structure) — that diagnosis is the natural next cheap probe, ahead
  of committing to the sampler/architecture/training-pipeline build.

**All code changes made today are additive and off-by-default:** `--oracle_gt_partition_from`,
`--dense_gt_correspondence_from`, `--w_dense_gt_correspondence` in `optimise_hierarchical.py`;
`dense_gt_verts`/`w_dense_gt` in `trainer_hierarchical.py`; `per_specimen_per_leg_acc`/
`per_specimen_seg_acc` in `run_audit.py`. Nothing here changes any existing arm's behavior.
