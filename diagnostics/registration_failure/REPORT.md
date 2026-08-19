# Task 6 — why does the optimizer mis-register thin distal-leg geometry?

## Bottom line

**Local minima reachable from the current zero-init are the decisive, load-bearing mechanism**
(D5), and they are made hard to escape by a structural, pose-independent sampling defect that
starves the pretarsus/tarsus of chamfer supervision (D2a). Cross-leg correspondence confusion
(D2b) is the specific pose-triggered symptom of that trapping, not an independent cause. Two
hypothesized mechanisms — pretarsus exploiting its missing joint-rotation limit (D3), and the
mean-based offset penalty diluting thin-structure snapping (D4) — were tested directly and are
**not** well supported by the data. No intervention was implemented; this report ends with
candidates ranked by how directly they address the confirmed bottlenecks.

## Causal hierarchy

| Mechanism | Evidence | Strength | Condition |
|---|---|---|---|
| Local minima / bad initialization | D5 | **decisive** | pose-dependent; GT-init keeps leg_distal joint error within ~1.1-1.3x of leg_prox at every pose level tested, vs. 1.5-2.4x for zero-init |
| Objective pulls a correct pose away from truth | D5 | **absent** (tested, not observed) | GT-init does not drift back down to zero-init's error level — rules out "any init converges to the same wrong answer" |
| Within-leg segment ambiguity (chamfer sample starvation) | D2a, D2b | **strong, structural root cause** | pretarsus specifically (median 2/8000 target samples, 99.5% of resampling windows below the optimizer's own skip threshold); pose-INDEPENDENT, present even at pose_scale=0 |
| Cross-leg correspondence confusion (wrong leg) | D2b | **strong, but a symptom not a root cause** | pose-dependent: 0.00 at pose≤0.10, rising to 0.21-0.32 at pose 0.25-0.35; catastrophic (0.88-0.99) under missing/damaged distal geometry |
| Pose sensitivity (general) | D1, D2b | **strong** | joint-position error and cross-leg confusion increase monotonically with pose magnitude; the derived R metric is noisier/non-monotonic at n=12 and should not be read as the primary signal |
| Pretarsus exploiting its missing joint-rotation limit | D3 | **weak / not supported** | pretarsus rotation stays small relative to its own (practically unconstrained, ±π) range in every condition (median 4-9% of range, P(saturating)≈0) |
| Offset-penalty dilution on thin structures | D4 | **weak / not supported** | distal-leg mean |offset| is only ~1.0-1.2x the corpus-wide mean the penalty actually sees — not dramatically inflated |

## Evidence, in the order it was gathered

### D1 — systematic pose sweep (7 points, pose_scale 0.00-0.35)

Fit the same 12-specimen corpus (only `pose_scale` varied, everything else identical to Task 3's
recipe) at pose_scale ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35}. `leg_distal` R: 0.461, 0.509,
0.516, 0.504, 0.226, 0.144(≈0.141 unsymmetrised), 0.333 — **not monotonic**, with a trough at
0.20-0.25. Raw joint-position error (reused from `appendage_evidence/probe_1a_joint_localization.py`),
by contrast, increases smoothly and monotonically with pose (0.0161 → 0.0255 → 0.0384 at pose
0/0.25/0.35) — so is D2b's cross-leg confusion rate (below). **R's non-monotonicity is a
statistical artifact of n=12 rank correlation being sensitive to the specific scatter pattern at
each pose level, not evidence that the underlying failure mode itself is non-monotonic** — do not
over-read the R curve's exact shape.

### D2a — sample-starvation probe (no fitting needed, direct from target-mesh geometry)

While reading `fitter_3d/trainer_hierarchical.py` for this task, found its own docstring already
documents (on the template) that tarsus+pretarsus hold 2.4% of a leg's surface area, so at the
fitter's `n_sample=8000` the tarsus expects ~12.6 target points and the pretarsus ~0.3 — against
the code's own `n_t<10` group-skip threshold (`_partitioned_chamfer`, line 437/443) — and that
target points are resampled only every `reassign_every=50` iterations, not every step. Measured
this directly (`probe_d2a_sample_starvation.py`, 200 independent area-weighted resamples per
specimen, at the actual n_sample=8000, on every pose-sweep corpus): pooled over all 12
`clean_pose25` specimens × 6 legs, per-SEGMENT median sample counts are co=108, tr=202, fe=122,
ti=29, ta=24, **pt=2**, with **P(pretarsus draws < 10 target points) = 99.5%** and P(exactly zero)
= 17.6%. This is **pose-invariant by construction** (rigid rotation does not change face areas;
confirmed empirically — sample counts barely move across the whole pose sweep for the same
specimen). Since `--split_distal` was never used in any run to date, the whole-leg pool (co
through pt combined, ~490 points) never triggers the group-level skip — but within that pool,
pretarsus's own true-region points are so rare that a pretarsus-owning fitted vertex's nearest
chamfer match is essentially never an actual pretarsus target point.

### D2b — correspondence audit: wrong leg vs. wrong segment, at the fitter's own converged state

Reconstructed the fitter's own `TargetPartition` (imported from `fitter_3d/trainer_hierarchical.py`,
not reimplemented) at each run's final fitted-vertex state, against target points sampled with
KNOWN true leg/segment labels (exact in this ceiling test). Two distinct metrics:

- **Cross-leg confusion** (assigned leg ≠ true leg): **0.00 for every specimen at pose_scale ≤
  0.10**, then rises smoothly — 0.03 (pose 0.15) → 0.09 (0.20) → 0.21-0.24 (0.25) → 0.28-0.32
  (0.35) — and becomes near-total (0.88-0.99) once distal geometry is missing (drop30/drop60,
  where there genuinely is no true target left to match). This is the mechanism that makes pose
  "worse," and it is monotonic even though R is not.
- **Within-leg segment mismatch** (right leg, nearest-matched fitted vertex is the wrong
  segment): already severe **at pose_scale=0** where cross-leg confusion is exactly zero —
  pretarsus mismatch ranges 25-100% per specimen (median cases often exactly 100%: literally
  every pretarsus target point's nearest fitted match belongs to some other segment). This
  directly corroborates D2a: it is not that pretarsus points go to the wrong leg, it is that
  there are effectively none of them to match against in the first place.

### D3 — joint-limit engagement (tested, hypothesis not supported)

Confirmed directly against `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`: pretarsus (`l_*_pt`)
has all 3 rotation axes at exactly ±π (fully unconstrained) while every other leg joint has 2
tightly-bound axes (~20-40°) and one wider-but-bounded bend axis (~70-170°) — the single largest
asymmetry in the rig, and the reason this was flagged as a candidate mechanism. **But measured
fitted rotation usage does not support it as a driver**: pretarsus's median rotation stays at only
4-9% of its own (already generous) ±π range across every condition tested (clean/pose0/drop30/
drop60), and P(rotation > 90% of range) ≈ 0 everywhere. Tarsus shows the most limit engagement of
any segment (median 25-64% of its own range, rising with pose/damage severity) — but this reads as
a downstream symptom of the correspondence problem pulling tarsus around, not a cause in its own
right. **Conclusion: the missing pretarsus constraint is a real rig asymmetry but not, on this
evidence, what is exploited to produce the catastrophic failures.**

### D4 — offset-penalty dilution (tested, hypothesis not strongly supported)

`w_offset` penalizes `deform_verts.pow(2).sum(-1).mean()`, averaged over all ~10235 vertices;
distal-leg vertices are only 8.0% of the mesh, so the a priori concern was that large individual
offsets there would be invisible to the mean. Measured directly: distal-leg mean |offset| is
0.96-1.17x the corpus-wide mean across all 4 conditions checked — **not dramatically elevated**.
The dilution mechanism exists in principle (a small group's offset genuinely contributes less to
a global mean, by arithmetic) but the fitted magnitudes themselves don't show the runaway
"snapping" pattern hypothesized going in.

### D5 — init sensitivity: local minima vs. an inherently-ambiguous objective (decisive)

Added a narrow, diagnostic-only `init_joint_rot` override (`fitter_3d/trainer.py`, default `None`
preserves today's exact zero-init; `--init_joint_rot_from <ground_truth.npz>` wired into
`optimise_hierarchical.py`) and refit `clean_pose25` and `clean_pose35` from the KNOWN true pose
instead of zero, everything else identical:

| condition | leg_distal R, zero-init | leg_distal R, GT-init | leg_distal joint error, zero-init | leg_distal joint error, GT-init | leg_prox joint error (for scale) |
|---|---|---|---|---|---|
| pose25 | 0.144 | **0.604** | 0.0255 | **0.0160** | 0.0149 (zero) / 0.0149 (GT) |
| pose35 | 0.333 | **0.403** | 0.0384 | **0.0175** | 0.0238 (zero) / 0.0158 (GT) |

GT-init does **not** drift back down to the zero-init error level at either pose — it stays within
1.1-1.3x of `leg_prox`'s own error, essentially closing the proximal/distal gap that is the
signature of the catastrophic failure throughout this investigation. This directly answers the
task's central question: **the failure is substantially local-minima trapping reachable from the
current zero-init, not an objective that inevitably pulls even a correct solution toward a wrong
one.** (Some residual gap remains under GT-init, consistent with D2a/D2b's pose-independent
within-leg ambiguity floor — GT-init mitigates the pose-triggered trapping, it does not eliminate
the structural sampling problem.)

## What this rules in and out

- **Ruled in, primary**: bad initialization / local minima (D5), compounded by structural
  chamfer-sample starvation on tarsus/pretarsus (D2a) that likely makes those minima both easier
  to fall into and harder to climb out of (there is little true-region signal to climb toward).
  Cross-leg confusion (D2b) is real and pose-monotonic but reads as the visible symptom of this
  trapping, not a separate root cause.
- **Ruled out / not load-bearing on this evidence**: pretarsus's missing rotation limit (D3),
  offset-penalty dilution (D4). Both are real structural facts about the rig/loss but neither
  shows up as a driver when measured directly.

## Candidate interventions, ranked by directness (not implemented — for a follow-up task)

1. **`--split_distal`** (already implemented in `fitter_3d/trainer_hierarchical.py`, never used in
   any run to date) — gives tibia/tarsus/pretarsus their own equally-weighted chamfer term
   instead of pooling with the well-sampled proximal segments, directly attacking D2a's confirmed
   root cause. Zero new code; a one-flag experiment.
2. **A pose initialization strategy for leg joints** less naive than exact zero — e.g. extending
   the existing `--pf_init` centroid-matching mechanism (already built for a related purpose,
   `optimise_hierarchical.py:84-127`) to leg `joint_rot`, or a small multi-restart-and-select
   scheme (since real specimens have no ground truth to init from) — directly targets D5's
   confirmed mechanism.
3. **Increase `n_sample` and/or `reassign_every` frequency for leg groups specifically** —
   attacks D2a's starvation from the sampling-budget side rather than the partition-structure
   side; complementary to #1, not a substitute.
4. Lower priority given D3/D4's results: tightening the pretarsus joint limit, or reweighting
   `w_offset` by group instead of a global mean — plausible but not indicated as high-value by
   this evidence; would need their own validation if pursued.

## Reproducing

```
conda activate pytorch3d
# D1: pose sweep corpora + fits (SLURM job 3043380) already generated/fit; rescore any level:
python diagnostics/absolute_scale/calib_features.py --runs SYN_clean_pose20_w5 --primary SYN_clean_pose20_w5 --corpus synth_clean_pose20

# D2a: no fitting needed
python diagnostics/registration_failure/probe_d2a_sample_starvation.py

# D2b: correspondence audit on any already-fit run
python diagnostics/registration_failure/probe_d2b_correspondence_audit.py

# D3/D4: joint-limit engagement + offset dilution, on already-fit runs
python diagnostics/registration_failure/probe_d3_d4_limits_offset.py

# D5: GT-init fits (SLURM job 3043561) already run; rescore:
python diagnostics/absolute_scale/calib_features.py --runs SYN_clean_pose25_gtinit_w5 --primary SYN_clean_pose25_gtinit_w5 --corpus synth_clean
PROBE_RUN=SYN_clean_pose25_gtinit_w5 PROBE_CORPUS=synth_clean python diagnostics/appendage_evidence/probe_1a_joint_localization.py
```

Artifacts: `diagnostics/registration_failure/out/{d2a_sample_starvation,d2b_correspondence_audit,d3_d4}.json`,
fit logs/outputs under `diagnostics/registration_failure/{logs,runs}/` and
`diagnostics/moonshot/runs/SYN_clean_{pose05,pose10,pose15,pose20,pose35,pose25_gtinit,pose35_gtinit}_w5/`.

## Code changes made in this task

`fitter_3d/trainer.py` (`SMAL3DFitter.__init__`) and `fitter_3d/optimise_hierarchical.py`: added
an optional `init_joint_rot` override / `--init_joint_rot_from` flag, diagnostic-only, default
`None` preserves today's exact zero-init behavior (verified: `torch.all(joint_rot==0)` still holds
with no override). No other production code was modified; no default behavior changed.
