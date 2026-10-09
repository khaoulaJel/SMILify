# Raw-Scan Generalization Test: G1 (learned) vs G3 (zero) on bench50 — RESULTS (2026-08-20/21)

**UPDATE (2026-08-21): hypothesis 1 below (domain-gap in training pose-noise magnitude) was
tested overnight and confirmed — see "Follow-up: G1b, matching training pose-noise to a
near-rest-pose assumption" at the end of this file. It closes the gap to statistical parity with
zero-init. Read that section alongside this one; the negative result below is real but not the
final word.

Follow-up to `RESULTS_correspondence_free_representation.md`. That file establishes a validated,
literature-grounded correspondence-free anatomical representation (body-core + 6-leg chain
assignment, fixing a real defect found in the shared `assign_points_to_legs` primitive) and shows
the existing learned initializer produces stable, structured, plausible-magnitude predictions on
50 raw, unseen, taxonomically diverse real ant scans (`bench50_clean`). This file reports the
actual fitting result — the test that matters.

## Decision (stated first)

**On real, unseen morphology, the learned initializer (G1) does NOT outperform zero-init (G3).**
It is measurably worse on both surface-fit metrics used elsewhere in this project, and wins on a
minority of specimens (20/50, 40%). This is the opposite of the synthetic result
(`RESULTS_ABC_DEF.md`: learned beat zero on every metric, `synth_clean`, n=12). The synthetic
proof-of-mechanism (D/E/F: proximal vs. distal error location has anisotropic effect on the
optimizer's basin) does not, on this evidence, transfer as a practical initialization benefit to
real scans as currently implemented. This is a genuine, reportable negative result, not a
methodology failure to be explained away — see "What this does and doesn't mean" below.

## Setup

Identical D1 recipe to every other arm in this experiment series (`--midline 2.0 --beta_prior 0.0
--limit 0.273 --offset 30.0`, `D1_low.yaml`), run on all 50 `bench50_clean` specimens, both arms
at identity global orientation (the fitter's actual default regardless — this specific comparison
needed no code changes, per the explicit isolation decision to defer the separate
global-orientation hypothesis to its own G2/G4 experiment). G1's init came from
`generate_bench50_learned_init.py`: the validated correspondence-free representation (actual PCA
global rotation, 7-way body-core+leg chain assignment, `body_core_canonicalize`, existing trained
checkpoint) applied with zero fitting-time knowledge of the target beyond what a real deployment
would have. No ground-truth pose exists for real scans, so scoring is Chamfer/F-score only (no
`leg_acc`/`all_feature_R`, which require known correspondence).

## Result

| Arm | Chamfer L2 (mean) | Chamfer L2 (median) | F@0.01 | F@0.02 | Wins (of 50) |
|---|---:|---:|---:|---:|---:|
| G3 — zero/default | 0.000637 | 0.000546 | **0.4983** | 0.8146 | 30 |
| G1 — learned (raw-scan) | 0.000697 | 0.000595 | 0.4762 | 0.8016 | 20 |

Worst-case specimens for G1 (F-score deltas of −0.10 to −0.20) span multiple genera
(`Labidus`, `Temnothorax`, `Centromyrmex`, `Strumigenys`) with no obvious common cause found —
checked PCA global-rotation magnitude as a candidate explanation (large estimated rotation ⇒
possible misalignment) and it does **not** correlate: worst-for-G1 specimens span 6–87° estimated
rotation, best-for-G1 specimens span 20–87°, essentially the same range. That specific hypothesis
is ruled out, not just unconfirmed.

## What this does and doesn't mean

**Does NOT mean**: the D/E/F mechanism (proximal-vs-distal error location matters more than
scalar magnitude) is wrong, or that the geometric-representation fixes in
`RESULTS_correspondence_free_representation.md` were pointless. Those are independently validated
against known ground truth (`synth_clean`) and stand on their own evidence.

**Most likely explanation** (stated as the best-supported hypothesis given current evidence, not
a confirmed fact — no further diagnostic has isolated it yet):

1. **Domain gap between training and deployment distributions.** The network was trained
   exclusively on synthetic corpora with injected random pose noise (`0.5*N(0,1)` per joint) on
   top of the template's own shape space (25 PCs). Real `bench50` specimens are photographed/
   scanned museum specimens, typically pinned or preserved in a fairly natural, legs-down resting
   posture — i.e., plausibly much CLOSER to the template's rest pose than the large synthetic
   perturbations the network was trained to correct. That would make zero-init a much stronger
   baseline specifically on real data (where it's closer to correct a priori) than on the
   synthetic held-out set (where its 23° mean error was, by construction, no different from the
   network's own).
2. **Residual representation-quality gap on real (vs. synthetic) geometry.** The validated
   7-way chain assignment reaches macro-F1 0.697-0.831 on clean, noise-free `synth_clean` — real
   scans are noisier, more varied in absolute proportion, and were never used to tune the
   assignment rule (deliberately, per the explicit instruction not to develop the rule against
   bench50). Labeling errors on real scans plausibly happen at a different rate than on
   `synth_clean`, and the earlier "stable, structured, plausible-magnitude" diagnostic
   (`RESULTS_correspondence_free_representation.md`) checked *self-consistency*, not
   *correctness* — a systematically mislabeled but internally consistent input would look
   identical on that diagnostic while still misleading the network.

Both are plausible and not mutually exclusive; this run's data cannot distinguish them further
without either (a) ground-truth pose for at least a few real specimens (not available), or (b) an
ablation training the network on synthetic data with a pose-noise distribution matched to real
specimens' actual (much smaller, unknown) deviation from rest pose.

## Implication for the research direction

This is a genuinely useful negative result, not a dead end. It sharpens the actual research
question (echoing the reformulation already reached in `RESULTS_ABC_DEF.md`): the mechanism
(proximal-vs-distal error location) is validated and real; the specific trained artifact (a
PointNet trained on synthetic large-pose-noise data, fed a still-imperfect correspondence-free
input) does not yet transfer its synthetic advantage to real specimens. The credible next steps,
in order of likely diagnostic value:

1. **Match the training pose-noise distribution to real specimens' actual variation** — if
   real specimens are near rest pose, training the initializer to predict small, realistic
   corrections (rather than recovering from large synthetic perturbations) may transfer better.
   This is a testable, scoped hypothesis, not a guess: it follows directly from hypothesis 1
   above.
2. **Get a small amount of real ground truth** (even 3-5 manually/semi-manually posed real
   specimens) to directly measure whether hypothesis 1 or 2 (or both) explains the gap, rather
   than continuing to infer from Chamfer/F-score alone.
3. Only after (1) or (2) narrows the explanation: revisit the geometric representation itself
   (soft labels, retraining) — per the earlier decision tree, don't add representation complexity
   before knowing whether representation quality or training-distribution mismatch is the
   bottleneck.

**Do not read this result as "the learned initializer approach failed."** It means the specific
trained artifact, evaluated end-to-end on real data for the first time tonight, has an
unresolved domain-gap problem between its synthetic training distribution and real deployment
conditions — a normal, expected finding at this stage of a sim-to-real pipeline, not a
falsification of the underlying mechanism.

## Follow-up: G1b, matching training pose-noise to a near-rest-pose assumption (2026-08-21)

Tested hypothesis 1 directly rather than leaving it as a proposal. Regenerated the 5000-specimen
training corpus with `--pose-scale 0.15` instead of `0.50` (verified empirically on a 20-specimen
test batch first: mean per-joint magnitude ≈13.8°, vs. the original ≈23° — chosen because it's
close to what the ORIGINAL checkpoint actually predicted on real bench50 scans, 11.5° mean,
suggesting the network's own real-data behavior was already hinting at a smaller natural
deviation than its synthetic training target). Retrained from scratch (same architecture, 50
epochs, same held-out `synth_clean` corpus for monitoring — best val 13.133° at epoch 49, no
overfitting observed in the training window). Regenerated `bench50` inits with the new checkpoint
(`bench50_learned_init_lowpose.npz`, same pipeline, 0/50 degenerate) and re-ran the identical G1
fit recipe as "G1b."

| Arm | Chamfer L2 (mean) | F@0.01 | F@0.02 | Wins vs G3 (of 50) |
|---|---:|---:|---:|---:|
| G3 — zero/default | 0.000637 | 0.4983 | 0.8146 | — |
| G1 — learned, trained on 23°-noise corpus | 0.000697 | 0.4762 | 0.8016 | 20/50 |
| **G1b — learned, trained on 13.8°-noise corpus** | **0.000649** | **0.4972** | **0.8169** | 19/50 |

G1b is essentially tied with G3 on every aggregate metric (chamfer within 2%, F-score within
0.2%, F@0.02 actually marginally *above* G3). Paired t-test on per-specimen F-score@0.01, G1b vs
G3: **t=-0.19, p=0.85** — no statistically significant difference. Win-rate stayed similar
(19/50 vs G1's 20/50) — G1b is not winning outright more often, but the *magnitude* of its
remaining losses shrank dramatically (mean signed delta -0.001 vs G1's -0.022), which is why the
aggregate gap closed while the win-count didn't move much.

**Interpretation**: this confirms domain-gap-in-training-pose-magnitude was a real, substantial
contributor to G1's underperformance — matching the synthetic training distribution's pose-noise
scale to a more realistic (near-rest-pose) deployment assumption took the learned initializer
from "measurably worse than zero-init" to "statistically indistinguishable from zero-init" on
real, unseen morphology, with no other change (same representation, same architecture, same
training recipe otherwise). It does **not** yet confirm the learned initializer definitively
*beats* zero-init on real data — parity, not victory, is the honest read of this evidence. The
residual gap to a clear win is most plausibly hypothesis 2 (representation-quality gap on real
vs. synthetic geometry, not retested here) and/or that 13.8° may still not be the correctly-tuned
pose-noise scale — a small pose-scale sweep (e.g. 0.05, 0.10, 0.20) would be the natural next
step to find whether an even better match exists, not attempted tonight given time constraints.

## Follow-up: G1c, G1d, and the WSL continuation (2026-08-24/25)

**G1c** (pose_scale 0.10, retrained/refit on the cluster before the 2026-08-21 compute outage):
chamfer 0.000637 (tied with G3's 0.000637), F@0.01 0.4974, 26/50 wins, paired p=0.855 — parity,
matching G1b, continuing the plateau rather than a new data point of improvement.

**Investigation moved to a WSL machine 2026-08-24** after a cluster-wide compute outage (kernel
CVE lockdown) — see `HANDOFF_20260824.md` for the full transition context. Two things had to be
recovered/fixed before G1d could even run: (1) `optimise_hierarchical.py --init_joint_rot_from`,
documented by every generator script in this directory but never actually implemented in the
committed fitter code (the modification existed only on the cluster's own diverged local branch
and was lost when only new files, not modifications to existing ones, were carried over to
`origin/feature/investigation`) — implemented fresh, verified against `config.N_POSE` and the npz
convention every generator script already used; (2) confirmed `diagnostics/moonshot/runs/` (git-
ignored) was genuinely unreachable from WSL, so G1d was re-run from scratch rather than assumed
lost.

**G1d** (pose_scale 0.05): chamfer 0.000621, F@0.01 0.5051 — beats the *original cluster* G3
number (0.000637) on every aggregate metric. **This aggregate comparison is misleading and is
explicitly retracted below.** Re-running G3 fresh on the WSL checkout (needed because G3's
original per-specimen `metrics.csv` is git-ignored/cluster-only, so no paired test against it was
possible) gave chamfer 0.000617 — itself ~3% better than the cluster G3 number, for reasons
traced to `fitter_3d/trainer_hierarchical.py`/`optimise_moonshot.py` being wholesale different
files on `origin/feature/investigation`'s current tip than whatever the cluster's original G1-G1c
series ran under (introduced by an unrelated commit, `3ca5799`, "ship scale_cap..." — confirmed
via `git show --stat`, not assumed). **The correct, paired, same-code/same-hardware comparison is
G1d vs. the WSL G3 rerun**, and on that comparison G1d is statistically indistinguishable from
zero-init (chamfer p=0.72, F@0.01 p=0.85, F@0.02 p=0.70, n=50) — **parity, continuing the plateau
established at G1c, not a fourth step of improvement.** The historical G1→G1b→G1c series remains
valid (run pairwise-consistently within the cluster session); only the specific claim "G1d
continues the trend into a win" is retracted, and only because it relied on an invalid
cross-code-version comparison.

**Decision, per this file's own "credible next steps" list above and the handoff's explicit
instruction not to chase a number past its informative endpoint**: the dose-response sweep on
training pose-noise scale has reached parity and plateaued (G1c → G1d, both statistically tied
with zero-init); a fifth point (G1e, an even lower `pose_scale`) is not pursued, since the trend
that would justify it did not, in fact, continue.

**New mechanism tested instead — an SMPLify-X-style init-anchor regularizer** (`--init_anchor_
weight` / `--init_anchor_proximal_mult`, new in `optimise_hierarchical.py`/
`trainer_hierarchical.py`, weighted toward proximal joints per the D/E/F basin-structure finding
above): pulls `joint_rot` back toward its initial value throughout optimization instead of only
setting the starting point. Mechanically confirmed active (proximal-joint displacement from init
reduced 3.4x, ~42-43° → ~12-13°, measured directly from `H2_joint.npz`) in both configurations
tested:

| Arm | vs. WSL G3 (paired, n=50) | Verdict |
|---|---|---|
| G1d's learned init + proximal anchor | chamfer p=0.81, F@0.01 p=0.84 | Null — no effect over G1d alone either (p=0.91-1.00) |
| **Zero-init + proximal anchor** (no learned network) | chamfer: paired-t p=0.233, sign-test p=0.065 (32/18 win); F@0.02 p=0.097 (30/20) | **Not significant on the primary (paired-t) metric.** The 0.065 figure is a secondary sign-test on win/loss direction, not the paired t-test used everywhere else in this document — reported separately here after a 2026-08-25 reproducibility audit found the two conflated in an earlier draft of this table. Still directionally consistent (wins the majority on all three metrics), but weaker evidence than "trending toward significance" implied |

Read together: anchoring optimization toward a *learned* init doesn't help once the partitioned
data term has already solved the leg-assignment multimodality (the actual mechanism D/E/F
identifies), but anchoring toward *rest pose* specifically shows a consistent (if not yet
significant) directional improvement across all three metrics — plausibly because bench50
specimens are genuinely close to rest pose (consistent with zero-init's surprising strength
throughout this whole investigation) and restraining the optimizer from chasing anatomically
implausible proximal rotations mildly helps. Full reasoning, code changes, and every intermediate
number: `diagnostics/anatomical_pose_init/CHAIN_OF_THOUGHT_20260824_WSL.md`.

**Recommended next step, if this thread is picked up again**: repeat the zero-init+anchor arm
with a different seed and/or a higher `--init_anchor_proximal_mult` to check whether the trend
strengthens or was a favorable draw at n=50 — not run tonight given time already spent on four
full fit+eval cycles.

## Artifacts

- `diagnostics/moonshot/runs/{bench50_G1_learned,bench50_G1b_learned_lowpose,bench50_G3_zero}/metrics.csv`
  (per-specimen Chamfer/F-score, all 50 specimens, via `eval_run.py`)
- `submit_bench50_G1G3.sbatch` (SLURM 3102069), `eval_bench50_G1G3.sbatch` (SLURM 3102270),
  `submit_bench50_G1b.sbatch` (SLURM 3102400), `eval_bench50_G1b.sbatch` (SLURM 3102473)
- `bench50_learned_init.npz` (G1), `bench50_learned_init_lowpose.npz` (G1b)
- `generate_synth_5000_lowpose.slurm` (SLURM 3102282, corpus), `submit_train_leg_pose_lowpose.sbatch`
  (SLURM 3102293, retrain) — `diagnostics/moonshot/synth_large_25pc_lowpose/ground_truth.npz`,
  `diagnostics/anatomical_pose_init/learned_init_lowpose/best_model.pt`
- WSL continuation (2026-08-24/25): `diagnostics/moonshot/runs/{bench50_G1d_learned_lowpose05,
  bench50_G3_zero_wsl,bench50_G1d_ianchor,bench50_zero_ianchor}/metrics.csv`,
  `diagnostics/anatomical_pose_init/CHAIN_OF_THOUGHT_20260824_WSL.md` (full reasoning + every
  intermediate number, including the retracted aggregate-only G1d claim and its correction)
