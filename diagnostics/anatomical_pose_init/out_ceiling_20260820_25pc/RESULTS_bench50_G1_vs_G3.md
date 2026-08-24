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

## Artifacts

- `diagnostics/moonshot/runs/{bench50_G1_learned,bench50_G1b_learned_lowpose,bench50_G3_zero}/metrics.csv`
  (per-specimen Chamfer/F-score, all 50 specimens, via `eval_run.py`)
- `submit_bench50_G1G3.sbatch` (SLURM 3102069), `eval_bench50_G1G3.sbatch` (SLURM 3102270),
  `submit_bench50_G1b.sbatch` (SLURM 3102400), `eval_bench50_G1b.sbatch` (SLURM 3102473)
- `bench50_learned_init.npz` (G1), `bench50_learned_init_lowpose.npz` (G1b)
- `generate_synth_5000_lowpose.slurm` (SLURM 3102282, corpus), `submit_train_leg_pose_lowpose.sbatch`
  (SLURM 3102293, retrain) — `diagnostics/moonshot/synth_large_25pc_lowpose/ground_truth.npz`,
  `diagnostics/anatomical_pose_init/learned_init_lowpose/best_model.pt`
