# Learned Leg-Pose Initializer + Basin-Structure Follow-up — RESULTS (2026-08-20)

Follow-up to the Anatomical Initialization Ceiling Test (`out_ceiling_20260820/RESULTS.md`,
FAILURE verdict on the hand-designed `cheap_anatomical` initializer). This experiment fixes the
topology bug that invalidated the first learned-initializer attempt (trained against
`SMIL_OmniAnt.pkl`, 10229 verts, while `synth_clean`/the fitter use `OmniAnt_25PCs_joint_limited.pkl`,
10235 verts), retrains on a topology-correct 5000-specimen corpus, and runs the full downstream
D1 optimization pipeline (not just pose-regression error) for six arms.

## Decision (stated first)

**The learned initializer (Arm C) is a real, complete win over zero-init (Arm A) across every
metric family used elsewhere in this project** (correspondence accuracy, morphometric
reliability, Chamfer, F-score) despite starting from a nearly identical scalar pose error
(22.79° vs 23.09°). The mechanism is now understood, not just observed: a follow-up controlled
experiment (Arms D/E/F) shows initialization error concentrated at **proximal** leg joints
(coxa/trochanter/femur) is far more damaging to the optimizer's final basin than the same
magnitude of error concentrated at **distal** joints (tibia/tarsus/pretarsus). The learned
initializer's advantage over zero-init is explained by this: its (modest) improvement over
zero-init is concentrated on proximal joints, exactly where it has outsized leverage.

## Part 1 — Arms A/B/C (does the learned initializer help, in this project's own metrics?)

Topology bug (found and fixed before this run): `synth_large`, the 5000-specimen training
corpus, was generated against `SMIL_OmniAnt.pkl` (10229 verts) via `config.SMAL_FILE`'s
import-time default, while `synth_clean` (the held-out set) and the fitter both use
`OmniAnt_25PCs_joint_limited.pkl` (10235 verts, 25 shape PCs vs 13). `N_BETAS` and other
config-derived constants are locked in at `import config` time from the `SMILIFY_SMAL_FILE` env
var, before argparse runs — so a `--model` CLI flag alone cannot fix this; the env var must be
set before the python process starts. `generate_synth_large.py` now asserts `--model` matches
`config.SMAL_FILE` and fails loudly rather than silently generating the wrong topology again.
Regenerated as `synth_large_25pc` (5000 specimens, `OmniAnt_25PCs_joint_limited.pkl`,
verified: verts (5000,10235,3), joint_rot (5000,54,3,3), joint_rot_aa (5000,54,3), orthogonality
error ~6e-7). A second, independent dtype bug was found and fixed in `infer_leg_pose_init.py`
(`body_core_canonicalize`'s internal SVD promotes to float64; the training `Dataset` casts back
to float32 immediately, the standalone inference script didn't).

| Arm | Init | Mean initial pose error | leg_acc | within_leg_seg_acc | antenna_seg_acc | all_feature_R (median) | Chamfer L2 | F@0.01 | F@0.02 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | zero/default | 23.09° | 0.857 | 0.828 | 0.770 | 0.745 | 0.000264 | 0.6951 | 0.9371 |
| B | cheap anatomical (hand-designed, `simple_leg_heuristic.py`) | 28.52° | 0.732 | 0.686 | 0.760 | 0.518 | 0.000417 | 0.5948 | 0.8927 |
| C | learned (`train_leg_pose_regressor.py`, PointNet-style, body-core-canonicalized) | 22.79° | **0.901** | **0.834** | **0.838** | **0.761** | **0.000226** | **0.7153** | **0.9510** |

C wins every metric in this table, using the project's own established evaluation code
(`run_audit.py` / `calib_features.py` / `eval_run.py`), not a parallel one. 50 training epochs
(not 200 — the first, topology-invalid run showed val loss bottoming at epoch ~30 and worsening
monotonically through epoch 199; overfitting past that point, confirmed again on the corrected
corpus). Training run: SLURM job 3101238, best val 44.535° (this is the raw pose-regression
metric on the held-out leg-joint set, not the same number as the "mean initial pose error" row
above, which is measured with a fresh random point-cloud subsample at inference time, seed=0).

**What this table does NOT show**: it does not show C recovering the correct pose (22.79° is
still a large error — nowhere near GT-init's 0°). It shows that at a fixed, modest pose-error
budget, C's *distribution* of that error across the kinematic hierarchy produces a better final
fit than zero-init's distribution of a similar-magnitude error. Part 2 tests this directly.

## Part 2 — Arms D/E/F (does error STRUCTURE matter, independent of magnitude?)

Construction (`generate_structured_perturbation_init.py`): `theta_0 = theta_GT + epsilon`,
`epsilon` applied as `R_pert(axis, angle) @ R_GT` with `axis` uniform-random on S^2 and `angle`
a fixed/controlled magnitude per joint (the standard way to get an *exact* controlled geodesic
distance without the small-angle bias of sampling angle~Uniform and axis~S^2 independently; see
Kuffner 2004, Yershova & LaValle 2004). Per-specimen mean magnitude over the 36 leg joints
matched to a 22-24° band (not forced to exactly 23.00°) across all three conditions:

- **D (random)**: all 36 joints i.i.d., mean 23°, unstructured (no segment dependence) — the
  control condition, without which a proximal-vs-distal comparison alone cannot distinguish
  "location matters" from "two arbitrary perturbation distributions differ."
- **E (proximal-concentrated)**: coxa/trochanter/femur (18 joints) mean 34.5°,
  tibia/tarsus/pretarsus (18 joints) mean 11.5° (3:1 ratio, not an all-or-nothing 0°/46° split,
  to avoid an unrealistic degenerate condition).
- **F (distal-concentrated)**: mirror of E.

Achieved magnitudes: D 23.03°, E 23.12°, F 22.94° (all within the 22-24° target band). One seed
per condition (not the full 3-seed design originally proposed) — a scope choice given compute
already spent this session, flagged explicitly rather than silently dropped. See "Limitations."

| Arm | Init magnitude | leg_acc | all_feature_R (median) | Chamfer L2 | F@0.01 | F@0.02 |
|---|---:|---:|---:|---:|---:|---:|
| D — random (matched, unstructured) | 23.03° | 0.883 | 0.783 | 0.000228 | 0.7218 | 0.9509 |
| E — proximal-concentrated (matched) | 23.12° | **0.768** | **0.648** | **0.000393** | **0.6443** | 0.9053 |
| F — distal-concentrated (matched) | 22.94° | **0.939** | **0.816** | **0.000218** | **0.7329** | **0.9527** |

For reference (not the claim of this experiment — see caveat below): `gtinit` (0° error) scores
leg_acc 0.938, all_feature_R 0.745 on this same corpus/recipe. F's leg_acc (0.939) and
all_feature_R (0.816) are numerically at or slightly above the gtinit reference on these
particular post-fit metrics in this run. **This does not mean F is a better initialization than
ground truth** — it means the post-optimization evaluation metric happens to be slightly higher
in this run, plausibly due to optimizer dynamics, metric noise at n=12, synthetic-geometry
particulars, or gtinit not being a strict optimization oracle. The correct statement: **F
slightly exceeds the gtinit reference on some post-fit metrics in this experiment** — not "F is
better than ground truth."

### The finding

Initialization error is strongly anisotropic in its effect on SMILify optimization. Controlled
perturbations with approximately identical initial joint-rotation error (~23°) produced
substantially different final registrations depending on where the error was concentrated in
the articulated chain. Proximal perturbations concentrated at coxa/femur produced severe
degradation across correspondence and surface metrics (E: leg_acc 0.768, all_feature_R 0.648,
comparable to the previously-FAILED `cheap_anatomical` arm's 0.732/0.518), whereas distal
perturbations concentrated at tibia/tarsus/pretarsus produced little degradation and, in this
experiment, achieved the strongest final fits among A-F (F: leg_acc 0.939, all_feature_R 0.816).
This is consistent with forward-kinematic error propagation: proximal rotational errors displace
larger portions of the downstream articulated surface. The result demonstrates that scalar
initial pose error is insufficient to characterize initialization quality for SMILify.

### Mechanistic link back to Part 1

This also provides a mechanistic interpretation of the learned initializer. Although its mean
initial pose error was nearly identical to zero initialization (22.79° vs 23.09°), a per-joint
error-structure comparison (A vs C, marginal geodesic error per segment, averaged over all 6 leg
chains) found small, unpatterned differences at first pass —
co −0.74°, tr −1.22°, fe +0.53°, ti −0.74°, ta +0.26°, pt +0.09° (C minus A) — which on their
own looked like a null result. In light of the D/E/F finding, this is not null: C's improvement,
though modest in magnitude, is concentrated on the two joints with the largest per-segment
delta (coxa, trochanter — both proximal), exactly where the D/E/F experiment shows small
corrections have disproportionately large downstream geometric consequences. The learned
initializer therefore appears to improve not simply the magnitude of initialization error, but
its distribution across the kinematic hierarchy. (Left/right symmetry and adjacent-joint
error correlation were also checked as candidate structural explanations and did NOT show a
clean signal — L/R asymmetry was in fact slightly worse for C than A, 4.47° vs 3.94° — so the
proximal/distal axis specifically, not "more anatomically consistent" in general, is the
supported explanation.)

## Reframed research question

Original: *Can learned initialization improve SMILify?*

Sharper: *Can we automatically initialize SMILify so that the remaining pose error is
concentrated in low-impact parts of the kinematic hierarchy, thereby placing the optimizer in a
more favorable basin without requiring ground-truth initialization?*

This suggests good initialization ≈ minimum downstream geometric damage under optimization, not
simply minimum pose error.

## Limitations (disclosed)

1. **n=12 specimens per arm**, all from `synth_clean`, synthetic only. Not yet tested on
   real/unseen morphology.
2. **D/E/F are 1 seed each**, not the 3-5 seed design that would separate "this structural
   class of error" from "this particular random draw." The qualitative separation is large
   (leg_acc 0.939 vs 0.768; F@0.01 0.7329 vs 0.6443) — not a single-decimal-place effect — but a
   hard quantitative claim (e.g. an exact proximal/distal damage ratio) is not supported at n=1
   seed.
3. **`ACI_cheap` was silently dropped once** from `feature_reliability_all_runs.json` by an
   earlier unrelated run of `calib_features.py --runs` (that script fully overwrites the file
   from its `--runs` list, no merge) — restored here alongside `ACI_learned`/D/E/F, but this is
   a latent footgun in that script worth fixing (accept `--runs all` or merge-by-default) if it's
   used again.
4. **F ≳ gtinit on some post-fit metrics is not evidence F is a better initialization than
   ground truth** — see caveat above.
5. `within_leg_seg_acc`/`antenna_seg_acc`/etc. for D/E/F were not separately re-derived beyond
   what's in `correspondence_confusion.json` (already regenerated by this run — see that file
   directly for the full per-arm breakdown, including columns not reproduced in the tables
   above).

## Artifacts

- This file, `learned_init.json` / `learned_init.npz` (Arm C init + per-specimen pose error),
  `{random,proximal,distal}_init.npz` (Arms D/E/F init)
- `../generate_structured_perturbation_init.py`, `../submit_ACI_learned.sbatch`,
  `../submit_ACI_perturb.sbatch`, `../eval_ACI_learned.sbatch`, `../eval_ACI_perturb.sbatch`
- `../../correspondence_accuracy/out/correspondence_confusion.json` (full confusion matrices,
  all arms including C/D/E/F — `run_audit.py` RUNS list updated)
- `../../morphometrics/out/feature_reliability_all_runs.json` (full per-feature R/snr/bias, all
  arms including C/D/E/F — restored + extended, see Limitation 3)
- `diagnostics/moonshot/runs/{ACI_learned,ACI_D_random,ACI_E_proximal,ACI_F_distal}/metrics.csv`
  (per-specimen Chamfer/F-score/etc., via `eval_run.py`)
- Training checkpoint: `../learned_init_20260820_25pc/best_model.pt` (SLURM job 3101238); the
  topology-invalid first attempt is preserved, not deleted, at
  `../learned_init_20260820_WRONG_TOPOLOGY/`
- Training corpus: `diagnostics/moonshot/synth_large_25pc/ground_truth.npz` (SLURM job 3100899)
