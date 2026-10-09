# Correspondence-Free Anatomical Representation + Raw-Scan Generalization — RESULTS (2026-08-20)

Follow-up to `RESULTS_ABC_DEF.md` (learned leg-pose initializer + basin-structure mechanism).
That experiment's training/evaluation used template-vertex-indexed leg labels (`vertex_labels`,
derived from skinning weights), which only exist for synthetic corpora generated as deformations
of the template itself. This experiment asks whether the same learned initializer can be applied
to raw, real, unseen-morphology scans with no such correspondence — and along the way finds and
fixes a real defect in a shared geometric primitive.

## Stage 4A: what SMILify already has for scan→anatomy association

Investigated before writing new code, per the explicit question "at what point does an arbitrary
scan become associated with the model's anatomical coordinate system?" Found:
`cheap_anatomical_init.py`'s `estimate_global_pose()` (PCA rigid alignment, correspondence-free)
and `fitter_3d/geom_leg_init.py`'s `analytic_coxa_anchors()`/`assign_points_to_legs()`
(coxa-position-from-rigid-alignment, then nearest-anchor point assignment) already form most of a
correspondence-free scan→limb pipeline. Two real gaps found: no body-core class (every point was
forced into one of 6 legs), and `optimise_hierarchical.py --init_joint_rot_from` has no
global-orientation override at all (always zero-inits `global_rot`) — the latter deferred to a
separate G2/G4 experiment (see below) to avoid confounding two hypotheses in one test.

## A validated defect in a shared primitive: `assign_points_to_legs` (coxa method)

Tested the existing `assign_points_to_legs` (single-coxa-point nearest-anchor assignment) against
`synth_clean`'s known template correspondence (ground truth: per-vertex leg membership from
skinning weights, near-balanced across all 6 legs, 670-923 verts each). Found systematic
**mesothoracic (middle-leg) starvation**: `l2_r`/`l2_l` recall only **0.349** pooled (n=12),
vs. macro-F1 0.656 overall — reproduced on `synth_clean` (known correspondence), not scan noise.
Mechanism: a coxa is a single point; an extended limb's surface can curve geometrically closer to
a *neighboring* leg's coxa than to its own under pure nearest-anchor-point competition.

**This also revises the historical interpretation of Arm B** (`cheap_anatomical_init.py`'s
end-to-end FAILURE verdict, `out_ceiling_20260820/RESULTS.md`): that experiment used this exact
defective assignment internally. The correct statement is now: *Arm B demonstrated failure of the
complete cheap-anatomical pipeline, but the failure's attribution to the single-rigid-rod IK
heuristic alone is unconfirmed — the upstream point-to-leg assignment stage was itself
systematically defective and may have contributed.* Arm B's own result stands (it was fit and
measured correctly); its mechanistic story does not.

## Fix: nearest-articulated-chain assignment (validated, standard technique)

Grounded in literature before implementing (not assumed): "distance to nearest bone segment," not
distance to a single joint, is the standard skeletal-rigging point-assignment primitive (classical
skinning-by-distance; RigNet's volumetric-distance-to-nearest-bone features use the same idea —
[Zhan et al., MoRig, arXiv:2210.09463](https://ar5iv.labs.arxiv.org/html/2210.09463); see also
[Kuffner 2004](https://www.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf)
for the related controlled-perturbation-sampling grounding used in the D/E/F experiment).

Implemented `assign_points_to_legs_chain` + `rest_chain_points` in `fitter_3d/geom_leg_init.py`
(the old `assign_points_to_legs` function is preserved, renamed `assign_points_to_legs_coxa` with
a backward-compat alias, so every historical caller — `cheap_anatomical_init.py`,
`simple_leg_heuristic.py` — gets byte-identical behavior; nothing silently changed).

| Method | macro-F1 (pooled, n=12) | ML/MR (middle-leg) recall |
|---|---:|---:|
| coxa (old) | 0.656 | 0.349 |
| **chain (new)** | **0.831** | **0.768** |

Chain wins on 11/12 specimens (1 near-tie, still non-worse on ML/MR recall there). A real,
consistent repair, not overfit to one specimen — `diagnostics/anatomical_pose_init/validate_leg_assignment.py`.

### Residual failure mode, and what explains it

Chain is not a complete fix: `synth_011`'s `l2_r` recall is still only 14.8%, most of its points
still captured by `l3_r`. Two follow-up diagnostics (`validate_leg_assignment_v2.py`) isolate why:

1. **GT posed-chain ceiling probe** (same nearest-polyline rule, but using the *actual* per-specimen
   articulated pose via forward kinematics instead of the straight rest pose — diagnostic only,
   needs GT pose, never available for real inference): macro-F1 0.831→**0.902**, ML/MR recall
   0.768→**0.858**. Real further improvement on average — but NOT uniform: `synth_005`
   (0.788→0.431) and `synth_003` (0.832→0.678) get *worse*, plausibly because a genuinely bent
   real leg's posed polyline can self-overlap and confuse nearest-point competition worse than the
   straight rest-pose chain. So: pose-aware geometry helps on average, isn't a strict win, and
   isn't deployable anyway (no GT pose for real scans).
2. **Soft (temperature-softmax) assignment diagnostic**: top-1 accuracy 0.843 (temperature-
   invariant, as expected for an argmax-preserving transform), but **top-2=0.976, top-3=0.995** —
   the true leg is in the top-2 candidates 97.6% of the time even when top-1 is wrong. But
   `synth_011`'s specific `l2_r` failure is NOT rescued by softening: even at the most ambiguous
   temperature tested (τ=0.2), `P(l3_r)=0.45` still dominates `P(l2_r)=0.16`. Per the pre-registered
   interpretation rule: this specific case is a genuine geometry problem, not an
   argmin-destroys-information problem — but the aggregate top-2/3 evidence still supports soft
   representation as a real information superset of hard assignment for future retraining.

## Body-core: the same defect, the same fix, independently confirmed

A hard global distance-threshold rule (point vs. root/thorax position) does not cleanly separate
body-core from leg on `synth_clean` (τ=0.05: only 23% of true leg points correctly included;
τ=0.15: 71% included but 36% of true core wrongly included) — no threshold clears both bars.
Root-cause: a single root point badly under-represents an elongated body (core_recall 0.366 in a
7-way, root-point-only classifier) — the *same* single-point-for-an-extended-region failure mode
the coxa method had for legs, now on the body axis. Same fix, same validated pattern: built
`body_core_chain_points` (head→thorax→gaster joint polyline, from the rig's own existing
`part_groups.py`-style joint names) and `assign_points_to_parts_chain` (7-way: body-core + 6 legs,
one unified nearest-chain rule, no arbitrary threshold) in `fitter_3d/geom_leg_init.py`.

| | Overall 7-way accuracy | core recall | leg recall (mean) | macro-F1 |
|---|---:|---:|---:|---:|
| single-point body-core | 0.581 | 0.366 | 0.831 | 0.608 |
| **chain body-core** | **0.723** | **0.663** | 0.794 | **0.697** |

(`validate_7way_soft.py`.) A real, substantial, independently-confirmed instance of the same
mechanism — not yet perfect (core recall 0.663 still leaves ~1/3 of true core points misassigned),
but a validated, principled improvement, not an arbitrary threshold.

## Practical constraint: hard labels, not soft, for the existing checkpoint

The trained checkpoint (`learned_init_20260820_25pc/best_model.pt`) was trained on hard one-hot
labels (fixed input dimensionality, exactly-one-1.0 per row). Feeding it soft probabilities now
would be an unevaluated train/inference distribution shift, not a fair test of this checkpoint —
so the bench50 adapter uses the validated **hard** hard 7-way chain assignment. A soft-label
retrain is a well-motivated, clearly-scoped follow-up (the top-2/3 evidence above supports it),
not attempted here.

## Phase 3+4: applying the existing checkpoint to raw, unseen bench50 scans (no fitting yet)

Built the full pipeline (`infer_bench50_raw_scan.py`, `generate_bench50_learned_init.py`): raw
`.obj` → normalize (verified to match `fitter_3d.utils.load_meshes`'s exact convention: center,
divide by max abs coordinate — checked empirically that this puts a bench50 specimen's bbox
diagonal, 2.330, right on top of the template's own native scale, 2.346, not assumed) →
`estimate_global_pose` using the **actual PCA rotation** (not the identity shortcut
`cheap_anatomical_init.py` uses, which is valid only because `synth_clean` is canonically
pre-aligned by construction and real scans are not) → 7-way chain assignment →
`body_core_canonicalize` (same function training used — verified generic over any same-length
points+labels, not template-vertex-index-dependent) → same point sampling/onehot scheme as
training → trained PointNet.

Before any fitting (explicit agreed gate: check whether predictions look anatomically meaningful
first), ran on all 50 bench50 specimens (0 skipped for degenerate labeling):

- **Stability**: mean resample variance (3 independent random point-subsample trials per
  specimen) = **0.48°** (49/50 specimens under ~1°, one outlier at 2.04°) — not noisy garbage.
- **Magnitude**: mean predicted rotation magnitude 11.5° — a modest, plausible amount of natural
  articulation for a static pinned specimen, not a degenerate/exploded number.
- **Structure**: consistent front>hind>middle-leg magnitude ordering (`l1`≈13.3-13.6°,
  `l3`≈9.5-12.0°, `l2`≈10.1-10.6°) and rough left/right symmetry, stable across 50 independent,
  taxonomically diverse real specimens (std 2-3.5° across specimens, much smaller than the
  between-leg-group differences) — evidence of genuine, non-random, leg-position-dependent signal
  on morphology the network never saw during training (many genera: Acanthostichus, Acromyrmex,
  Cephalotes, Dorylus, Eciton, Odontomachus, Solenopsis, Strumigenys, ...).

## Next: G1 vs. G3 real fits on bench50 (identity global orientation, isolated from the
global-rotation question)

Per the explicit isolation decision (don't confound the raw-scan representation test with the
separate global-orientation hypothesis, which needs a new `--init_global_rot_from`-style fitter
flag not yet built): submitted the real D1 fit (`optimise_hierarchical` + `optimise_moonshot`,
identical recipe to every other arm in this experiment series) for all 50 bench50 specimens under
two conditions — G3 (zero/default init) and G1 (learned raw-scan init, this pipeline's output) —
both at identity global orientation (the fitter's actual default regardless, so no code change
needed for this specific comparison). SLURM array job 3102069. No ground-truth pose exists for
real scans, so this can only be scored on Chamfer/F-score (target-mesh-only metrics, no
`leg_acc`/`all_feature_R`, which need known correspondence) — see the follow-up results file for
outcome once the jobs complete.

## Artifacts

- Shared primitive fix (auditable, old behavior preserved): `fitter_3d/geom_leg_init.py`
  (`assign_points_to_legs_coxa`/`_chain`, `rest_chain_points`, `body_core_chain_points`,
  `assign_points_to_parts_chain`)
- Validation scripts: `validate_leg_assignment.py`, `validate_leg_assignment_v2.py`,
  `validate_7way_soft.py`, `inspect_bench50_geometry.py`
- Raw-scan pipeline: `infer_bench50_raw_scan.py`, `generate_bench50_learned_init.py`
- `bench50_learned_init.npz` (this run's actual per-specimen leg-pose predictions)
- Fit jobs: `submit_bench50_G1G3.sbatch` (SLURM 3102069)
