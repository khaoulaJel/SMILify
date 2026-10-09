# Correspondence-accuracy metric (2026-08-19)

## Why

Everything measured on this branch so far (D1 promotion, GNC arms, split_distal, stratified
sampling) is scored by Chamfer/edge/etc. Those can improve while correspondence -- which scan
point got matched to which anatomical template region -- stays broken or gets worse. Before
building anatomical constraints on top of GNC, we need a metric that answers the correspondence
question directly, independent of the geometric loss, so "the constraint helped" can be told
apart from "the loss landscape got smoother but matching didn't change."

## What it measures

For each scan (target) point sampled from a synthetic specimen with known ground-truth
correspondence, two questions:

1. **True anatomical label** -- which template region did this point actually come from? Known
   exactly because the `synth_clean` corpus is a posed/deformed copy of the template (see
   `diagnostics/moonshot/SYNTHETIC_ROUNDTRIP.md`), not an independent scan+remesh -- face
   topology (`f_idx.shape`) is asserted to match the template every run before trusting this.
2. **Matched anatomical label** -- which template region does the FITTED mesh's nearest vertex
   to that point belong to? This is deliberately the same correspondence direction
   `TargetPartition.update` and `_partitioned_chamfer`'s target->source term use
   (`fitter_3d/trainer_hierarchical.py`) -- the metric audits the fitter's own rule, not a
   reimplementation of it.

Confusion matrices (true x matched counts, not just a mismatch rate) are accumulated across all
12 `synth_clean` specimens, at three granularities:

- **leg-level** (`l{1,2,3}_{r,l}`, 6x6): cross-leg swaps, unrestricted nearest-vertex search
  over the whole fitted mesh (matches what a target point actually experiences).
- **within-leg segment-level** (`co/tr/fe/ti/ta/pt`, 6x6): RESTRICTED to points whose leg-level
  match was already correct, isolating segment swaps (e.g. femur<->tibia) from leg swaps --
  same split Task 6's D2b probe used, but the full matrix instead of one binary rate.
- **antenna** (side r/l, and `an_1/an_2/an_3` proximal->distal segment), tracked completely
  separately from legs per the task spec. Antenna is anatomically the same kind of thin chain
  as a leg (`an_3` has 4 dominant-weight vertices on the template, thinner than the leg
  pretarsus's 8) but is currently NEVER its own `TargetPartition` group in any GNC run
  (`split_anterior` was never passed) -- so antenna correspondence is presently unconstrained by
  construction, which is exactly why this number matters as a baseline before anatomical
  constraints touch it.

## Where it lives

- `labels.py` -- per-template-vertex/face ground-truth labels (region/leg_id/leg_seg/
  ant_side/ant_seg), built from `M["jnames"]`/`M["dominant"]`, the same source
  `fitter_3d.trainer_hierarchical.vertex_groups` uses. Extends that existing machinery to cover
  antenna, which the D2a/D2b probes never labeled (they folded head/mandible/antenna into one
  unlabeled bucket).
- `confusion.py` -- `ConfusionAccumulator` + the three confusion-matrix builders
  (`leg_confusion`, `within_leg_segment_confusion`, `antenna_confusion`), reusing
  `TargetPartition`'s nearest-vertex correspondence rule via `pytorch3d.ops.knn_points`
  directly (not the `TargetPartition` class itself, since that class only knows leg-granularity
  groups by default -- reusing its exact *rule*, unrestricted-nearest-vertex, is what matters
  for fidelity, not reusing the class).
- `run_audit.py` -- orchestrator. Targets every converged arm that shares the `synth_clean`
  ground-truth corpus (`RUNS` list), so results are directly comparable across arms without a
  different-corpus confound. Output: `out/correspondence_confusion.json` (full matrices per
  run) plus a summary table on stdout.

Known data-quality wrinkle, handled not patched: a small number of template faces straddle two
*different* legs' coxae (adjacent-leg boundary), producing a 3-way tie in the per-face leg_id
majority vote that an independent "region" majority vote doesn't share. Gating `true_is_leg`/
`true_is_ant` on `leg_id`/`ant_seg` being non-None directly (not on the separately-voted
`region` field) drops these genuinely-ambiguous boundary points instead of crashing or silently
mislabeling them.

## First results (`synth_clean`, 12 specimens, n_sample=8000/specimen, seed=1)

| run                              | leg_acc | within-leg seg_acc | antenna side_acc | antenna seg_acc |
|-----------------------------------|--------:|--------------------:|------------------:|------------------:|
| SYN_clean_w5 (stock baseline)      | 0.857   | 0.828                | 0.985              | 0.770 |
| SYN_clean_pose25_gnclegonly_w5     | 0.858   | 0.823                | 0.986              | 0.804 |
| SYN_clean_pose25_splitdistal_w5    | 0.871   | 0.838                | 0.972              | 0.789 |
| SYN_clean_pose25_gtinit_w5         | 0.938   | 0.866                | 1.000              | 0.828 |
| SYN_clean_pose25_stratq05_w5       | 0.871   | 0.838                | 0.969              | 0.780 |
| SYN_clean_pose25_stratq10_w5       | 0.859   | 0.847                | 0.977              | 0.768 |
| SYN_clean_pose25_stratq20_w5       | 0.850   | 0.839                | 0.975              | 0.779 |
| SYN_clean_pose25_c5protected_w5    | 0.868   | 0.835                | 0.964              | 0.766 |

Sanity checks that the metric is measuring something real, not noise:

- **`gtinit` (ground-truth pose init) wins every column**, most dramatically on leg_acc
  (0.938 vs 0.857 baseline, xleg-error 6.2% vs 14.3%) and gets antenna side accuracy to a
  perfect 1.000. This is independent confirmation of Task 6's finding (`init_joint_rot`
  override "nearly quadruples leg_distal R") using a completely different measurement (matching
  accuracy, not downstream morphometric R) -- the two metrics agree.
- **The leg-level confusion matrix is spatially sane**: cross-leg confusion concentrates
  between *adjacent* legs on the *same side* (l1_r<->l2_r 13%, l2_l<->l3_l 10-13%), essentially
  zero between left/right mirror pairs of the same leg number (l1_r<->l1_l ~0%) -- legs on
  opposite sides of the body are far apart in 3D, adjacent legs on the same side are close, so
  this is the physically expected confusion pattern, not an artifact.
- **Within-leg segment confusion reproduces the known pretarsus-starvation failure
  quantitatively**: baseline pretarsus (`pt`) points get matched to the tarsus (`ta`) 69% of the
  time (row_normalized `pt->ta` = 0.69), i.e. the pretarsus is essentially swallowed by its
  neighbor -- exactly the mechanism Task 6 diagnosed from chamfer-sampling starvation
  (`ti/ta/pt` hold 2.4% of a leg's area). Antenna shows the same shape at its own thin tip:
  `an_3` (4 dominant verts, thinner than pretarsus's 8) gets matched to `an_2` 61% of the time.

None of the tested interventions (GNC leg-only, split_distal, stratified sampling, c5-protected
kernel) move leg_acc or seg_acc by more than ~1.5 points from baseline in either direction on
this corpus -- i.e. **by this metric, none of them have yet demonstrably fixed correspondence**,
only `gtinit` (which sidesteps the correspondence problem by not needing pose search at all)
does. That is itself the useful null result the task asked this metric to be able to detect.

## How to run

```
conda activate pytorch3d
python diagnostics/correspondence_accuracy/run_audit.py
```

Needs `Stage_3_deform_fine.npz` for each run in `RUNS` (edit the list to add new arms) and the
matching corpus under `diagnostics/moonshot/<corpus>/`. Both `SYN_clean_pose0_*`/`drop30`/
`drop60` corpora referenced by earlier probes no longer exist on disk (regenerated at run time,
not persisted) -- this run targets only arms sharing the still-present `synth_clean` corpus so
every row above is a fair, matched comparison. Regenerating the pose0/drop30/drop60 corpora
(via `diagnostics/moonshot/make_synth_corpus.py` + whatever produced the drop variants) to
extend this audit to those conditions is a natural next step but was not attempted here to
avoid guessing at seeds/parameters not recorded in this session.
