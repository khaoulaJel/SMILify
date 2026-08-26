#!/usr/bin/env bash
# Dense per-vertex correspondence oracle (2026-08-25, Phase 10 step 1 follow-up): does
# perfect DENSE correspondence -- not just group/leg-level, per why_partitions_null.py's own
# 16.7%/83.3% between-part/within-part split -- move leg_acc/seg_acc beyond the group-only
# oracle (Oracle_GT_partition, leg_acc=0.937)? Same D1 recipe, zero-init pose, differing ONLY
# in the added dense additive term (w_dense_gt=1.0).
set -eo pipefail

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
T=Dense_GT_oracle

echo "[$(date +%H:%M:%S)] $T: hierarchical stage (dense GT correspondence oracle active)"
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
  --dense_gt_correspondence_from $D/ground_truth.npz \
  --results_dir $R/${T}_hier

echo "[$(date +%H:%M:%S)] $T: D1 fit"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/${T}_hier/H2_joint.npz \
  --results_dir $R/$T

echo "[$(date +%H:%M:%S)] $T: eval_run.py"
python -u diagnostics/moonshot/eval_run.py --run_dir $R/$T --mesh_dir $D

echo "[$(date +%H:%M:%S)] $T done."
