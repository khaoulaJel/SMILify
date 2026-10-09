#!/usr/bin/env bash
# Correspondence-oracle test (2026-08-25, Phase 10 design doc step 1): does perfect (ground-truth)
# correspondence, fed through the EXACT partition-injection hook a learned network would use
# (--oracle_gt_partition_from, reusing PartFieldPartition unmodified), move leg_acc beyond what
# the current recipe already gets from zero-init -- BEFORE any network is trained to predict it.
#
# Same D1 recipe as SYN_clean_zero_wsl (--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0,
# D1_low.yaml, zero-init pose), differing ONLY in the added oracle partition -- isolates
# correspondence-availability as the single manipulated variable, same discipline as every other
# arm in this investigation.
set -eo pipefail

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
T=Oracle_GT_partition

echo "[$(date +%H:%M:%S)] $T: hierarchical stage (oracle GT partition active)"
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
  --oracle_gt_partition_from $D/ground_truth.npz \
  --results_dir $R/${T}_hier

echo "[$(date +%H:%M:%S)] $T: D1 fit"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/${T}_hier/H2_joint.npz \
  --results_dir $R/$T

echo "[$(date +%H:%M:%S)] $T: eval_run.py"
python -u diagnostics/moonshot/eval_run.py --run_dir $R/$T --mesh_dir $D

echo "[$(date +%H:%M:%S)] $T done. Next: add ('Oracle_GT_partition', 'synth_clean', None) to"
echo "  diagnostics/correspondence_accuracy/run_audit.py's RUNS list and re-run it."
