#!/usr/bin/env bash
# Runs the two new multi-start coherent candidates (cluster-axis, tip-direction) through the
# IDENTICAL D1 recipe used for every other arm in this series (PCA_coherent, IK_tip_only/
# _waypoint, BASIN_*), synth_clean, --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0,
# D1_low.yaml -- so results join the same comparison table without a code-version confound.
set -eo pipefail

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
INIT_DIR=diagnostics/anatomical_pose_init/out_ik_20260825

for pair in "cluster_coherent_init.npz:Cluster_coherent" "tipdir_coherent_init.npz:Tipdir_coherent"; do
  init_file="${pair%%:*}"
  t="${pair##*:}"
  INIT=$INIT_DIR/$init_file

  echo "[$(date +%H:%M:%S)] $t: hierarchical init stage"
  python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
    --init_joint_rot_from $INIT \
    --results_dir $R/${t}_hier

  echo "[$(date +%H:%M:%S)] $t: D1 fit"
  python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
    --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t

  echo "[$(date +%H:%M:%S)] $t: eval_run.py"
  python -u diagnostics/moonshot/eval_run.py --run_dir $R/$t --mesh_dir $D

  echo "[$(date +%H:%M:%S)] $t done"
done

echo "[$(date +%H:%M:%S)] both arms fit+eval'd. Next: add Cluster_coherent/Tipdir_coherent to"
echo "  diagnostics/correspondence_accuracy/run_audit.py's RUNS list and re-run it."
