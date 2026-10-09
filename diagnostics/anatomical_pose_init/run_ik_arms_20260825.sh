#!/usr/bin/env bash
# Runs the two IK-based init arms (tip-only, tip+waypoint) through the IDENTICAL D1 recipe used
# for every other arm in this series (ACI_cheap/ACI_learned/ACI_D_random etc., synth_clean,
# --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0, D1_low.yaml), so results join the
# same comparison table without a code-version confound. See generate_ik_init.py for how the
# init files were built and why the raw pose-error signal alone (28.4/30.1 deg, worse than
# zero-init's 23.09) is not decisive without an actual D1 run, given this project's own D/E/F
# finding that error STRUCTURE, not magnitude, determines optimizer basin quality.
set -eo pipefail

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
INIT_DIR=diagnostics/anatomical_pose_init/out_ik_20260825

for pair in "ik_tip_init.npz:IK_tip_only" "ik_tip_waypoint_init.npz:IK_tip_waypoint"; do
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

echo "[$(date +%H:%M:%S)] both IK arms fit+eval'd. Next: add IK_tip_only/IK_tip_waypoint to"
echo "  diagnostics/correspondence_accuracy/run_audit.py's RUNS list and re-run it, and"
echo "  diagnostics/morphometrics/calib_features.py --runs SYN_clean_pose25_gtinit_w5 IK_tip_only IK_tip_waypoint --primary SYN_clean_pose25_gtinit_w5"
