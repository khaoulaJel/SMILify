#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=E1E2_frz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/sbatch_logs/E1E2_frz_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/sbatch_logs/E1E2_frz_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_power48

# E1/E2 -- bar fixed in advance in PREREGISTRATION_E1E2_frozen_pose_20260828.md.
#
# HIER-ONLY, deliberately: optimise_moonshot has no freeze flags and its D1 stage would unfreeze
# pose, so these arms stop at H2_joint and are audited there against hier-only baselines
# (C14p_gtinit_nocse_hier_H2). --deform_its 0 keeps free-form per-vertex offsets out, which would
# otherwise satisfy any placement and make the test vacuous.
#
# E1: pose HARD-FROZEN at ground truth, shape free. Reachability, NOT representability -- P48
#     targets are generated from the model and E0_identity measures a perfect fit at co=0.0908,
#     so parameters placing the coxa well demonstrably exist.
# E2: E1 plus log_beta_scales hard-frozen at ground truth. Discriminates whether the r=+0.503
#     correlation with coxal error is causal or a compensating symptom.

run_hier () {
  local t=$1; shift
  echo "[$(date +%H:%M:%S)] $t: hierarchical (frozen)"
  python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
    --init_joint_rot_from $D/ground_truth.npz --freeze_joint_rot \
    "$@" --results_dir $R/${t}_hier
}

run_hier E1_frozen_pose
run_hier E2_frozen_pose_gtscale \
  --init_log_beta_scales_from $D/ground_truth.npz --freeze_log_beta_scales

echo "[$(date +%H:%M:%S)] E1E2 done"
