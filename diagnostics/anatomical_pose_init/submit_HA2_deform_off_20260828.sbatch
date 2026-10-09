#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=HA2_nodef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/sbatch_logs/HA2_nodef_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/sbatch_logs/HA2_nodef_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_power48
G=diagnostics/anatomical_pose_init/out_C3_cse_head_20260826

# H_A2 -- bar in PREREGISTRATION_HA2_deform_off_20260828.md. Same as D1_with_cse but scheme:pose,
# removing deform_verts. Tests whether ANY of H_A's 127% recovery is anatomical rather than bought
# with free-form per-vertex deformation.
echo "[$(date +%H:%M:%S)] D1_with_cse_nodeform"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/hb/D1_no_deform.yaml \
  --init_from $R/C13_uniform_hier/H2_joint.npz \
  --cse_correspondence_from $G/cse_p48_all.npz \
  --results_dir $R/D1_with_cse_nodeform
python -u diagnostics/moonshot/eval_run.py --run_dir $R/D1_with_cse_nodeform --mesh_dir $D
echo "[$(date +%H:%M:%S)] H_A2 done"
