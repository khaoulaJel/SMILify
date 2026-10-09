#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=HA4_bp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/sbatch_logs/HA4_bp_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/sbatch_logs/HA4_bp_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_power48
G=diagnostics/anatomical_pose_init/out_C3_cse_head_20260826

# H_A4 -- closing arm of the coxa thread. Bar in PREREGISTRATION_HA4_betaprior_20260828.md.
# PREREGISTRATION_HA3_coxa_placement_20260828.md, written before this ran.
# Exists because H_A2's coxa-placement number was measured POST-HOC; re-scoring the same run
# against a check designed after seeing it would be the same evidence relabelled, not confirmation.
echo "[$(date +%H:%M:%S)] D1_cse_nodeform_betaprior"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/hb/D1_nodeform_betaprior.yaml \
  --init_from $R/C13_uniform_hier/H2_joint.npz \
  --cse_correspondence_from $G/cse_p48_all.npz \
  --seed 1 \
  --results_dir $R/D1_cse_nodeform_betaprior
python -u diagnostics/moonshot/eval_run.py --run_dir $R/D1_cse_nodeform_betaprior --mesh_dir $D
echo "[$(date +%H:%M:%S)] H_A3 done"
