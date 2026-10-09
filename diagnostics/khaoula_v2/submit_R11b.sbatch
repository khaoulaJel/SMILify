#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=R11b_rot
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/khaoula_v2/sbatch_logs/R11b_%j.log
#SBATCH --error=diagnostics/khaoula_v2/sbatch_logs/R11b_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export TORCH_HOME=/hpcwork/nao48500/torch_hub
export HF_HUB_OFFLINE=1
python -u diagnostics/khaoula_v2/r11b_rotation_sweep.py
