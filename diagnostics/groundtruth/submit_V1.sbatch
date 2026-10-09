#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=V1_realscan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/groundtruth/sbatch_logs/V1_%j.log
#SBATCH --error=diagnostics/groundtruth/sbatch_logs/V1_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
python -u diagnostics/groundtruth/v1_real_scan_validation.py
