#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=R3_correct
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=diagnostics/groundtruth/sbatch_logs/R3_%j.log
#SBATCH --error=diagnostics/groundtruth/sbatch_logs/R3_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export R3_ITERS=800
python -u diagnostics/groundtruth/r3_correct_parts.py
