#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=V8_p2s
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=diagnostics/groundtruth/sbatch_logs/V8_%j.log
#SBATCH --error=diagnostics/groundtruth/sbatch_logs/V8_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export V8_ITERS=800
python -u diagnostics/groundtruth/v8_point_to_surface.py
