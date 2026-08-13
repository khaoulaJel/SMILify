#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=train_refine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=diagnostics/khaoula_v2/runs/train_refine_slurm_%j.log
#SBATCH --error=diagnostics/khaoula_v2/runs/train_refine_slurm_%j.log
set -eo pipefail

# Refinement-layer training: 8 training specimens' DINO extraction (~1-1.5 min each per
# T1.0/T1.1 timing) + geodesic-guided contrastive/reconstruction training (fast, MLP-scale).
# Gate and named symmetry risk are pre-registered in V2_CHANGELOG.md / REPORT_V2.md BEFORE
# this job runs.
source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export HF_HOME=/p/scratch/cias-7/jellal1/hf_cache
export HF_HUB_OFFLINE=1

python -u diagnostics/khaoula_v2/train_refine_autoencoder.py
echo "[$(date +%H:%M:%S)] refinement-layer training complete"
