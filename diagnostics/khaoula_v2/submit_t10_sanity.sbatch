#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=t10_sanity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/khaoula_v2/runs/t10_slurm_%j.log
#SBATCH --error=diagnostics/khaoula_v2/runs/t10_slurm_%j.log
set -eo pipefail

# T1.0 -- cheap Diff3F/DINO sanity check, single template, 8 views, no corpus loop.
# Compute nodes have no internet (confirmed earlier), so HF_HUB_OFFLINE=1 is required --
# models were pre-downloaded to HF_HOME from the login node.
source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export HF_HOME=/p/scratch/cias-7/jellal1/hf_cache
export HF_HUB_OFFLINE=1

python -u diagnostics/khaoula_v2/t10_diff3f_sanity.py
echo "[$(date +%H:%M:%S)] T1.0 sanity check complete"
