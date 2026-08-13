#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=t11_dino
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=diagnostics/khaoula_v2/runs/t11_slurm_%j.log
#SBATCH --error=diagnostics/khaoula_v2/runs/t11_slurm_%j.log
set -eo pipefail

# T1.1 -- the actual test: does DINO recover within-part correspondence better than HKS
# (12.91% median err) and ideally better than the fitted pipeline (4.32%)? Fork of
# within_part_signal.py, DINO in place of HKS, same protocol/controls/gate style.
source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export HF_HOME=/p/scratch/cias-7/jellal1/hf_cache
export HF_HUB_OFFLINE=1

python -u diagnostics/khaoula_v2/within_part_signal_dino.py --run T04_baseline
echo "[$(date +%H:%M:%S)] T1.1 within-part DINO check complete"
