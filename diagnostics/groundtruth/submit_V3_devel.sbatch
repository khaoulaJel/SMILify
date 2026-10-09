#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --job-name=V3_ident_cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/groundtruth/sbatch_logs/V3cpu_%j.log
#SBATCH --error=diagnostics/groundtruth/sbatch_logs/V3cpu_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
python -u diagnostics/groundtruth/v3_identifiability.py
