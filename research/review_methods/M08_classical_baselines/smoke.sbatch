#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --job-name=RM_M08_smk
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=research/review_methods/M08_classical_baselines/sbatch_logs/smoke_%j.log
#SBATCH --error=research/review_methods/M08_classical_baselines/sbatch_logs/smoke_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" OMP_NUM_THREADS=16
python -u research/review_methods/M08_classical_baselines/m08_register.py
