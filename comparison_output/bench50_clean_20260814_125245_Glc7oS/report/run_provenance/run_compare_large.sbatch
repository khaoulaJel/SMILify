#!/usr/bin/env bash
#SBATCH --job-name=compare_pipelines_large
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/comparison_output/bench50_clean_20260814_125245_Glc7oS/report/slurm_logs/%x_%j.out
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/comparison_output/bench50_clean_20260814_125245_Glc7oS/report/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

set -eo pipefail

RUN="/rwthfs/rz/cluster/home/nao48500/SMILify/comparison_output/bench50_clean_20260814_125245_Glc7oS"
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source "/home/nao48500/miniforge3/etc/profile.d/conda.sh"
conda activate pytorch3d
set -u

python -u compare_pipelines.py "$RUN/specimens" \
    --out "$RUN/report" \
    --errors-out "$RUN/report/errors.txt" \
    --workers 4 \
    --large-workers 1
