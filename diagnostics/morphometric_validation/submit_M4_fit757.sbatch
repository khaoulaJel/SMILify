#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=M4_fit757
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/morphometric_validation/sbatch_logs/M4fit_%j.log
#SBATCH --error=diagnostics/morphometric_validation/sbatch_logs/M4fit_%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
# Unmodified D1_PROD recipe. Paths overridden only; NO recipe edits (M4 prereg do-NOT #5).
CONDA_SH=/home/nao48500/miniforge3/etc/profile.d/conda.sh \
REPO_DIR=$SLURM_SUBMIT_DIR \
W=/rwthfs/rz/cluster/hpcwork/nao48500/worker_ALT \
CHUNK=64 \
SKIP_CLEAN=1 \
TAG_PREFIX=M4_ \
STAGE=diagnostics/morphometric_validation/stage_M4 \
bash diagnostics/morphometrics/run_m1_fit_all.sh
