#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_pen_offset_gentle_pairscoped_seed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=fit3d_results_seeded/slurm_pairscoped_%j.log
#SBATCH --error=fit3d_results_seeded/slurm_pairscoped_%j.log
#
# One-seed replication check for the pair-scoped gentle+offset arm (see
# submit_all_offset_gentle_pairscoped.sbatch / ants_cfg_all_offset_gentle_pairscoped.yaml).
# Submit with SEED set:
#     mkdir -p fit3d_results_seeded
#     SEED=1 sbatch --export=ALL,SEED scripts/penetration_joint_study/submit_offset_gentle_pairscoped_seed.sbatch
#
# NOTE: intentionally no set -u -- conda's activate hooks reference unset vars
# (e.g. MKL_INTERFACE_LAYER) and would abort under set -u.
set -eo pipefail

: "${SEED:?Set SEED=<int> via --export}"

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
REPO_DIR="${REPO_DIR:-.}"
CONDA_SH="${CONDA_SH:-/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-pytorch3d}"
RESULTS_DIR="fit3d_results_seeded/offset_gentle_pairscoped_seed${SEED}"

echo "Job $SLURM_JOB_ID on $(hostname) -- $(date) -- SEED=$SEED -> $RESULTS_DIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SMIL_DISABLE_PLOTTING=1

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p "$RESULTS_DIR"

python -m fitter_3d.optimise --yaml_src fitter_3d/ants_cfg_all_offset_gentle_pairscoped.yaml \
    --results_dir "$RESULTS_DIR" --seed "$SEED" --require_seed

echo "Done -- $(date)"
