#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_pen_all_offset_gentle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=penetration_joint_study_results/slurm_offset_gentle_%j.log
#SBATCH --error=penetration_joint_study_results/slurm_offset_gentle_%j.log
#
# TASK 2 (structural-constraint arm) -- GENTLE+OFFSET (gentle w_penetration
# 0.02/0.05, scheme:'all', w_offset 5.0/2.0). Runs on the same bench50_clean
# set as TASK 1, directly comparable to submit_all_offset_baseline.sbatch. See
# fitter_3d/ants_cfg_all_offset_gentle.yaml for the full config and rationale.
#
# Submit FROM THE REPO ROOT:
#     mkdir -p penetration_joint_study_results
#     sbatch scripts/penetration_joint_study/submit_all_offset_gentle.sbatch
# Monitor:
#     squeue --me
#     tail -f penetration_joint_study_results/slurm_offset_gentle_<jobid>.log
# Cancel:
#     scancel <jobid>
#
# NOTE: intentionally no set -u -- conda's activate hooks reference unset vars
# (e.g. MKL_INTERFACE_LAYER) and would abort under set -u.
set -eo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
REPO_DIR="${REPO_DIR:-.}"
CONDA_SH="${CONDA_SH:-/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-pytorch3d}"

echo "Job $SLURM_JOB_ID on $(hostname) -- $(date)"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SMIL_DISABLE_PLOTTING=1

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p penetration_joint_study_results

python -m fitter_3d.optimise --yaml_src fitter_3d/ants_cfg_all_offset_gentle.yaml \
    --results_dir fit3d_results_all_offset_gentle

echo "Done -- $(date)"
