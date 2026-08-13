#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_pen_all_gentle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=penetration_joint_study_results/slurm_gentle_%j.log
#SBATCH --error=penetration_joint_study_results/slurm_gentle_%j.log
#
# Penetration x unfrozen-pose joint experiment -- GENTLE-WEIGHT arm
# (w_penetration=0.02/0.05, scheme:'all' in the deform stages, ramp actively
# trained on). Runs on the same bench50_clean specimen set as the baseline
# arm (submit_all_baseline.sbatch), so the two are directly comparable. See
# fitter_3d/ants_cfg_all_penetration_gentle.yaml for the full config.
#
# Submit FROM THE REPO ROOT:
#     mkdir -p penetration_joint_study_results
#     sbatch scripts/penetration_joint_study/submit_all_gentle.sbatch
# Monitor:
#     squeue --me
#     tail -f penetration_joint_study_results/slurm_gentle_<jobid>.log
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

# ---- avoid the OpenBLAS/pthread_create thread-storm seen on the login node --
# See submit_all_baseline.sbatch for the full explanation. Same fix here so
# both arms of the ablation run under identical conditions.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p penetration_joint_study_results

python -m fitter_3d.optimise --yaml_src fitter_3d/ants_cfg_all_penetration_gentle.yaml

echo "Done -- $(date)"
