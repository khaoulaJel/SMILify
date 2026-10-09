#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_pen_all_offset_baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=penetration_joint_study_results/slurm_offset_baseline_%j.log
#SBATCH --error=penetration_joint_study_results/slurm_offset_baseline_%j.log
#
# TASK 2 (structural-constraint arm) -- BASELINE+OFFSET (w_penetration=0,
# scheme:'all', w_offset 5.0/2.0 -- the D1 recipe setting, newly ported into
# this trainer for this study). Runs on the same bench50_clean set as TASK 1.
# See fitter_3d/ants_cfg_all_offset_baseline.yaml for the full config and
# rationale.
#
# Submit FROM THE REPO ROOT:
#     mkdir -p penetration_joint_study_results
#     sbatch scripts/penetration_joint_study/submit_all_offset_baseline.sbatch
# Monitor:
#     squeue --me
#     tail -f penetration_joint_study_results/slurm_offset_baseline_<jobid>.log
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

python -m fitter_3d.optimise --yaml_src fitter_3d/ants_cfg_all_offset_baseline.yaml \
    --results_dir fit3d_results_all_offset_baseline

echo "Done -- $(date)"
