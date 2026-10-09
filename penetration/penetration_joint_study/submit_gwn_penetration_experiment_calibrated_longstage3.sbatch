#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_gwn_pen_cal_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=fit3d_results_gwn_gentle_calibrated_longstage3/slurm_%j.log
#SBATCH --error=fit3d_results_gwn_gentle_calibrated_longstage3/slurm_%j.log
#
# EXTENDED-Stage_3 variant of submit_gwn_penetration_experiment_calibrated.sbatch.
# The calibrated re-run (job 15521230, fit3d_results_gwn_gentle_calibrated/) showed
# essentially flat aggregate count on Acromyrmex (1044->1041), but a checkpoint-
# bracketed post-hoc check (Stage_2-end vs Stage_3-end, no new run) found Acromyrmex's
# own GWN loss and B_into_A proximity count both still declining substantially
# (-18.7% / -19.3%) across Stage_3's 1000 iterations, with the full unramped weight
# active throughout -- a rate problem, not a weight-ceiling problem. This run just
# extends Stage_3 to 3x the iterations (3000) at the SAME calibrated weight, no
# redesign, to see whether the decline continues or itself plateaus given more time.
#
# Deliberately writes to a NEW output directory rather than overwriting
# fit3d_results_gwn_gentle_calibrated/, so that run stays on disk for comparison.
#
# Submit FROM THE REPO ROOT:
#     mkdir -p fit3d_results_gwn_gentle_calibrated_longstage3
#     sbatch scripts/penetration_joint_study/submit_gwn_penetration_experiment_calibrated_longstage3.sbatch
# Monitor:
#     squeue --me
#     tail -f fit3d_results_gwn_gentle_calibrated_longstage3/slurm_<jobid>.log
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

# See submit_all_baseline.sbatch for the full explanation.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export GWN_PEN_OUT_ROOT=fit3d_results_gwn_gentle_calibrated_longstage3
export GWN_PEN_STAGE3_NITS=3000

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p fit3d_results_gwn_gentle_calibrated_longstage3

python -m fitter_3d.run_gwn_penetration_experiment

echo "Done -- $(date)"
