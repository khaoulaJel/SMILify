#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_gwn_pen_cal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=fit3d_results_gwn_gentle_calibrated/slurm_%j.log
#SBATCH --error=fit3d_results_gwn_gentle_calibrated/slurm_%j.log
#
# Re-run of the GWN-based penetration loss test (fitter_3d/gwn_penetration_loss.py)
# with a CALIBRATED w_penetration_gwn (0.001 in both Stage_2/Stage_3, ramp added),
# after the first attempt (job 15520880, see fit3d_results_gwn_gentle/) was found to
# have penetration_gwn at 70.5% of total weighted loss, causing catastrophic mesh
# distortion. Calibration via fitter_3d/gwn_weight_calibration_probe.py (job
# 15521221) -- see fitter_3d/run_gwn_penetration_experiment.py's STAGE2/3_LOSS_WEIGHTS
# comment for the full reasoning.
#
# Deliberately writes to a NEW output directory rather than overwriting
# fit3d_results_gwn_gentle/, so the original failed run (including the mesh renders
# that were personally inspected) stays on disk as a diagnostic artifact.
#
# Submit FROM THE REPO ROOT:
#     mkdir -p fit3d_results_gwn_gentle_calibrated
#     sbatch scripts/penetration_joint_study/submit_gwn_penetration_experiment_calibrated.sbatch
# Monitor:
#     squeue --me
#     tail -f fit3d_results_gwn_gentle_calibrated/slurm_<jobid>.log
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

export GWN_PEN_OUT_ROOT=fit3d_results_gwn_gentle_calibrated

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p fit3d_results_gwn_gentle_calibrated

python -m fitter_3d.run_gwn_penetration_experiment

echo "Done -- $(date)"
