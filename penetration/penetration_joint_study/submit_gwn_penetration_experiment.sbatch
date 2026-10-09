#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_gwn_pen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=fit3d_results_gwn_gentle/slurm_%j.log
#SBATCH --error=fit3d_results_gwn_gentle/slurm_%j.log
#
# GWN-based penetration loss test (fitter_3d/gwn_penetration_loss.py), the
# concrete next step after Step 6's confirmed proximity-test-accuracy
# mechanism (see scripts/penetration_joint_study/FINDINGS.md). Single new
# arm on 3 specimens, scoped to the (gaster, legs) pair only. Correctness
# and differentiability already verified offline (torch vs. numpy winding
# number agree to 5.55e-16; gradients finite and nonzero) before this job.
#
# Submit FROM THE REPO ROOT:
#     mkdir -p fit3d_results_gwn_gentle
#     sbatch scripts/penetration_joint_study/submit_gwn_penetration_experiment.sbatch
# Monitor:
#     squeue --me
#     tail -f fit3d_results_gwn_gentle/slurm_<jobid>.log
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

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p fit3d_results_gwn_gentle

python -m fitter_3d.run_gwn_penetration_experiment

echo "Done -- $(date)"
