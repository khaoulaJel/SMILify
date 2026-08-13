#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_gwn_calib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=gwn_calibration_probe_scratch/slurm_%j.log
#SBATCH --error=gwn_calibration_probe_scratch/slurm_%j.log
#
# Weight-calibration probe for w_penetration_gwn (fitter_3d/gwn_weight_calibration_probe.py).
# The first real GWN-penetration run (job 15520880) found penetration_gwn at 70.5% of total
# weighted loss with w_penetration_gwn=0.02/0.05 (copied by analogy from the unrelated
# proximity loss's weight), causing catastrophic mesh distortion. This is a cheap, no-full-
# training sweep over candidate weights (0.0005, 0.001, 0.005) to find a value whose weighted-
# contribution fraction is single-digit % (comparable to edge/normal), not dominating chamfer.
#
# A tiny local CPU smoke test (1 candidate, 2 iterations) already confirmed the script runs
# correctly end-to-end; this submits the real 3-candidate x 20-iteration sweep on GPU.
#
# Submit FROM THE REPO ROOT:
#     mkdir -p gwn_calibration_probe_scratch
#     sbatch scripts/penetration_joint_study/submit_gwn_weight_calibration_probe.sbatch
# Monitor:
#     squeue --me
#     tail -f gwn_calibration_probe_scratch/slurm_<jobid>.log
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
mkdir -p gwn_calibration_probe_scratch

python -m fitter_3d.gwn_weight_calibration_probe

echo "Done -- $(date)"
