#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=X2_measure
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/anterior_mechanism/sbatch_logs/X2_measure_%j.log
#SBATCH --error=diagnostics/anterior_mechanism/sbatch_logs/X2_measure_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# Measurement-only rerun: arm C's Stage_3_deform_fine.npz already exists on disk from job
# 3865868 (killed only because the CPU-device chamfer cost computation was too slow for the
# 30-min budget; the fresh arm-C TRAINING itself completed and its outputs are intact -- see
# diagnostics/moonshot/runs/X2_C_allometric/Stage_3_deform_fine.npz, dated before the kill).
# x2_measurements.py fixed to run the chamfer cost condition on GPU instead of the CPU default.
echo "[$(date +%H:%M:%S)] measurements only (arm C training reused from job 3865868)"
python -u diagnostics/anterior_mechanism/x2_measurements.py
echo "[$(date +%H:%M:%S)] X2 measurements done"
