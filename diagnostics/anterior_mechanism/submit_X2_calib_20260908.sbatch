#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=X2_calib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/anterior_mechanism/sbatch_logs/X2_calib_%j.log
#SBATCH --error=diagnostics/anterior_mechanism/sbatch_logs/X2_calib_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

echo "[$(date +%H:%M:%S)] X2 Task 1 -- w_allo calibration (checkpointed search through Stage_2)"
python -u diagnostics/anterior_mechanism/x2_calibrate_wallo.py --max_its 1000 --ckpt_every 25 --seed 0
echo "[$(date +%H:%M:%S)] calibration done"
