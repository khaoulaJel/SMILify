#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=Z6_floor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/noise_floor/sbatch_logs/Z6_%j.log
#SBATCH --error=diagnostics/noise_floor/sbatch_logs/Z6_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0

# Z6 -- is the worker residual a floor? Bar in PREREGISTRATION_Z6_noise_floor.md, fixed before
# these fits. 25 species x 2 conspecific workers; D1_PROD unchanged.
C=diagnostics/noise_floor

echo "[$(date +%H:%M:%S)] fit paired50 with D1_PROD as shipped"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $C/paired50 \
  --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml --seed 0 \
  --results_dir diagnostics/moonshot/runs/Z6_paired50

echo "[$(date +%H:%M:%S)] score"
python -u $C/z6_score.py
echo "[$(date +%H:%M:%S)] Z6 done"
