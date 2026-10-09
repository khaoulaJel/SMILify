#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=Z2_antlimits
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/anterior_limits/sbatch_logs/Z2_%j.log
#SBATCH --error=diagnostics/anterior_limits/sbatch_logs/Z2_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

# Z2 -- does closing the anterior/body-axis rotation gap move the head-carried deform ratio?
# Bar fixed in PREREGISTRATION_Z2_anterior_limits.md BEFORE this ran.
# Arms differ ONLY in SMILIFY_SMAL_FILE. D1_PROD.yaml is used unchanged in both.
C=diagnostics/anterior_limits
M=diagnostics/moonshot/bench50_clean

echo "[$(date +%H:%M:%S)] arm A -- stock model (99/165 axes constrained)"
SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl \
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir $M --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml --seed 0 \
  --results_dir diagnostics/moonshot/runs/Z2_A_stock

echo "[$(date +%H:%M:%S)] arm B -- anterior limits authored (135/165 axes constrained)"
SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_anterior_limited.pkl \
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir $M --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml --seed 0 \
  --results_dir diagnostics/moonshot/runs/Z2_B_limits

echo "[$(date +%H:%M:%S)] score"
python -u $C/z2_score.py
echo "[$(date +%H:%M:%S)] Z2 done"
