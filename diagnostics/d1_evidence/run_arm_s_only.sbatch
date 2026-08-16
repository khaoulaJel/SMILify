#!/bin/bash
#SBATCH --job-name=d1_arm_s
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_evidence/logs/slurm_arm_s_%j.out

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_evidence/mesh_set"
OUT=diagnostics/d1_evidence/runs
LOGS=diagnostics/d1_evidence/logs
mkdir -p "$OUT" "$LOGS"

echo "=== Arm S: stock optimise.py (full budget; config.PLOT_RESULTS temporarily False -- see scorecard.md note) ==="
python -u -m fitter_3d.optimise \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/d1_evidence/ants_cfg_arm_s.yaml 2>&1 | tee "$LOGS/arm_s_stock.log"

echo "=== Arm S: eval (same metric suite as Arm D, via eval_stage_npz.py) ==="
python -u diagnostics/d1_evidence/eval_stage_npz.py \
  --npz "$OUT/stock/Stage_3_deform_fine.npz" \
  --mesh_dir "$MESH_DIR" \
  --out_csv diagnostics/d1_evidence/metrics_stock.csv 2>&1 | tee "$LOGS/arm_s_eval.log"

echo "ARM_S_DONE"
