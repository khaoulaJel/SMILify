#!/bin/bash
#SBATCH --job-name=d1_evidence
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_evidence/logs/slurm_%j.out

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_evidence/mesh_set"
OUT=diagnostics/d1_evidence/runs
LOGS=diagnostics/d1_evidence/logs
mkdir -p "$OUT" "$LOGS"

echo "=== nvidia-smi ==="
nvidia-smi || true

echo "=== Arm D step 1/2: hierarchical (H0..H3, full budget) ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --results_dir "$OUT/hier" 2>&1 | tee "$LOGS/arm_d_hier.log"

echo "=== Arm D step 2/2: moonshot D1_low handoff from H2_joint.npz (full budget) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from "$OUT/hier/H2_joint.npz" \
  --results_dir "$OUT/d1" \
  --eval 2>&1 | tee "$LOGS/arm_d_moonshot.log"

echo "=== Arm S: stock optimise.py (full budget, ants_cfg.yaml unmodified aside from results_dir) ==="
python -u -m fitter_3d.optimise \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/d1_evidence/ants_cfg_arm_s.yaml 2>&1 | tee "$LOGS/arm_s_stock.log"

echo "=== Arm S: eval (same metric suite as Arm D, via eval_stage_npz.py) ==="
python -u diagnostics/d1_evidence/eval_stage_npz.py \
  --npz "$OUT/stock/Stage_3_deform_fine.npz" \
  --mesh_dir "$MESH_DIR" \
  --out_csv diagnostics/d1_evidence/metrics_stock.csv 2>&1 | tee "$LOGS/arm_s_eval.log"

echo "ALL DONE"
