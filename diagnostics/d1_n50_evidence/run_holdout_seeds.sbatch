#!/bin/bash
#SBATCH --job-name=d1_holdout
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_n50_evidence/logs/slurm_holdout_%A_%a.out

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_n50_evidence/all_ants_clean_holdout"

SEED=$SLURM_ARRAY_TASK_ID
OUT="diagnostics/d1_n50_evidence/runs_holdout/seed${SEED}"
LOGS=diagnostics/d1_n50_evidence/logs
mkdir -p "$OUT" "$LOGS"

echo "=== holdout seed $SEED: ALL_ANTS_CLEAN (80 specimens), D1_low_scalecap.yaml ==="
nvidia-smi || true

echo "=== hierarchical placement, seed=$SEED (full budget) ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --seed $SEED \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --results_dir "$OUT/hier" 2>&1 | tee "$LOGS/holdout_seed${SEED}_hier.log"

echo "=== moonshot D1_low_scalecap handoff, seed=$SEED (full budget) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low_scalecap.yaml \
  --init_from "$OUT/hier/H2_joint.npz" \
  --results_dir "$OUT/d1" \
  --seed $SEED \
  --eval 2>&1 | tee "$LOGS/holdout_seed${SEED}_moonshot.log"

echo "=== holdout seed $SEED DONE ==="
