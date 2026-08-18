#!/bin/bash
#SBATCH --job-name=d1_holdout_smoke
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_n50_evidence/logs/smoke_holdout_%j.out

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_n50_evidence/all_ants_clean_holdout"
OUT=/tmp/d1_holdout_smoke_out
LOGS=diagnostics/d1_n50_evidence/logs
mkdir -p "$OUT" "$LOGS"

nvidia-smi || true

echo "=== smoke hier ALL_ANTS_CLEAN holdout (80 specimens, tiny its) ==="
time python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --body_its 20 --leg_its 20 --joint_its 20 --deform_its 10 \
  --results_dir "$OUT/hier" 2>&1 | tee "$LOGS/smoke_holdout_hier.log"

echo "=== smoke moonshot D1_low_scalecap on holdout (check it starts, 60s) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low_scalecap.yaml \
  --init_from "$OUT/hier/H2_joint.npz" \
  --results_dir "$OUT/moonshot" \
  --seed 0 \
  --eval 2>&1 | tee "$LOGS/smoke_holdout_moonshot_START.log" &
MS_PID=$!
sleep 60
kill $MS_PID 2>/dev/null || true

echo "SMOKE DONE"
