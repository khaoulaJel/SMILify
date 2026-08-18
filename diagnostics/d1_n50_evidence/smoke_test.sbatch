#!/bin/bash
#SBATCH --job-name=d1_n50_smoke
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_n50_evidence/logs/smoke_%j.out

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_n50_evidence/mesh_set"
OUT=/tmp/d1_n50_smoke_out
LOGS=diagnostics/d1_n50_evidence/logs
mkdir -p "$OUT" "$LOGS"

nvidia-smi || true

echo "=== smoke hier N=50 (tiny its, timed, watch peak memory) ==="
time python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --body_its 20 --leg_its 20 --joint_its 20 --deform_its 10 \
  --results_dir "$OUT/hier" 2>&1 | tee "$LOGS/smoke_hier.log" &
HIER_PID=$!
( while kill -0 $HIER_PID 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$LOGS/smoke_mem.log"; sleep 5; done ) &
MEM_PID=$!
wait $HIER_PID
kill $MEM_PID 2>/dev/null || true

echo "=== peak GPU mem during hier smoke ==="
sort -t' ' -k1 -n "$LOGS/smoke_mem.log" | tail -1

echo "=== smoke moonshot D1_low N=50 (check it starts, 60s) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low_scalecap.yaml \
  --init_from "$OUT/hier/H2_joint.npz" \
  --results_dir "$OUT/moonshot" \
  --seed 0 \
  --eval 2>&1 | tee "$LOGS/smoke_moonshot_START.log" &
MS_PID=$!
sleep 60
kill $MS_PID 2>/dev/null || true

echo "SMOKE DONE"
