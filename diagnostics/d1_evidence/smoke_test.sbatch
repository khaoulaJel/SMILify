#!/bin/bash
#SBATCH --job-name=d1_smoke
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_evidence/logs/smoke_%j.out

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_evidence/mesh_set"
OUT=/tmp/d1_smoke_out
LOGS=diagnostics/d1_evidence/logs
mkdir -p "$OUT" "$LOGS"

nvidia-smi || true

echo "=== smoke Arm D hier (real 10-mesh set, tiny its, timed) ==="
time python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --body_its 20 --leg_its 20 --joint_its 20 --deform_its 10 \
  --results_dir "$OUT/hier" 2>&1 | tee "$LOGS/smoke_hier.log"

echo "=== smoke Arm D moonshot (tiny its via yaml override not available -- use full D1_low but check it runs) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from "$OUT/hier/H2_joint.npz" \
  --results_dir "$OUT/d1" \
  --eval 2>&1 | tee "$LOGS/smoke_moonshot_START.log" &
MOONSHOT_PID=$!
sleep 90
kill $MOONSHOT_PID 2>/dev/null || true
echo "(moonshot killed after 90s -- just checking it starts cleanly and logs a few its)"

echo "=== smoke Arm S stock (real 10-mesh set, full ants_cfg.yaml -- just time it briefly) ==="
timeout 90 python -u -m fitter_3d.optimise \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/d1_evidence/ants_cfg_arm_s.yaml 2>&1 | tee "$LOGS/smoke_stock_START.log" || true

echo "SMOKE DONE"
