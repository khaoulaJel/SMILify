#!/bin/bash
#SBATCH --job-name=offset_smoke
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_sweep_evidence/logs/smoke_%j.out

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/synth_clean"
OUT=/tmp/offset_smoke_out
LOGS=diagnostics/offset_sweep_evidence/logs
mkdir -p "$OUT" "$LOGS"

nvidia-smi || true

echo "=== smoke hier (real synth_clean set, tiny its, timed) ==="
time python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --body_its 20 --leg_its 20 --joint_its 20 --deform_its 10 \
  --results_dir "$OUT/hier" 2>&1 | tee "$LOGS/smoke_hier.log"

echo "=== smoke moonshot D1_low (real synth_clean set, check it starts + writes npz) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from "$OUT/hier/H2_joint.npz" \
  --results_dir "$OUT/moonshot" \
  --eval 2>&1 | tee "$LOGS/smoke_moonshot_START.log" &
MS_PID=$!
sleep 90
kill $MS_PID 2>/dev/null || true
echo "(moonshot killed after 90s -- just checking it starts cleanly and logs a few its)"

echo "=== smoke score_synth_roundtrip import/CLI sanity (no runs on disk yet, expect [skip] lines) ==="
python -u diagnostics/moonshot/score_synth_roundtrip.py --pairs SYN_clean_w5:synth_clean 2>&1 | tee "$LOGS/smoke_score.log" || true

echo "SMOKE DONE"
