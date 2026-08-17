#!/bin/bash
#SBATCH --job-name=rest_edge_evidence
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/rest_edge_evidence/logs/slurm_%j.out

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/d1_evidence/mesh_set"
OUT=diagnostics/rest_edge_evidence/runs
LOGS=diagnostics/rest_edge_evidence/logs
mkdir -p "$OUT" "$LOGS"

echo "=== nvidia-smi ==="
nvidia-smi || true

# Hierarchical placement stage does not read edge_mode at all (verified: grep -n edge_mode
# fitter_3d/optimise_hierarchical.py fitter_3d/trainer_hierarchical.py -> no hits), so it is run
# once and its H2_joint.npz init is shared by both arms below -- same as Task 1's Arm D step 1.
echo "=== shared hierarchical placement (H0..H3, full budget) ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --results_dir "$OUT/hier" 2>&1 | tee "$LOGS/hier.log"

echo "=== Arm A (control, edge_mode: shrink, D1_low.yaml unchanged) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from "$OUT/hier/H2_joint.npz" \
  --results_dir "$OUT/arm_a_shrink" \
  --eval 2>&1 | tee "$LOGS/arm_a_shrink.log"

echo "=== Arm B (test, edge_mode: rest, D1_low_rest.yaml) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low_rest.yaml \
  --init_from "$OUT/hier/H2_joint.npz" \
  --results_dir "$OUT/arm_b_rest" \
  --eval 2>&1 | tee "$LOGS/arm_b_rest.log"

echo "ALL DONE"
