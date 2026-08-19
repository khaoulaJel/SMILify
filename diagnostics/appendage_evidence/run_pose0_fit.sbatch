#!/bin/bash
#SBATCH --job-name=appendage_pose0
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/appendage_evidence/logs/pose0_%j.out

# Phase 1b of Task 3 (appendage measurement reliability). Reproduces the EXACT recipe that
# produced SYN_clean_w5 (see diagnostics/offset_sweep_evidence/run_offset_sweep_array.sbatch,
# array index 1: corpus=clean label=w5 yaml=D1_low.yaml), changing ONLY the input mesh_dir --
# synth_clean_pose0 (pose_scale=0.0) instead of synth_clean (pose_scale=0.25). Every optimiser
# flag below (--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0, full budget, no
# --body_its/--leg_its/etc overrides, same D1_low.yaml handoff) is copied verbatim from that
# script, not re-derived, so this isolates pose_scale and nothing else.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/synth_clean_pose0"
RUN_NAME="SYN_clean_pose0_w5"
HIER_OUT="diagnostics/appendage_evidence/runs/hier_${RUN_NAME}"
MOONSHOT_OUT="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/appendage_evidence/logs
mkdir -p "$HIER_OUT" "$MOONSHOT_OUT" "$LOGS"

nvidia-smi || true

echo "=== hierarchical placement (full budget) ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --results_dir "$HIER_OUT" 2>&1 | tee "$LOGS/${RUN_NAME}_hier.log"

echo "=== moonshot handoff, yaml=D1_low.yaml (full budget) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from "$HIER_OUT/H2_joint.npz" \
  --results_dir "$MOONSHOT_OUT" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}_moonshot.log"

echo "=== DONE ==="
