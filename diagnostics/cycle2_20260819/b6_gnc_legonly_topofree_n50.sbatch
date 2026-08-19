#!/bin/bash
#SBATCH --job-name=b6_topofree_n50
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/cycle2_20260819/logs/b6n50_%A_%a.out

# B6 clean-data control (2 seeds): confirms robust_leg_only_topofree does NOT regress vs B1's
# robust_leg_only (gnc_legonly) on TOPOLOGY-PRESERVING data, where the two should be nearly
# equivalent (topofree's NN-based target-side label transfer approximates, rather than exactly
# reproduces, the exact face-group target split -- see robust_chamfer_leg_split_topofree
# docstring for why they can differ: label noise near the leg/body boundary, proportional to
# registration quality, vs the old version's exact ground-truth face labels). If this control
# regresses substantially, the topofree mechanism itself is suspect regardless of the damage
# result.
#
# Reuses EXISTING hierarchical placements (hier_N50_gnc_legonly_seed{0,1}, split_flag identical
# to baseline/gnc_legonly -- no new hierarchical compute).

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

SEEDS=(0 1)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/synth_clean_n50"
RUN_NAME="N50_gnc_legonly_topofree_seed${SEED}"
HIER_DIR="diagnostics/cycle2_20260819/runs/hier_N50_gnc_legonly_seed${SEED}"
OUT_DIR="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/cycle2_20260819/logs
mkdir -p "$OUT_DIR" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: seed=$SEED hier_dir=$HIER_DIR ==="
test -f "$HIER_DIR/H2_joint.npz" || { echo "MISSING $HIER_DIR/H2_joint.npz"; exit 1; }
nvidia-smi || true

python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/cycle2_20260819/cfg/gnc_legonly_topofree.yaml \
  --init_from "$HIER_DIR/H2_joint.npz" \
  --results_dir "$OUT_DIR" \
  --seed "$SEED" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
