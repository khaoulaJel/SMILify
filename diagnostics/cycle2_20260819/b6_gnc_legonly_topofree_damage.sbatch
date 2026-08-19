#!/bin/bash
#SBATCH --job-name=b6_topofree_damage
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/cycle2_20260819/logs/b6_%A_%a.out

# B6 (follow-up to B4's FAILED gnc_legonly-on-damage run, per user request): tests whether a
# TOPOLOGY-ROBUST version of leg-scoped GNC (robust_leg_only_topofree -- source-side face split
# unchanged, target-side split now done via NN label transfer from the fitted source mesh
# instead of assuming the target shares the template's face indexing, see
# robust_chamfer_leg_split_topofree in fitter_3d/trainer_moonshot.py) can run on drop30/drop60
# and whether it beats the B4 substitute (gnc_medium, +0.176/+0.198 R). Smoke-tested on CPU on
# both drop30 and synth_clean_n50 before submission (no crash, both directions).
#
# Reuses EXISTING hierarchical placements for drop30/drop60 (same as B4) -- no new hierarchical
# compute.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

CORPORA=(synth_clean_drop30 synth_clean_drop60)
LABELS=(drop30 drop60)
HIER_DIRS=(diagnostics/appendage_evidence/runs/hier_SYN_clean_drop30_w5 diagnostics/appendage_evidence/runs/hier_SYN_clean_drop60_w5)

CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
HIER_DIR=${HIER_DIRS[$SLURM_ARRAY_TASK_ID]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
RUN_NAME="SYN_clean_${LABEL}_gnclegonlytopofree_w5"
OUT_DIR="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/cycle2_20260819/logs
mkdir -p "$OUT_DIR" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL hier_dir=$HIER_DIR ==="
test -f "$HIER_DIR/H2_joint.npz" || { echo "MISSING $HIER_DIR/H2_joint.npz"; exit 1; }
nvidia-smi || true

python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/cycle2_20260819/cfg/gnc_legonly_topofree.yaml \
  --init_from "$HIER_DIR/H2_joint.npz" \
  --results_dir "$OUT_DIR" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
