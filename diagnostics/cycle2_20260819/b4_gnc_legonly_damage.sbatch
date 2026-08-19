#!/bin/bash
#SBATCH --job-name=b4_gnc_damage
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/cycle2_20260819/logs/b4_%A_%a.out

# B4 (task list): gnc_legonly on damage conditions (drop30/drop60). Gated on B1 (replication,
# PASS) and B2 (pose sweep, GNC benefit confirmed real and growing with severity, though the
# specific cross-leg-confusion mechanism was NOT confirmed -- see
# diagnostics/cycle2_20260819/out/b2_pose_sweep_analysis.json). Per the task list: "the eventual
# production target is real scans with missing geometry ... a result that only holds on clean
# data is not yet a production candidate regardless of how strong it looks there."
#
# Reuses EXISTING hierarchical placements for drop30/drop60 (zero-init, same recipe as every
# other baseline run) -- no new hierarchical compute.

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
RUN_NAME="SYN_clean_${LABEL}_gnclegonly_w5"
OUT_DIR="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/cycle2_20260819/logs
mkdir -p "$OUT_DIR" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL hier_dir=$HIER_DIR ==="
test -f "$HIER_DIR/H2_joint.npz" || { echo "MISSING $HIER_DIR/H2_joint.npz"; exit 1; }
nvidia-smi || true

python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/cycle2_20260819/cfg/gnc_legonly.yaml \
  --init_from "$HIER_DIR/H2_joint.npz" \
  --results_dir "$OUT_DIR" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
