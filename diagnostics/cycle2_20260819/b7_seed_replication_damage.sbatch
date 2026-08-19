#!/bin/bash
#SBATCH --job-name=b7_seedrep_damage
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-7
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/cycle2_20260819/logs/b7_%A_%a.out

# B7 (user follow-up to B6): the drop30/drop60 topofree-vs-gnc_medium comparison was single-seed
# (n=1) -- not distinguishable from seed noise. Adds 2 more seeds (1, 2) x 2 arms
# (gnc_medium, gnc_legonly_topofree) x 2 corpora (drop30, drop60) = 8 tasks, holding EVERYTHING
# else fixed (same corpora, same reused hierarchical placement/init npz, same yaml configs) --
# only --seed changes, which reseeds torch/numpy and therefore the moonshot stage's own
# stochastic target/source point sampling around the SAME fixed starting pose. One variable
# changed, per the master prompt's execution rules.
#
# Decision rule (pre-registered by the user before results were seen): if the delta_R direction
# (topofree matches/beats gnc_medium) holds across the new seeds within noise, damage-data status
# is promoted to match the clean-data language ("bar met", topofree becomes the unqualified
# damage recommendation too). If the direction flips or the gap collapses into noise, gnc_medium
# stays the damage default and topofree stays clean-data-only -- a negative result here is exactly
# what this run is for, not a failure.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

CORPORA=(synth_clean_drop30 synth_clean_drop60)
LABELS=(drop30 drop60)
HIER_DIRS=(diagnostics/appendage_evidence/runs/hier_SYN_clean_drop30_w5 diagnostics/appendage_evidence/runs/hier_SYN_clean_drop60_w5)
ARMS=(gnc_medium gnc_legonly_topofree)
SEEDS=(1 2)

N_ARMS=2
N_SEEDS=2
COND_IDX=$(( SLURM_ARRAY_TASK_ID / (N_ARMS * N_SEEDS) ))
REM=$(( SLURM_ARRAY_TASK_ID % (N_ARMS * N_SEEDS) ))
ARM_IDX=$(( REM / N_SEEDS ))
SEED_IDX=$(( REM % N_SEEDS ))

CORPUS=${CORPORA[$COND_IDX]}
LABEL=${LABELS[$COND_IDX]}
HIER_DIR=${HIER_DIRS[$COND_IDX]}
ARM=${ARMS[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

case "$ARM" in
  gnc_medium)             YAML=diagnostics/overnight_20260818/cfg/gnc_medium.yaml ;;
  gnc_legonly_topofree)   YAML=diagnostics/cycle2_20260819/cfg/gnc_legonly_topofree.yaml ;;
esac

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
case "$ARM" in
  gnc_medium)             RUN_NAME="SYN_clean_${LABEL}_gncmedium_w5_seed${SEED}" ;;
  gnc_legonly_topofree)   RUN_NAME="SYN_clean_${LABEL}_gnclegonlytopofree_w5_seed${SEED}" ;;
esac
OUT_DIR="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/cycle2_20260819/logs
mkdir -p "$OUT_DIR" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL arm=$ARM seed=$SEED yaml=$YAML hier_dir=$HIER_DIR ==="
test -f "$HIER_DIR/H2_joint.npz" || { echo "MISSING $HIER_DIR/H2_joint.npz"; exit 1; }
nvidia-smi || true

python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src "$YAML" \
  --init_from "$HIER_DIR/H2_joint.npz" \
  --results_dir "$OUT_DIR" \
  --seed "$SEED" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
