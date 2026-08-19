#!/bin/bash
#SBATCH --job-name=appendage_damage
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/appendage_evidence/logs/damage_%A_%a.out

# Phase 3 damage corpora, SAME recipe as SYN_clean_w5 (D1_low.yaml, full budget), changing only
# --mesh_dir to the drop30/drop60 target sets from make_damage_corpus.py. Ground truth for
# scoring is each corpus's own (UNCHANGED, copied from synth_clean) ground_truth.npz.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

CORPORA=(synth_clean_drop30 synth_clean_drop60)
LABELS=(drop30 drop60)
CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
RUN_NAME="SYN_clean_${LABEL}_w5"
HIER_OUT="diagnostics/appendage_evidence/runs/hier_${RUN_NAME}"
MOONSHOT_OUT="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/appendage_evidence/logs
mkdir -p "$HIER_OUT" "$MOONSHOT_OUT" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL ==="
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

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
