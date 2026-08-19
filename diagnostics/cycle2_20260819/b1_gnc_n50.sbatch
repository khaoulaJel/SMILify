#!/bin/bash
#SBATCH --job-name=b1_gnc_n50
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/cycle2_20260819/logs/b1_%A_%a.out

# Cycle2 B1 (task list "Overnight/Next-Cycle Task List -- Rewritten with SOTA-Grounded
# Additions"): GNC replication at scale. N=50 (diagnostics/moonshot/synth_clean_n50, a FRESH
# independent synthetic draw at pose_scale=0.25, seed=777, generated this cycle -- NOT the same
# 12 specimens as SYN_clean_w5/GNC_medium/GNC_slow from the previous overnight cycle), 2 seeds.
# 5 arms x 2 seeds = 10 array tasks:
#   arm 0: baseline (D1_low.yaml, no split_distal)
#   arm 1: GNC_medium (overnight_20260818/cfg/gnc_medium.yaml)
#   arm 2: GNC_slow   (overnight_20260818/cfg/gnc_slow.yaml)
#   arm 3: GNC_medium + split_distal (hierarchical placement gets --split_distal; same handoff
#          yaml as arm 1 -- tests whether GNC and split_distal stack or duplicate each other,
#          per the task list's explicit question)
#   arm 4: GNC_legonly (cfg/gnc_legonly.yaml, NEW robust_leg_only code this cycle -- anneals the
#          GM kernel only on leg residuals, non-leg gets plain L2)
# One variable at a time within each arm; baseline included in the SAME batch so all comparisons
# are apples-to-apples on the same N=50 corpus/seeds (the previous cycle's N=12 baseline is NOT
# reused here since specimen sets differ).

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

ARMS=(baseline gnc_medium gnc_slow gnc_medium_splitdistal gnc_legonly)
SEEDS=(0 1)
N_SEEDS=2
ARM_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
ARM=${ARMS[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/synth_clean_n50"
RUN_NAME="N50_${ARM}_seed${SEED}"
HIER_OUT="diagnostics/cycle2_20260819/runs/hier_${RUN_NAME}"
MOONSHOT_OUT="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/cycle2_20260819/logs
mkdir -p "$HIER_OUT" "$MOONSHOT_OUT" "$LOGS"

case "$ARM" in
  baseline)                SPLIT_FLAG=""              ; YAML=diagnostics/moonshot/cfg/D1_low.yaml ;;
  gnc_medium)               SPLIT_FLAG=""              ; YAML=diagnostics/overnight_20260818/cfg/gnc_medium.yaml ;;
  gnc_slow)                 SPLIT_FLAG=""              ; YAML=diagnostics/overnight_20260818/cfg/gnc_slow.yaml ;;
  gnc_medium_splitdistal)   SPLIT_FLAG="--split_distal"; YAML=diagnostics/overnight_20260818/cfg/gnc_medium.yaml ;;
  gnc_legonly)               SPLIT_FLAG=""              ; YAML=diagnostics/cycle2_20260819/cfg/gnc_legonly.yaml ;;
esac

echo "=== task $SLURM_ARRAY_TASK_ID: arm=$ARM seed=$SEED split_flag='$SPLIT_FLAG' yaml=$YAML ==="
nvidia-smi || true

echo "=== hierarchical placement ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --seed "$SEED" $SPLIT_FLAG \
  --results_dir "$HIER_OUT" 2>&1 | tee "$LOGS/${RUN_NAME}_hier.log"

echo "=== moonshot handoff, yaml=$YAML ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src "$YAML" \
  --init_from "$HIER_OUT/H2_joint.npz" \
  --results_dir "$MOONSHOT_OUT" \
  --seed "$SEED" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}_moonshot.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
