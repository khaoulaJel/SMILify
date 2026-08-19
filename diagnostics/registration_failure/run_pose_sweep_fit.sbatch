#!/bin/bash
#SBATCH --job-name=posesweep
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/registration_failure/logs/posesweep_%A_%a.out

# Task 6 D1: pose sweep. SAME recipe as Task 3's SYN_clean_w5 (D1_low.yaml, full budget),
# changing only --mesh_dir across pose_scale in {0.05, 0.10, 0.15, 0.20, 0.35}. pose_scale=0.0
# and 0.25 already fit in Task 3 (SYN_clean_pose0_w5, SYN_clean_w5) and are reused unchanged.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

LABELS=(pose05 pose10 pose15 pose20 pose35)
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
CORPUS="synth_clean_${LABEL}"

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
RUN_NAME="SYN_clean_${LABEL}_w5"
HIER_OUT="diagnostics/registration_failure/runs/hier_${RUN_NAME}"
MOONSHOT_OUT="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/registration_failure/logs
mkdir -p "$HIER_OUT" "$MOONSHOT_OUT" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS ==="
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
