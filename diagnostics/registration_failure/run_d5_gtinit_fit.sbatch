#!/bin/bash
#SBATCH --job-name=d5_gtinit
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/registration_failure/logs/d5_%A_%a.out

# Task 6 D5: init-sensitivity probe. SAME recipe as SYN_clean_w5 (D1_low.yaml, full budget),
# adding ONLY --init_joint_rot_from <corpus>/ground_truth.npz (diagnostic-only override added to
# fitter_3d/trainer.py + optimise_hierarchical.py this task; default behavior elsewhere
# unchanged). Compares against the existing zero-init runs (SYN_clean_w5 / SYN_clean_pose35_w5)
# on the SAME corpora, to see whether starting at the true pose drifts away under the same loss.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

CORPORA=(synth_clean synth_clean_pose35)
LABELS=(pose25 pose35)
CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
RUN_NAME="SYN_clean_${LABEL}_gtinit_w5"
HIER_OUT="diagnostics/registration_failure/runs/hier_${RUN_NAME}"
MOONSHOT_OUT="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/registration_failure/logs
mkdir -p "$HIER_OUT" "$MOONSHOT_OUT" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS (GT joint_rot init) ==="
nvidia-smi || true

echo "=== hierarchical placement (full budget, GT-init) ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --init_joint_rot_from "$MESH_DIR/ground_truth.npz" \
  --results_dir "$HIER_OUT" 2>&1 | tee "$LOGS/${RUN_NAME}_hier.log"

echo "=== moonshot handoff, yaml=D1_low.yaml (full budget) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from "$HIER_OUT/H2_joint.npz" \
  --results_dir "$MOONSHOT_OUT" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}_moonshot.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
