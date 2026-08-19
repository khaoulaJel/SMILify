#!/bin/bash
#SBATCH --job-name=b3_ceiling
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/cycle2_20260819/logs/b3_%A_%a.out

# Cycle2 B3: the properly-isolated capacity-vs-sampling ceiling cell (GT pose + free bone scale
# + leg-protected target sampler). Completes the 2x2 from the task list; the other 3 cells
# (baseline zero-init, and the unprotected ceiling) already exist from overnight_20260818 Rank 2.

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
OUT_DIR="diagnostics/cycle2_20260819/runs/B3_CEILING_PROTECTED_${LABEL}"
LOGS=diagnostics/cycle2_20260819/logs
mkdir -p "$OUT_DIR" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL ==="
nvidia-smi || true

python -u diagnostics/cycle2_20260819/b3_ceiling_isolated.py \
  --mesh_dir "$MESH_DIR" \
  --results_dir "$OUT_DIR" \
  --nits 1200 --quota 0.3 2>&1 | tee "$LOGS/B3_${LABEL}.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
