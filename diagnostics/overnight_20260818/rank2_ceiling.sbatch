#!/bin/bash
#SBATCH --job-name=rank2_ceiling
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/overnight_20260818/logs/rank2_%A_%a.out

# Master prompt Section 9.5.2 -- model-capacity ceiling. GT joint_rot held FIXED for the entire
# run (scheme: shape excludes joint_rot from the optimizer, see cfg/ceiling_shape_only.yaml),
# only global_rot/trans/betas/log_beta_scales/betas_trans are free. No deform_verts (scheme
# excludes it too), so this measures pure PARAMETRIC shape/scale capacity, not offset capacity.
# This is a MEASUREMENT, not an intervention -- see OVERNIGHT_REPORT.md Rank 2.

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
RUN_NAME="CEILING_${LABEL}"
OUT_DIR="diagnostics/overnight_20260818/runs/${RUN_NAME}"
INIT_NPZ="diagnostics/overnight_20260818/runs/${RUN_NAME}_gtinit.npz"
LOGS=diagnostics/overnight_20260818/logs
mkdir -p "$OUT_DIR" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL ==="
nvidia-smi || true

echo "=== building GT joint_rot init npz ==="
python -u diagnostics/overnight_20260818/rank2_build_gt_init_npz.py \
  --mesh_dir "$MESH_DIR" --out "$INIT_NPZ" 2>&1 | tee "$LOGS/${RUN_NAME}_buildinit.log"

echo "=== ceiling fit: GT pose fixed, scheme=shape (log_beta_scales free, no deform) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/overnight_20260818/cfg/ceiling_shape_only.yaml \
  --init_from "$INIT_NPZ" \
  --results_dir "$OUT_DIR" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}_fit.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
