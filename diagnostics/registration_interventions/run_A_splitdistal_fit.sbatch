#!/bin/bash
#SBATCH --job-name=A_splitdistal
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/registration_interventions/logs/A_splitdistal_%A_%a.out

# Task 7, Intervention A. SAME recipe as the Task 3/6 baseline arms (SYN_clean_pose0_w5,
# SYN_clean_w5 [=pose_scale 0.25], SYN_clean_drop30_w5, SYN_clean_drop60_w5 -- all already fit,
# reused unchanged as the baseline comparator, see diagnostics/appendage_evidence/run_pose0_fit.sbatch
# and run_damage_fit.sbatch, diagnostics/registration_failure/run_pose_sweep_fit.sbatch), changing
# ONLY the hierarchical placement stage's --split_distal flag. Nothing else (lr, optimizer,
# pose init, deform weights, joint limits, n_sample, stages, D1_low.yaml handoff) changes.
#
# split_distal gives each leg TWO vertex/target groups (proximal: coxa/trochanter/femur;
# distal: tibia/tarsus/pretarsus) instead of one, in the H1_legs/H2_joint partitioned-chamfer
# stages of fitter_3d/trainer_hierarchical.py. See MECHANISM.md in this directory.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

CORPORA=(synth_clean_pose0 synth_clean synth_clean_drop30 synth_clean_drop60)
LABELS=(pose0 pose25 drop30 drop60)
CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
RUN_NAME="SYN_clean_${LABEL}_splitdistal_w5"
HIER_OUT="diagnostics/registration_interventions/runs/hier_${RUN_NAME}"
MOONSHOT_OUT="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/registration_interventions/logs
mkdir -p "$HIER_OUT" "$MOONSHOT_OUT" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL (split_distal) ==="
nvidia-smi || true

echo "=== hierarchical placement (full budget, --split_distal) ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --split_distal \
  --results_dir "$HIER_OUT" 2>&1 | tee "$LOGS/${RUN_NAME}_hier.log"

echo "=== moonshot handoff, yaml=D1_low.yaml (full budget) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from "$HIER_OUT/H2_joint.npz" \
  --results_dir "$MOONSHOT_OUT" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}_moonshot.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
