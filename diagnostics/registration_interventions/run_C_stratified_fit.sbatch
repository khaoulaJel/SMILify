#!/bin/bash
#SBATCH --job-name=C_stratsamp
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/registration_interventions/logs/C_strat_%A_%a.out

# Task 7, Intervention C, Phase C1. SAME recipe as the Task 3/6 baseline arms (SYN_clean_pose0_w5,
# SYN_clean_w5 -- both already fit, reused unchanged as baseline, NOT re-run here), changing ONLY
# --distal_quota on the hierarchical stage (H1/H2/H3; H0 untouched, see MECHANISM_C.md). No
# --split_distal, no other flag changes. --distal_quota requires topology-preserving corpora, so
# only pose0/pose25 are covered here (Phase C1); drop30/drop60 are Phase C2, deferred.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

CORPORA=(synth_clean_pose0 synth_clean_pose0 synth_clean_pose0 synth_clean synth_clean synth_clean)
LABELS=(pose0 pose0 pose0 pose25 pose25 pose25)
QUOTAS=(0.05 0.10 0.20 0.05 0.10 0.20)
QLABELS=(q05 q10 q20 q05 q10 q20)

CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
QUOTA=${QUOTAS[$SLURM_ARRAY_TASK_ID]}
QLABEL=${QLABELS[$SLURM_ARRAY_TASK_ID]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
RUN_NAME="SYN_clean_${LABEL}_strat${QLABEL}_w5"
HIER_OUT="diagnostics/registration_interventions/runs/hier_${RUN_NAME}"
MOONSHOT_OUT="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/registration_interventions/logs
mkdir -p "$HIER_OUT" "$MOONSHOT_OUT" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL quota=$QUOTA ==="
nvidia-smi || true

echo "=== hierarchical placement (full budget, --distal_quota $QUOTA) ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --distal_quota "$QUOTA" \
  --results_dir "$HIER_OUT" 2>&1 | tee "$LOGS/${RUN_NAME}_hier.log"

echo "=== moonshot handoff, yaml=D1_low.yaml (full budget) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from "$HIER_OUT/H2_joint.npz" \
  --results_dir "$MOONSHOT_OUT" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}_moonshot.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
