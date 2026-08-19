#!/bin/bash
#SBATCH --job-name=rank5_c5
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/overnight_20260818/logs/rank5_%A_%a.out

# Master prompt Section 11 C5 -- model-anchored / leg-protected stratified target sampling.
# NEW code this run (fitter_3d/stratified_sampling.py:sample_target_distal_protected,
# leg_face_mask; wired via a NEW --distal_quota_protected flag, mutually exclusive with the
# existing --distal_quota). Tests whether Intervention C's antenna regression (Section 4) was a
# target-side BUDGET-THEFT artifact (distal_quota reallocates from ALL non-distal faces,
# including antenna/head/body) rather than a fundamental H7 resource constraint: here the quota
# is confined to a fixed LEG sub-budget, so non-leg sampling density is invariant to this flag
# by construction. Falsifier (pre-registered, per the master prompt's own C5 spec): if the
# antenna regresses anyway, H7 is confirmed fundamental, not a sampler artifact.
#
# ONE quota value (0.3), chosen a priori, not swept/tuned on results. split_distal is ON (13
# groups) because the plain 7-group partition never gives pretarsus its own correspondence
# term at all (Section 2/Intervention A) -- distal_quota_protected without split_distal would
# only rebalance sampling WITHIN an already-pooled leg group's chamfer, not give the distal
# segment its own data term, so this combination is the minimal meaningful test of C5's claim.
#
# Runs on pose25 (synth_clean, the primary/default corpus) and pose0 (synth_clean_pose0, already
# used as a second hypothesis-set anchor by Rank 1) -- reuses the SAME hierarchical-placement
# recipe and moonshot D1_low.yaml handoff as every other baseline/intervention arm in this
# project, changing ONLY the target-sampling flag.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

CORPORA=(synth_clean synth_clean_pose0)
LABELS=(pose25 pose0)
CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
QUOTA=0.3

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
RUN_NAME="SYN_clean_${LABEL}_c5protected_w5"
HIER_OUT="diagnostics/overnight_20260818/runs/hier_${RUN_NAME}"
# moonshot output deliberately lands under diagnostics/moonshot/runs/ (NOT overnight_20260818/),
# matching every other run in this project, so it is directly readable by
# diagnostics/absolute_scale/measure.py:measure_run() (hardcodes MOON/runs/<run>) without any
# path-handling changes -- same convention Rank 1/2/4's analysis scripts already rely on.
MOONSHOT_OUT="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/overnight_20260818/logs
mkdir -p "$HIER_OUT" "$MOONSHOT_OUT" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL quota=$QUOTA ==="
nvidia-smi || true

echo "=== hierarchical placement, split_distal + distal_quota_protected=$QUOTA ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --split_distal --distal_quota_protected "$QUOTA" \
  --results_dir "$HIER_OUT" 2>&1 | tee "$LOGS/${RUN_NAME}_hier.log"

echo "=== moonshot handoff, yaml=D1_low.yaml (baseline, unmodified) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from "$HIER_OUT/H2_joint.npz" \
  --results_dir "$MOONSHOT_OUT" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}_moonshot.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
