#!/bin/bash
#SBATCH --job-name=rank4_gnc
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/overnight_20260818/logs/rank4_%A_%a.out

# Master prompt Section 13 -- GNC / annealed Geman-McClure on the moonshot handoff stage.
# Mechanism ALREADY implemented in fitter_3d/trainer_moonshot.py:MoonshotStage (robust_scale ->
# robust_scale_end geometric interpolation over the stage); D1_low.yaml (baseline) never turns
# it on (defaults to plain L2). Three schedules, everything else identical to D1_low.yaml,
# one-variable-at-a-time.
#
# The SYN_clean_w5 baseline this compares against is hierarchical-placement (H0-H3, zero-init,
# --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0) THEN moonshot handoff --init_from
# H2_joint.npz with D1_low.yaml (see diagnostics/appendage_evidence/run_pose0_fit.sbatch:12-16,
# which documents the exact recipe SYN_clean_w5 was produced with). This script reproduces the
# IDENTICAL hierarchical placement step (so the fair comparison is same-hierarchical-init,
# handoff-yaml-only differs) and only swaps D1_low.yaml for the GNC variant at handoff.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

SCHEDULES=(fast medium slow)
SCHED=${SCHEDULES[$SLURM_ARRAY_TASK_ID]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/synth_clean"
RUN_NAME="GNC_${SCHED}"
HIER_OUT="diagnostics/overnight_20260818/runs/hier_${RUN_NAME}"
# moonshot output lands under diagnostics/moonshot/runs/ (matching every other run in this
# project) so it is directly readable by measure.py:measure_run() (hardcodes MOON/runs/<run>).
OUT_DIR="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/overnight_20260818/logs
mkdir -p "$HIER_OUT" "$OUT_DIR" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: schedule=$SCHED ==="
nvidia-smi || true

echo "=== hierarchical placement (IDENTICAL to the SYN_clean_w5 baseline recipe, zero-init) ==="
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir "$MESH_DIR" \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --results_dir "$HIER_OUT" 2>&1 | tee "$LOGS/${RUN_NAME}_hier.log"

echo "=== moonshot handoff, yaml=gnc_${SCHED}.yaml (only difference from baseline) ==="
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src "diagnostics/overnight_20260818/cfg/gnc_${SCHED}.yaml" \
  --init_from "$HIER_OUT/H2_joint.npz" \
  --results_dir "$OUT_DIR" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}_fit.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
