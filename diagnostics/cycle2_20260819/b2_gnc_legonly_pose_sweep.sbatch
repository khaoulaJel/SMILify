#!/bin/bash
#SBATCH --job-name=b2_gnc_sweep
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-6
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/cycle2_20260819/logs/b2_%A_%a.out

# B2 (task list): GNC across the full pose sweep. gnc_legonly is the surviving GNC variant from
# B1 (largest R gain, +0.137, at N=50 with tight cross-seed CV=0.34%, AND antenna IMPROVES
# rather than regresses -- -8.6% vs baseline -- so it clearly beats gnc_medium/gnc_slow/
# gnc_medium_splitdistal as the candidate to carry into the pose sweep).
#
# Reuses EXISTING hierarchical placements (zero-init, same recipe as every baseline moonshot run
# in this project) for all 7 pose levels -- no new hierarchical compute needed, only the
# moonshot handoff with gnc_legonly.yaml vs the existing D1_low.yaml baseline runs (already on
# disk: SYN_clean_pose{0,05,10,15,20,35}_w5, SYN_clean_w5 -- NOT rerun here).
#
# Tracks cross-leg confusion specifically (not just pooled R) across pose severity, per the
# task list: "pose severity produces cross-leg correspondence failure, and annealed
# robustification suppresses it in proportion to that failure" is the claim being tested.
# Falsifier: if GNC's benefit is flat across pose severity rather than scaling with it.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

CORPORA=(synth_clean_pose0 synth_clean_pose05 synth_clean_pose10 synth_clean_pose15 synth_clean_pose20 synth_clean synth_clean_pose35)
LABELS=(pose0 pose05 pose10 pose15 pose20 pose25 pose35)
HIER_DIRS=(
  diagnostics/appendage_evidence/runs/hier_SYN_clean_pose0_w5
  diagnostics/registration_failure/runs/hier_SYN_clean_pose05_w5
  diagnostics/registration_failure/runs/hier_SYN_clean_pose10_w5
  diagnostics/registration_failure/runs/hier_SYN_clean_pose15_w5
  diagnostics/registration_failure/runs/hier_SYN_clean_pose20_w5
  diagnostics/offset_sweep_evidence/runs/hier_SYN_clean_w5
  diagnostics/registration_failure/runs/hier_SYN_clean_pose35_w5
)

CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
HIER_DIR=${HIER_DIRS[$SLURM_ARRAY_TASK_ID]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
RUN_NAME="SYN_clean_${LABEL}_gnclegonly_w5"
OUT_DIR="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/cycle2_20260819/logs
mkdir -p "$OUT_DIR" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL hier_dir=$HIER_DIR ==="
test -f "$HIER_DIR/H2_joint.npz" || { echo "MISSING $HIER_DIR/H2_joint.npz"; exit 1; }
nvidia-smi || true

python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/cycle2_20260819/cfg/gnc_legonly.yaml \
  --init_from "$HIER_DIR/H2_joint.npz" \
  --results_dir "$OUT_DIR" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
