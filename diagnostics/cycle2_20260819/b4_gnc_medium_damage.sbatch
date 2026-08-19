#!/bin/bash
#SBATCH --job-name=b4_gncmed_damage
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/cycle2_20260819/logs/b4med_%A_%a.out

# B4 SUBSTITUTE arm: gnc_legonly failed on drop30/drop60 with a confirmed, reproducible,
# structural cause -- sample_leg_nonleg_split assumes the target mesh shares the TEMPLATE's
# exact face indexing (topology-preserving), which drop30/drop60 violate by construction (they
# genuinely drop vertices/faces: drop30 has 9989/10235 verts, drop60 has 9743/10235 -- verified
# directly, not assumed). See diagnostics/cycle2_20260819/manifest.json's b4_gnc_legonly_damage
# entry for the full root-cause writeup. NOT a retry of that failed run under changed
# parameters -- this is gnc_medium (global GNC, robust_leg_only NOT set, plain
# sample_points_from_meshes with no face-index assumption), a DIFFERENT, already-validated
# (cycle 1 + B1 at N=50) variant that is not subject to the same failure mode.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

CORPORA=(synth_clean_drop30 synth_clean_drop60)
LABELS=(drop30 drop60)
HIER_DIRS=(diagnostics/appendage_evidence/runs/hier_SYN_clean_drop30_w5 diagnostics/appendage_evidence/runs/hier_SYN_clean_drop60_w5)

CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
HIER_DIR=${HIER_DIRS[$SLURM_ARRAY_TASK_ID]}

MESH_DIR="/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/moonshot/${CORPUS}"
RUN_NAME="SYN_clean_${LABEL}_gncmedium_w5"
OUT_DIR="diagnostics/moonshot/runs/${RUN_NAME}"
LOGS=diagnostics/cycle2_20260819/logs
mkdir -p "$OUT_DIR" "$LOGS"

echo "=== task $SLURM_ARRAY_TASK_ID: corpus=$CORPUS label=$LABEL hier_dir=$HIER_DIR ==="
test -f "$HIER_DIR/H2_joint.npz" || { echo "MISSING $HIER_DIR/H2_joint.npz"; exit 1; }
nvidia-smi || true

python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir "$MESH_DIR" \
  --yaml_src diagnostics/overnight_20260818/cfg/gnc_medium.yaml \
  --init_from "$HIER_DIR/H2_joint.npz" \
  --results_dir "$OUT_DIR" \
  --eval 2>&1 | tee "$LOGS/${RUN_NAME}.log"

echo "=== task $SLURM_ARRAY_TASK_ID DONE ==="
