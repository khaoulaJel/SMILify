#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=F6_soft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/sbatch_logs/F6_soft_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/sbatch_logs/F6_soft_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# F6 -- geodesic-softened targets. Bar in PREREGISTRATION_F6_soft_targets_20260828.md.
# The stock loss is F.cross_entropy against ONE true vertex, so on a 10,235-vertex mesh the vertex
# 0.1mm away is punished as hard as one on the opposite leg -- while E0's measured 0.0908 metric
# floor and C3's own docstring both say near-misses are cheap. F1 showed the information IS in the
# features (circumferential angle decodable at 52.9deg vs 90 chance) while retrieval top1 = 0.071.
# This is the loss-side fix that finding points at.
#
# Identical to the C3/C11 recipe except --soft_targets. Verified before launch: sigma reads off the
# mesh (median 3rd graph-neighbour, 0.011780), mean self-weight 0.172 vs 0.042 uniform, and
# cross-leg target mass is exactly 0.0 after masking (0.46 max on one coxa vertex before it).
python -u diagnostics/anatomical_pose_init/train_cse_head_20260826.py \
  --soft_targets \
  --out_dir diagnostics/anatomical_pose_init/out_F6_soft_20260828 \
  --epochs 120
echo "[$(date +%H:%M:%S)] F6 done"
