#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=W3b_conv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=diagnostics/within_part_correspondence/sbatch_logs/W3b_%j.log
#SBATCH --error=diagnostics/within_part_correspondence/sbatch_logs/W3b_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# W3 -- a descriptor trained FOR within-part cross-specimen correspondence, with the negatives the
# evaluation actually uses. Bar in PREREGISTRATION_W3_contextual_embedding.md, fixed before the run:
# beat the SUBSET-MATCHED XYZ_RIGID on co/tr/fe/ti individually, >=15% relative, sign p<0.05.
# Budget is 6h; if training has not converged the arm is INCONCLUSIVE ON BUDGET, not a FAIL.
python diagnostics/within_part_correspondence/w3_contextual_embedding.py \
    --epochs 150 --steps_per_epoch 250 --batch_pairs 4 --n_points 4096 \
    --n_subsets 8 --n_pairs 20 --seed 0 \
    --out_dir diagnostics/within_part_correspondence/out_W3b
