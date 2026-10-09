#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=W2_cse
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/within_part_correspondence/sbatch_logs/W2_%j.log
#SBATCH --error=diagnostics/within_part_correspondence/sbatch_logs/W2_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# W2 -- does the LEARNED descriptor (C11 CSE head) beat plain part-local rigid proximity INSIDE an
# already-correct part? Bar in PREREGISTRATION_W2_learned_descriptor.md, fixed before the run:
# CSE_C11 must beat XYZ_RIGID on co/tr/fe/ti individually, >=15% relative, sign p<0.05.
# Inference only -- no training. Runs before any new architecture is funded.
python diagnostics/within_part_correspondence/w2_learned_descriptor.py \
    --n_specimens 12 --n_pairs 20 --seed 0 \
    --out_dir diagnostics/within_part_correspondence/out_W2
