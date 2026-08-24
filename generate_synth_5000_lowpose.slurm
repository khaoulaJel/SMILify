#!/bin/bash
#SBATCH --account=rwth2151
#SBATCH --job-name=smil_synth5000_lowpose
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smil_synth5000_lowpose_%j.out
#SBATCH --error=logs/smil_synth5000_lowpose_%j.err

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
export PYTHONPATH="$PWD:$PYTHONPATH"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# Hypothesis test (2026-08-20, RESULTS_bench50_G1_vs_G3.md hypothesis 1): the original
# pose_scale=0.50 corpus (mean ~23deg/joint) may not match real bench50 specimens' actual
# (much smaller, near-rest-pose) deviation, making the learned initializer over-correct on
# real data. pose_scale=0.15 -> mean ~13.8deg/joint, verified empirically on a 20-specimen
# test batch before committing to the full 5000.
python diagnostics/moonshot/generate_synth_large.py \
    --model 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl \
    --n-specimens 5000 \
    --batch-size 64 \
    --pose-scale 0.15 \
    --shape-scale 2.0 \
    --seed 20260821 \
    --output diagnostics/moonshot/synth_large_25pc_lowpose/ground_truth.npz
