#!/bin/bash
#SBATCH --account=rwth2151
#SBATCH --job-name=smil_synth5000_25pc
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smil_synth5000_25pc_%j.out
#SBATCH --error=logs/smil_synth5000_25pc_%j.err

set -e

cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export PYTHONPATH="$PWD:$PYTHONPATH"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

echo "Node: $(hostname)"
echo "CUDA:"
nvidia-smi

python diagnostics/moonshot/generate_synth_large.py \
    --model 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl \
    --n-specimens 5000 \
    --batch-size 64 \
    --pose-scale 0.50 \
    --shape-scale 2.0 \
    --seed 20260820 \
    --output diagnostics/moonshot/synth_large_25pc/ground_truth.npz
