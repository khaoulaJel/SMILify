#!/bin/bash
#SBATCH --job-name=smil_synth5000
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smil_synth5000_%j.out
#SBATCH --error=logs/smil_synth5000_%j.err

set -e

cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Node: $(hostname)"
echo "CUDA:"
nvidia-smi

echo "PyTorch:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python diagnostics/moonshot/generate_synth_large.py \
    --n-specimens 5000 \
    --batch-size 64 \
    --pose-scale 0.50 \
    --shape-scale 2.0 \
    --seed 20260820 \
    --output diagnostics/moonshot/synth_large/ground_truth.npz
