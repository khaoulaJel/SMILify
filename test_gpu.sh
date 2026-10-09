#!/bin/bash
#SBATCH --job-name=smil_gpu_test
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=test_gpu_%j.log

module load CUDA/11.8.0
conda activate pytorch3d

echo "=== GPU ==="
nvidia-smi

echo "=== PyTorch ==="
python -c "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "=== PyTorch3D ==="
python -c "import pytorch3d; print('pytorch3d:', pytorch3d.__version__)"
