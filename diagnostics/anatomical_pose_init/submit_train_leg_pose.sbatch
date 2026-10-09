#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=train_leg_pose
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/learned_init_20260820_slurm_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/learned_init_20260820_slurm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD:$PYTHONPATH"

python -u diagnostics/anatomical_pose_init/train_leg_pose_regressor.py \
    --train_corpus diagnostics/moonshot/synth_large/ground_truth.npz \
    --held_out_corpus diagnostics/moonshot/synth_clean/ground_truth.npz \
    --model 3D_model_prep/SMIL_OmniAnt.pkl \
    --out_dir diagnostics/anatomical_pose_init/learned_init_20260820 \
    --epochs 200 \
    --batch_size 16 \
    --n_points 2048

echo "[$(date +%H:%M:%S)] train_leg_pose done"
