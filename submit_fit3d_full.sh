#!/bin/bash
#SBATCH --job-name=fit3d_full_gentle
#SBATCH --output=logs/fit3d_full_gentle_%j.log
#SBATCH --error=logs/fit3d_full_gentle_%j.err
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=ias-7

source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate pytorch3d 2>&1
cd /p/home/jusers/jellal1/jureca/SMILify

mkdir -p logs fit3d_results_full_gentle

echo "Starting Full Dataset Fit3D Optimization (Gentle Config)..."

python3 -m fitter_3d.optimise \
    --results_dir fit3d_results_full_gentle \
    --yaml_src fitter_3d/ants_cfg_penetration_gentle.yaml \
    --mesh_dir /p/scratch/cias-7/jellal1/antscan/antscan_processed_enhanced \
    --seed 0

echo "Optimization complete."
