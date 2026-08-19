#!/bin/bash
#SBATCH --job-name=B0_geominit_val
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/registration_interventions/logs/B0_val_%j.out

# Pre-registered validation of the geometric leg initialiser (fitter_3d/geom_leg_init.py)
# BEFORE it is used in any actual fitting run. See probe_B0_geom_init_validation.py's docstring.

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

python -u diagnostics/registration_interventions/probe_B0_geom_init_validation.py 2>&1
