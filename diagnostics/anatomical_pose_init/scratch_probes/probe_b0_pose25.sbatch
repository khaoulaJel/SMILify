#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=probeB0_pose25
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=diagnostics/anatomical_pose_init/scratch_probes/probeB0_pose25_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/scratch_probes/probeB0_pose25_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
python -u diagnostics/anatomical_pose_init/scratch_probes/run_probe_b0.py
echo done
