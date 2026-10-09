#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=JAB_oracle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=diagnostics/joint_alignment_benchmark/sbatch_logs/oracle_%j.log
#SBATCH --error=diagnostics/joint_alignment_benchmark/sbatch_logs/oracle_%j.log
# Reference analyses O0/O1/O2 -- needs A_prod_s0 to exist (submit with --dependency on the arm array).
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
python -u diagnostics/joint_alignment_benchmark/tools/oracle.py
