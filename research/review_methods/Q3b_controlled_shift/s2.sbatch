#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=RM_Q3b_S2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=research/review_methods/Q3b_controlled_shift/sbatch_logs/s2_%j.log
#SBATCH --error=research/review_methods/Q3b_controlled_shift/sbatch_logs/s2_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
echo "head=$(git rev-parse HEAD) host=$(hostname)"
python -u research/review_methods/common/make_shift_sets.py
python -u research/review_methods/Q3b_controlled_shift/s2_shift_battery.py
