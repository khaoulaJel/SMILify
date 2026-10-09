#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=RM_S2conf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=research/review_methods/Q3b_controlled_shift/sbatch_logs/s2confirm_%j.log
#SBATCH --error=research/review_methods/Q3b_controlled_shift/sbatch_logs/s2confirm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
echo "head=$(git rev-parse HEAD) host=$(hostname)"
python -u research/review_methods/P0_real_pose_corpus/p0_analyse.py
python -u research/review_methods/Q3b_controlled_shift/s2_confirm_make.py
S2_SETS=/hpcwork/nao48500/review_methods/shift_sets_P0 S2_DEG=0 S2_OUT=S2_confirm.json \
  python -u research/review_methods/Q3b_controlled_shift/s2_shift_battery.py
