#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=RM_Q2_orc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=research/review_methods/Q2_oracle_budget/sbatch_logs/oracle_%A_%a.log
#SBATCH --error=research/review_methods/Q2_oracle_budget/sbatch_logs/oracle_%A_%a.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
REGIMES=(REALPOSE REALSCALE)
R=${REGIMES[$((SLURM_ARRAY_TASK_ID / 3))]}; S=$((SLURM_ARRAY_TASK_ID % 3))
echo "regime=$R seed=$S head=$(git rev-parse HEAD) host=$(hostname)"
python -u research/review_methods/Q2_oracle_budget/q2_oracle.py --regime $R --seed $S
