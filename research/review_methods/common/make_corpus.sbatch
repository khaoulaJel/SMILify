#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=RM_corpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=research/review_methods/common/sbatch_logs/corpus_%j.log
#SBATCH --error=research/review_methods/common/sbatch_logs/corpus_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "head=$(git rev-parse HEAD) host=$(hostname)"
python -u research/review_methods/common/make_corpus_with_joints.py
