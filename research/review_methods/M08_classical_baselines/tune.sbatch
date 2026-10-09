#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=RM_M08_tune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-29
#SBATCH --output=research/review_methods/M08_classical_baselines/sbatch_logs/tune_%A_%a.log
#SBATCH --error=research/review_methods/M08_classical_baselines/sbatch_logs/tune_%A_%a.log
# M08 pre-registered tuning grid (30 configs) on the pose_train tuning set. CPU-bound (BCPD, trimesh);
# a GPU partition is used because account rwth2151 has no CPU-partition access.
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" OMP_NUM_THREADS=16 SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
echo "head=$(git rev-parse HEAD) host=$(hostname)"
python -u research/review_methods/M08_classical_baselines/m08_tune.py --config $SLURM_ARRAY_TASK_ID
