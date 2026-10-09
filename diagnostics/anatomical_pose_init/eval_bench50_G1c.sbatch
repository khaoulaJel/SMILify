#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=eval_bench50_G1c
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/eval_bench50_G1b_slurm_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/eval_bench50_G1b_slurm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u diagnostics/moonshot/eval_run.py \
    --run_dir diagnostics/moonshot/runs/bench50_G1c_learned_lowpose10 \
    --mesh_dir diagnostics/moonshot/bench50_clean
echo "[$(date +%H:%M:%S)] eval_bench50_G1b done"
