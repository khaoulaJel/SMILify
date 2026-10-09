#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=T1_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=holotype_paper/task1_weld_fix/sbatch_logs/smoke_%j.log
#SBATCH --error=holotype_paper/task1_weld_fix/sbatch_logs/smoke_%j.log
# Task 1 smoke: one raw specimen through V0, V1, V2 (CPU-bound Blender; rwth2151 has GPU partitions only).
cd "$SLURM_SUBMIT_DIR"
S=$(head -1 holotype_paper/task1_weld_fix/specimens.txt)
for V in V0 V1 V2; do bash holotype_paper/task1_weld_fix/run_one.sh $V $S & done; wait
