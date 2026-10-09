#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=fig_regstages_blender
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/fig_registration_stages/sbatch_logs/blender_%j.log
#SBATCH --error=diagnostics/fig_registration_stages/sbatch_logs/blender_%j.log
# Task 2 final renders (style B), re-render only (no fitting). DEVIATIONS.md D1 covers the account.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
B=/hpcwork/nao48500/bpy_venv/lib/python3.11/site-packages/bpy
export LD_LIBRARY_PATH=$B/lib:$LD_LIBRARY_PATH
D=diagnostics/fig_registration_stages
echo "host=$(hostname) head=$(git rev-parse HEAD) job=$SLURM_JOB_ID"
md5sum $D/run_cpu1/probes/focus_meshes.npz $D/blender_render.py
/hpcwork/nao48500/bpy_venv/bin/python -I $D/blender_render.py --npz $D/run_cpu1/probes/focus_meshes.npz \
  --out $D/run_cpu1/blender --size 1600 --samples 128
echo DONE
