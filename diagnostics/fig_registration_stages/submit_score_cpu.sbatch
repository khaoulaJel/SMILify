#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=fig_regstages_score_cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/fig_registration_stages/sbatch_logs/score_cpu_%j.log
#SBATCH --error=diagnostics/fig_registration_stages/sbatch_logs/score_cpu_%j.log
# Task 2: re-score (no fitting) the D1_PROD Atta run and render the panels and probes.
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
B=diagnostics/fig_registration_stages
R=/hpcwork/nao48500/fig_registration_stages
echo "host=$(hostname) head=$(git rev-parse HEAD) job=$SLURM_JOB_ID"
python -u $B/score_stages.py --hier_dir $R/D1_s0_cpu_hier --moon_dir $R/D1_s0_cpu --mesh_dir /hpcwork/nao48500/atta20 --out_dir $B/run_cpu
python -u $B/render_stages.py --out_dir $B/run_cpu
echo DONE
