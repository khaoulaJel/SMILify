#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=T1_fidel
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=holotype_paper/task1_weld_fix/sbatch_logs/fidelity_%j.log
#SBATCH --error=holotype_paper/task1_weld_fix/sbatch_logs/fidelity_%j.log
cd "$SLURM_SUBMIT_DIR"; D=holotype_paper/task1_weld_fix
source /home/nao48500/miniforge3/etc/profile.d/conda.sh && conda activate pytorch3d
export OMP_NUM_THREADS=4
cat $1 | xargs -P 8 -n 1 python -u $D/t1_fidelity_maps.py
