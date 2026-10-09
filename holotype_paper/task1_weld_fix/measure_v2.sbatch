#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=T1_mv2
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=holotype_paper/task1_weld_fix/sbatch_logs/measure_v2_%j.log
#SBATCH --error=holotype_paper/task1_weld_fix/sbatch_logs/measure_v2_%j.log
cd "$SLURM_SUBMIT_DIR"; D=holotype_paper/task1_weld_fix
source /home/nao48500/miniforge3/etc/profile.d/conda.sh && conda activate pytorch3d
export OMP_NUM_THREADS=4
cat $D/specimens_39.txt | xargs -P 14 -n 1 python -u $D/t1_measure_v2.py; echo "measure_v2 done $(date -Is)"
