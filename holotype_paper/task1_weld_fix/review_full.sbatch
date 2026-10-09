#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=T1_rev2
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=holotype_paper/task1_weld_fix/sbatch_logs/review2_%j.log
#SBATCH --error=holotype_paper/task1_weld_fix/sbatch_logs/review2_%j.log
cd "$SLURM_SUBMIT_DIR"; D=holotype_paper/task1_weld_fix; LIST=$1
source /home/nao48500/miniforge3/etc/profile.d/conda.sh && conda activate pytorch3d
export OMP_NUM_THREADS=4
cat $LIST | xargs -P 12 -n 1 python -u $D/t1_measure.py; echo "measure done $(date -Is)"
cat $LIST | xargs -P 12 -n 1 python -u $D/t1_fidelity_maps.py; echo "fidelity done $(date -Is)"
cat $LIST | xargs -P 12 -n 1 python -u $D/t1_plot_maps.py; echo "maps done $(date -Is)"
cat $LIST | xargs -P 8 -n 1 /home/nao48500/software/blender-4.2.23-linux-x64/blender --background --python $D/t1_render.py --; echo "render done $(date -Is)"
