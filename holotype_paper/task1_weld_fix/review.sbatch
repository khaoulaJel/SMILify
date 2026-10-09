#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=T1_review
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=holotype_paper/task1_weld_fix/sbatch_logs/review_%j.log
#SBATCH --error=holotype_paper/task1_weld_fix/sbatch_logs/review_%j.log
# Task 1 review of finished specimens, in parallel with the main run: measure (t1_measure.py), then
# render contact sheets (t1_render.py). Usage: sbatch review.sbatch <list-file>
cd "$SLURM_SUBMIT_DIR"; D=holotype_paper/task1_weld_fix; LIST=$1
source /home/nao48500/miniforge3/etc/profile.d/conda.sh && conda activate pytorch3d
export OMP_NUM_THREADS=4
cat $LIST | xargs -P 8 -n 1 python -u $D/t1_measure.py
echo "measure done $(date -Is)"
cat $LIST | xargs -P 6 -n 1 /home/nao48500/software/blender-4.2.23-linux-x64/blender --background --python $D/t1_render.py --
echo "render done $(date -Is)"
