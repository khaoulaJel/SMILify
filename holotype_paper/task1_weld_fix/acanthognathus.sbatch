#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=T1_acantho
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=holotype_paper/task1_weld_fix/sbatch_logs/acantho_%j.log
#SBATCH --error=holotype_paper/task1_weld_fix/sbatch_logs/acantho_%j.log
# The specimen that motivated the fix (master removed the whole head). V0/V1/V2, then measure + render.
cd "$SLURM_SUBMIT_DIR"; D=holotype_paper/task1_weld_fix; S=Acanthognathus_cf.ocellatus_CASENT0744647
export RAW_ROOT=/hpcwork/nao48500/holotype_task1/extra_raw
for V in V0 V1 V2; do bash $D/run_one.sh $V $S & done; wait
source /home/nao48500/miniforge3/etc/profile.d/conda.sh && conda activate pytorch3d
python -u $D/t1_measure.py $S
/home/nao48500/software/blender-4.2.23-linux-x64/blender --background --python $D/t1_render.py -- $S
