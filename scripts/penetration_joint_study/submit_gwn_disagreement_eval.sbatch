#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu-devel
#SBATCH --job-name=gwn_disagreement_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/khaoula_review/gwn_eval_%x_%j.log
#SBATCH --error=diagnostics/khaoula_review/gwn_eval_%x_%j.log
#
# TASK 3: CPU-only GWN-vs-proximity disagreement eval, run_dir passed via --export.
# Submit: sbatch --export=RUN_DIR=fit3d_results_all_baseline scripts/penetration_joint_study/submit_gwn_disagreement_eval.sbatch
set -eo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
python -u diagnostics/moonshot/eval_gwn_disagreement.py --run_dir "$RUN_DIR" --stage Stage_3_deform_fine
echo "Done -- $(date)"
