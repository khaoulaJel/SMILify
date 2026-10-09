#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=T1_pseudo
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=holotype_paper/task1_weld_fix/sbatch_logs/pseudo_%j.log
#SBATCH --error=holotype_paper/task1_weld_fix/sbatch_logs/pseudo_%j.log
# Pseudoneoponera V1 hit the 2 h cap in packed.sbatch (V0 took 44 min). Rerun with unbuffered output
# (progress visible in run.log) and a 9 h cap to see whether it finishes and which step costs the time.
cd "$SLURM_SUBMIT_DIR"; D=holotype_paper/task1_weld_fix
export PYTHONUNBUFFERED=1
S=Pseudoneoponera_rufipes_CASENT0877628
timeout 9h bash $D/run_one.sh V1 $S; echo "V1 done $(date -Is)"
