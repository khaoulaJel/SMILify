#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=t11_refined_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/khaoula_v2/runs/t11_refined_slurm_%j.log
#SBATCH --error=diagnostics/khaoula_v2/runs/t11_refined_slurm_%j.log
set -eo pipefail

# Refinement-layer eval, held-out specimens synth_000-003 (same 4 T1.1's raw-DINO/HKS
# numbers were measured on). Template DINO is already cached from training; synth_000-003
# are not (T1.1's original run didn't cache), so those extract fresh here. HKS is
# recomputed too (cheap, CPU) for the matched-coverage addendum comparison.
source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export HF_HOME=/p/scratch/cias-7/jellal1/hf_cache
export HF_HUB_OFFLINE=1

python -u diagnostics/khaoula_v2/within_part_signal_dino.py --run T04_baseline \
  --refined diagnostics/khaoula_v2/out/refine_net.pt
echo "[$(date +%H:%M:%S)] refinement-layer eval complete"
