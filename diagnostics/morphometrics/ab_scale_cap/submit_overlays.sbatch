#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=ab_overlays
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/morphometrics/ab_scale_cap/overlays_slurm_%j.log
#SBATCH --error=diagnostics/morphometrics/ab_scale_cap/overlays_slurm_%j.log
set -eo pipefail

# Task C: 12 stratified template(target)-vs-fit overlays for the final D1_PROD (arm B fits,
# now == D1_PROD's shipped weights). Reuses diagnostics/moonshot/render_3d.py unchanged --
# already does exactly this (target scan | fit, deform-coloured), just never pointed at the
# real production corpus before. n_specimens=12 selects evenly spaced by filename across the
# 81 ALL_ANTS_CLEAN specimens (a taxonomic spread, since filenames are genus-sorted) for the
# easy/hard/multi-genus stratification §2.2 asks for.
CONDA_SH=/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
REPO_DIR=/p/home/jusers/jellal1/jureca/SMILify
source "$CONDA_SH"
conda activate pytorch3d
cd "$REPO_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

python -u diagnostics/moonshot/render_3d.py \
  --runs diagnostics/moonshot/runs/AB_B_CLEAN \
  --mesh_dir diagnostics/morphometrics/ab_scale_cap/stage_B/clean \
  --n_specimens 12 --mode deform \
  --out diagnostics/morphometrics/ab_scale_cap/dashboard

echo "[$(date +%H:%M:%S)] overlays complete"
