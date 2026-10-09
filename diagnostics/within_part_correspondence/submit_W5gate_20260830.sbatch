#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=W5_gate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=diagnostics/within_part_correspondence/sbatch_logs/W5gate_%j.log
#SBATCH --error=diagnostics/within_part_correspondence/sbatch_logs/W5gate_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# W5 step 1 -- build W3b correspondence for synth_power48 and GATE the fitter runs on it.
# Bar in PREREGISTRATION_W5_fitter_propagation.md. The gate is voiding: if W3b's coxal
# correspondence is not actually better than C3's in the deployment regime, decision-tree row 3
# applies and the fitter arms are NOT launched.
python -u diagnostics/within_part_correspondence/w5_generate_and_gate.py
