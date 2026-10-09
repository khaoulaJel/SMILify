#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=Z5_dsym
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/deform_nature/sbatch_logs/Z5_%j.log
#SBATCH --error=diagnostics/deform_nature/sbatch_logs/Z5_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0   # Z3 refuted coupling; keep it off

# Z5 -- does constraining deform_verts to be bilateral improve transfer on WORKERS?
# Bar fixed in PREREGISTRATION_Z5_deform_symmetry.md BEFORE this ran.
C=diagnostics/deform_nature
M=diagnostics/moonshot/bench50_clean

echo "[$(date +%H:%M:%S)] arm A -- as shipped"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
  --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml --seed 0 \
  --results_dir diagnostics/moonshot/runs/Z5_A_off

echo "[$(date +%H:%M:%S)] arm B -- w_deform_sym 2.0"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
  --yaml_src $C/B_dsym2.yaml --seed 0 \
  --results_dir diagnostics/moonshot/runs/Z5_B_dsym2

echo "[$(date +%H:%M:%S)] arm C -- w_deform_sym 10.0"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
  --yaml_src $C/C_dsym10.yaml --seed 0 \
  --results_dir diagnostics/moonshot/runs/Z5_C_dsym10

echo "[$(date +%H:%M:%S)] score"
python -u $C/z5_score.py
echo "[$(date +%H:%M:%S)] Z5 done"
