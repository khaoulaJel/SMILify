#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=Z3_couple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/joint_blendshapes/sbatch_logs/Z3_%j.log
#SBATCH --error=diagnostics/joint_blendshapes/sbatch_logs/Z3_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# Z3 -- do the shape space's joint channels make shape transfer?
# Bar fixed in PREREGISTRATION_Z3_joint_blendshapes.md BEFORE this ran.
C=diagnostics/joint_blendshapes
M=diagnostics/moonshot/bench50_clean

echo "[$(date +%H:%M:%S)] re-verify coupling before any arm runs"
python -u $C/verify_coupling_PROBE.py

echo "[$(date +%H:%M:%S)] arm A -- as shipped, coupling OFF"
SMILIFY_COUPLE_JOINT_BLENDSHAPES=0 python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir $M --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml --seed 0 \
  --results_dir diagnostics/moonshot/runs/Z3_A_off

echo "[$(date +%H:%M:%S)] arm B -- coupling ON, free residual unpenalised"
SMILIFY_COUPLE_JOINT_BLENDSHAPES=1 python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir $M --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml --seed 0 \
  --results_dir diagnostics/moonshot/runs/Z3_B_couple

echo "[$(date +%H:%M:%S)] arm C -- coupling ON + w_jresid 5.0"
SMILIFY_COUPLE_JOINT_BLENDSHAPES=1 python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir $M --yaml_src $C/C_couple_jresid.yaml --seed 0 \
  --results_dir diagnostics/moonshot/runs/Z3_C_jresid

echo "[$(date +%H:%M:%S)] score"
python -u $C/z3_score.py
echo "[$(date +%H:%M:%S)] Z3 done"
