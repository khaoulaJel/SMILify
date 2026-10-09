#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=X1_anterior
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=diagnostics/anterior_mechanism/sbatch_logs/X1_%j.log
#SBATCH --error=diagnostics/anterior_mechanism/sbatch_logs/X1_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# X1 -- did the SHIPPED scale cap fix the anterior anatomy, or only its scale statistics?
# Bar in PREREGISTRATION_X1_scalecap_mechanism.md, fixed before these ran.
#   arm A = D1_PROD with w_scale 0.0  (the TRUE no-cap baseline; D1_PROD.yaml itself now CONTAINS
#           w_scale 0.052, so reusing it for both arms would have run two identical arms)
#   arm B = D1_PROD as shipped, w_scale 0.052
# The two configs differ on exactly two lines, both w_scale. results_dir is overridden here.
D=diagnostics/moonshot/bench50_clean
R=diagnostics/moonshot/runs
C=diagnostics/anterior_mechanism

echo "[$(date +%H:%M:%S)] arm A -- no cap"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src $C/A_nocap.yaml --seed 0 --results_dir $R/X1_A_nocap

echo "[$(date +%H:%M:%S)] arm B -- scale cap (shipped)"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src $C/B_scalecap.yaml --seed 0 --results_dir $R/X1_B_scalecap

echo "[$(date +%H:%M:%S)] analysis"
python -u $C/x1_anterior_mechanism.py
echo "[$(date +%H:%M:%S)] X1 done"
