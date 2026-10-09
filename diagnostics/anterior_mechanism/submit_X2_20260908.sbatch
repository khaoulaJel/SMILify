#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=X2_anterior
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/anterior_mechanism/sbatch_logs/X2_%j.log
#SBATCH --error=diagnostics/anterior_mechanism/sbatch_logs/X2_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# X2 -- does an externally-validated allometric prior fix the anterior anatomy where the scale
# cap didn't? Bar in PREREGISTRATION_X2_allometric_prior.md, fixed before this ran.
#   arm A = A_nocap.yaml, REUSED from X1 (diagnostics/moonshot/runs/X1_A_nocap) -- same seed 0,
#           config confirmed byte-identical to the one on disk (only w_scale=0.0 vs D1_PROD).
#   arm B = B_scalecap.yaml, REUSED from X1 (diagnostics/moonshot/runs/X1_B_scalecap) -- same
#           seed 0, config confirmed byte-identical to D1_PROD (shipped w_scale=0.052).
#   arm C = C_allometric.yaml, RUN FRESH here -- A_nocap.yaml base (w_scale=0.0) plus the
#           calibrated w_allo=0.00051 at both stages (RESULTS_X2_20260908.md has the
#           calibration record). Diff-verified against A_nocap.yaml: exactly the two w_allo
#           lines added, nothing else.
D=diagnostics/moonshot/bench50_clean
R=diagnostics/moonshot/runs
C=diagnostics/anterior_mechanism

echo "[$(date +%H:%M:%S)] arm C -- allometric prior (fresh run; A and B reused from X1)"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src $C/C_allometric.yaml --seed 0 --results_dir $R/X2_C_allometric

echo "[$(date +%H:%M:%S)] measurements (PROXY + MECHANISM + cost condition)"
python -u $C/x2_measurements.py
echo "[$(date +%H:%M:%S)] X2 done"
