#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=Y1_pose
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/pose_causality/sbatch_logs/Y1_%j.log
#SBATCH --error=diagnostics/pose_causality/sbatch_logs/Y1_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# Y1 -- is POSE the cause of the worker registration failure? Bar in
# PREREGISTRATION_Y1_pose_causality.md, fixed before this ran.
# A pure pose ladder (betas byte-identical, joint_rot exactly rescaled) is FITTED at each rung, and
# the shape space is built from the FITS -- not from ground truth, which is pose-invariant by
# construction and would measure an identity.
C=diagnostics/pose_causality

echo "[$(date +%H:%M:%S)] generate ladder"
python -u $C/y1_generate_corpora.py --n 50 --seed 0 --out_root $C/corpora

for PS in 0.05 0.15 0.25 0.50 0.75; do
  echo "[$(date +%H:%M:%S)] fit pose_scale $PS"
  python -u -m fitter_3d.optimise_moonshot \
    --mesh_dir $C/corpora/ps${PS} \
    --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml \
    --seed 0 \
    --results_dir diagnostics/moonshot/runs/Y1_ps${PS}
done

echo "[$(date +%H:%M:%S)] analysis"
python -u $C/y1_shape_space_analysis.py
echo "[$(date +%H:%M:%S)] Y1 done"
