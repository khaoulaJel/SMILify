#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=RM_Q2_prep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=research/review_methods/Q2_oracle_budget/sbatch_logs/prep_%j.log
#SBATCH --error=research/review_methods/Q2_oracle_budget/sbatch_logs/prep_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
Q=research/review_methods/Q2_oracle_budget; O=/hpcwork/nao48500/review_methods/Q2
echo "head=$(git rev-parse HEAD) host=$(hostname)"
python -u $Q/q2_make_sets_and_targets.py
for R in REALPOSE REALSCALE; do
  python -u diagnostics/anatomical_pose_init/generate_cse_correspondence_20260827.py \
    --mesh_dir $O/$R/meshes --normalise --filter cycle --keep_frac 0.5 --segments all \
    --out $O/$R/targets_generator.npz
done
python -u $Q/q2_parity_probe.py
