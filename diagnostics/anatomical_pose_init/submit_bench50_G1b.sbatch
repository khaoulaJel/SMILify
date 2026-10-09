#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=bench50_G1b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/bench50_G1b_slurm_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/bench50_G1b_slurm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/bench50_clean
INIT=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/bench50_learned_init_lowpose.npz
t=bench50_G1b_learned_lowpose

python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
  --init_joint_rot_from $INIT \
  --results_dir $R/${t}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/${t}_hier/H2_joint.npz \
  --results_dir $R/$t
echo "[$(date +%H:%M:%S)] $t done"
