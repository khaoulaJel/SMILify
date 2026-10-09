#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=ACI_noise20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/out_ceiling_20260820/runs_log/ACI_noise20_slurm_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/out_ceiling_20260820/runs_log/ACI_noise20_slurm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
  --init_joint_rot_from diagnostics/anatomical_pose_init/out_ceiling_20260820/pose_noise/noise20.npz \
  --results_dir $R/ACI_noise20_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/ACI_noise20_hier/H2_joint.npz \
  --results_dir $R/ACI_noise20
echo "[$(date +%H:%M:%S)] ACI_noise20 done"
