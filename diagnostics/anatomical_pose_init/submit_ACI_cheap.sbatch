#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=ACI_cheap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/out_ceiling_20260820/ACI_cheap_slurm_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/out_ceiling_20260820/ACI_cheap_slurm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
INIT=diagnostics/anatomical_pose_init/out_ceiling_20260820/cheap_init.npz
t=ACI_cheap

# Same D1 recipe as the existing SYN_clean_w5 (zero-init, Arm A) / SYN_clean_pose25_gtinit_w5
# (gt-init, Arm C) runs this experiment reuses -- run_m1_fit_all.sh's fit() function -- with
# only --init_joint_rot_from added, pointing at the cheap_anatomical initialiser's output
# instead of ground_truth.npz.
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
  --init_joint_rot_from $INIT \
  --results_dir $R/${t}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/${t}_hier/H2_joint.npz \
  --results_dir $R/$t
echo "[$(date +%H:%M:%S)] $t done"
