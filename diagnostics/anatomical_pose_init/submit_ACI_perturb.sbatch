#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=ACI_perturb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/ACI_perturb_%a_slurm_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/ACI_perturb_%a_slurm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean

CONDS=(random proximal distal)
NAMES=(ACI_D_random ACI_E_proximal ACI_F_distal)
cond=${CONDS[$SLURM_ARRAY_TASK_ID]}
t=${NAMES[$SLURM_ARRAY_TASK_ID]}
INIT=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/${cond}_init.npz

# Identical D1 recipe as SYN_clean_w5 (Arm A) / ACI_cheap (Arm B) / ACI_learned (Arm C) --
# only --init_joint_rot_from differs, pointing at the structured-perturbation init for this
# array task (see generate_structured_perturbation_init.py).
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
  --init_joint_rot_from $INIT \
  --results_dir $R/${t}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/${t}_hier/H2_joint.npz \
  --results_dir $R/$t
echo "[$(date +%H:%M:%S)] $t done"
