#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=bench50_G1G3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/bench50_%a_slurm_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/bench50_%a_slurm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/bench50_clean
INIT=diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/bench50_learned_init.npz

# G3 (array 0): zero/default init, identity global orientation (the fitter's own default).
# G1 (array 1): learned raw-scan init (generate_bench50_learned_init.py), identity global
# orientation -- optimise_hierarchical.py --init_joint_rot_from has no global-orientation
# override regardless (deferred to a separate G2/G4 experiment, per explicit isolation
# decision: don't confound the representation test with the global-rotation question).
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  t=bench50_G3_zero
  INIT_FLAG=""
else
  t=bench50_G1_learned
  INIT_FLAG="--init_joint_rot_from $INIT"
fi

python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
  $INIT_FLAG \
  --results_dir $R/${t}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/${t}_hier/H2_joint.npz \
  --results_dir $R/$t
echo "[$(date +%H:%M:%S)] $t done"
