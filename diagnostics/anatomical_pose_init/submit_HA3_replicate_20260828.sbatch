#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=HA3_rep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/sbatch_logs/HA3_rep_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/sbatch_logs/HA3_rep_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_power48
G=diagnostics/anatomical_pose_init/out_C3_cse_head_20260826

# H_A3 -- FRESH INDEPENDENT REPLICATE (--seed 1) of D1_with_cse_nodeform. Bar in
# PREREGISTRATION_HA3_coxa_placement_20260828.md, written before this ran.
# Exists because H_A2's coxa-placement number was measured POST-HOC; re-scoring the same run
# against a check designed after seeing it would be the same evidence relabelled, not confirmation.
echo "[$(date +%H:%M:%S)] D1_with_cse_nodeform_s1"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/hb/D1_no_deform.yaml \
  --init_from $R/C13_uniform_hier/H2_joint.npz \
  --cse_correspondence_from $G/cse_p48_all.npz \
  --seed 1 \
  --results_dir $R/D1_with_cse_nodeform_s1
python -u diagnostics/moonshot/eval_run.py --run_dir $R/D1_with_cse_nodeform_s1 --mesh_dir $D
echo "[$(date +%H:%M:%S)] H_A3 done"
