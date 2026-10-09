#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=HB_d1abl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/sbatch_logs/HB_d1abl_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/sbatch_logs/HB_d1abl_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_power48

# H_B of PREREGISTRATION_D1_coxal_degradation_20260828.md: does zeroing ONE existing D1 term stop
# the coxal regression? Config-only -- no fitter code is touched, so these cannot be confounded by
# the H_A port being developed in parallel.
#
# Every arm re-runs D1 from the SAME hierarchical handoff C13_uniform used
# (C13_uniform_hier/H2_joint.npz, co=0.2564), so the only variable is the one zeroed weight.
# D1_no_deform is the exception: it zeroes nothing and instead sets scheme:pose, removing
# deform_verts -- the other structural difference from the hier recipe, which ran --deform_its 0.
#
# PREDICTION RECORDED IN THE PRE-REGISTRATION: none of these clears the 70% bar (co <= 0.3030).
# They are run as the control that makes H_A's result interpretable, not as a hedge.

for t in D1_no_edge D1_no_sym D1_no_offset D1_no_midline D1_no_lap D1_no_deform; do
  echo "[$(date +%H:%M:%S)] $t"
  python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
    --yaml_src diagnostics/moonshot/cfg/hb/$t.yaml \
    --init_from $R/C13_uniform_hier/H2_joint.npz \
    --results_dir $R/$t
  python -u diagnostics/moonshot/eval_run.py --run_dir $R/$t --mesh_dir $D
done
echo "[$(date +%H:%M:%S)] H_B done"
