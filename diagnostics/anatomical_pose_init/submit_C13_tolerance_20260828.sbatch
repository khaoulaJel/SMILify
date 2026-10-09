#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=C13_tol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/sbatch_logs/C13_tol_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/sbatch_logs/C13_tol_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_power48
G=diagnostics/anatomical_pose_init/out_C3_cse_head_20260826

# C13: does weighting the dense correspondence term by INVERSE MEASURED INTER-LEG TOLERANCE
# move the coxal leg-level error? Bar fixed in advance in
# PREREGISTRATION_C13_tolerance_weighting_20260828.md -- PASS needs co leg error to fall by
# >= 0.05 absolute with sign p < 0.05, AND no distal regression > 0.02, AND overall leg_acc
# must not fall.
#
# All three arms share ONE correspondence file (cse_p48_all.npz, already on disk from C6) and
# one recipe. The ONLY thing that varies is the per-vertex weight, so the contrast is clean.
#
# C13_uniform is not redundant with the existing P48_cse_all row: it re-runs the control THROUGH
# THE EDITED TRAINER, so if the weighting plumbing perturbed the unweighted path at all, that
# shows up as C13_uniform != P48_cse_all instead of silently contaminating the contrast.

run_arm () {
  local t=$1; shift
  echo "[$(date +%H:%M:%S)] $t: hierarchical"
  python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \
    "$@" --results_dir $R/${t}_hier
  echo "[$(date +%H:%M:%S)] $t: D1 fit"
  python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
    --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t
  echo "[$(date +%H:%M:%S)] $t: eval"
  python -u diagnostics/moonshot/eval_run.py --run_dir $R/$t --mesh_dir $D
}

run_arm C13_uniform  --cse_correspondence_from $G/cse_p48_all.npz
run_arm C13_invtol   --cse_correspondence_from $G/cse_p48_all.npz --cse_tolerance_weighting invtol
run_arm C13_invtol2  --cse_correspondence_from $G/cse_p48_all.npz --cse_tolerance_weighting invtol2

echo "[$(date +%H:%M:%S)] C13 done"
