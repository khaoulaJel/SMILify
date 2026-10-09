#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=C14p_gt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/sbatch_logs/C14p_gt_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/sbatch_logs/C14p_gt_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_power48
G=diagnostics/anatomical_pose_init/out_C3_cse_head_20260826

# C14 PREREQUISITE PROBE -- NOT the C14 experiment, and NOT a deployable arm.
#
# Question: what is the CEILING on any pose initializer, under the recipe actually in use?
# Seed joint_rot with the corpus's OWN ground truth ($D/ground_truth.npz -- names (48,),
# joint_rot (48,54,3), matching optimise_hierarchical.py:377-396's contract exactly). No
# predictor can beat this, so if the gap over zero-init is small there is nothing for a
# kinematics-aware initializer to win and C14 should not be trained at all.
#
# WHY THIS IS NOT A REPEAT of the existing GT arm / the ACI_noise5..30 ladder: those ran on the
# n=12 ceiling corpus, BEFORE --cse_correspondence_from existed. n=12 is the exact power regime
# that produced the false "leg_acc never moves" claim (corrected at n=48: +0.0578, p=5.4e-05),
# and the value of a better INIT is conditional on the loss it initialises.
#
# Two arms, completing a 2x2 against runs already on disk. The two existing cells are the
# controls; nothing here re-runs them:
#
#                     | zero init          | GT init (this job)
#   ------------------+--------------------+---------------------------
#   no correspondence | P48_zero      (on disk) | C14p_gtinit_nocse
#   CSE correspondence| C13_uniform   (C13 job) | C14p_gtinit_cse
#
#   C14p_gtinit_cse   vs C13_uniform -> headroom a perfect init has ON TOP of correspondence.
#                                       This is the number that decides whether C14 is built.
#   C14p_gtinit_nocse vs P48_zero    -> what a perfect init buys WITHOUT correspondence, i.e.
#                                       whether the two mechanisms are redundant or additive.
#
# Recipe is byte-identical to C13's run_arm (same flags, same D1_low.yaml, same eval) so
# C14p_gtinit_cse is a clean paired contrast against C13_uniform on the same 48 specimens,
# through the same trainer. The ONLY differences are --init_joint_rot_from and, for the nocse
# arm, the absence of --cse_correspondence_from.
#
# NOTE: this arm reads ground truth, so it is a CEILING measurement only. It can never be
# reported as a result on real scans.

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

run_arm C14p_gtinit_cse   --init_joint_rot_from $D/ground_truth.npz \
                          --cse_correspondence_from $G/cse_p48_all.npz
run_arm C14p_gtinit_nocse --init_joint_rot_from $D/ground_truth.npz

echo "[$(date +%H:%M:%S)] C14p done"
