#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=HA_d1cse
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/anatomical_pose_init/sbatch_logs/HA_d1cse_%j.log
#SBATCH --error=diagnostics/anatomical_pose_init/sbatch_logs/HA_d1cse_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_power48
G=diagnostics/anatomical_pose_init/out_C3_cse_head_20260826

# H_A of PREREGISTRATION_D1_coxal_degradation_20260828.md: D1 previously had NO correspondence
# term of any kind. This arm carries it through, from the SAME handoff and the SAME correspondence
# file C13_uniform used, so the only variable is whether the term survives into D1.
#
# The port was verified BYTE-IDENTICAL when off before this ran: 20/20 arrays across both D1
# stages, full 48 specimens, bit-for-bit against a stashed baseline.
#
# GUARD-RAIL, restated: any result better than H2_joint's own co=0.2564 is a SUSPECTED BUG, not a
# win. D1 has strictly more information than H2; the honest expectation is "hold what H2 found".
# Note the term acts on the DEFORMED surface here (scheme: all optimises deform_verts) but acted
# on the posed-only surface in the hierarchical arms (--deform_its 0), so a suspiciously good
# result most likely means the term is being satisfied by free-form deformation rather than by
# anatomy -- check deform_verts magnitude before believing it.

echo "[$(date +%H:%M:%S)] D1_with_cse"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from $R/C13_uniform_hier/H2_joint.npz \
  --cse_correspondence_from $G/cse_p48_all.npz \
  --results_dir $R/D1_with_cse
python -u diagnostics/moonshot/eval_run.py --run_dir $R/D1_with_cse --mesh_dir $D
echo "[$(date +%H:%M:%S)] H_A done"
