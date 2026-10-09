#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=RM_Q3b_S1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-20
#SBATCH --output=research/review_methods/Q3b_controlled_shift/sbatch_logs/s1_%A_%a.log
#SBATCH --error=research/review_methods/Q3b_controlled_shift/sbatch_logs/s1_%A_%a.log
# Q3b S1 (PREREGISTRATION.md). D1_PROD = JAB A_prod recipe, exactly; only the target file and the
# stages it is passed to change. 7 arms x 3 seeds = 21 tasks.
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
M=diagnostics/moonshot/synth_power48
O=/hpcwork/nao48500/review_methods/Q3b_S1
YAML=diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml
HIER_BASE="--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3"
ARMS=(prod full_rho0.0 full_rho0.2 full_rho0.4 hier_rho0.0 hier_rho0.2 hier_rho0.4)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
HX=""; MX=""
case $ARM in
  full_rho*) T=$O/targets_${ARM#full_}.npz; HX="--cse_correspondence_from $T"; MX="--cse_correspondence_from $T" ;;
  hier_rho*) T=$O/targets_${ARM#hier_}.npz; HX="--cse_correspondence_from $T" ;;
esac
R=$O/runs/${ARM}_s${SEED}
echo "arm=$ARM seed=$SEED head=$(git rev-parse HEAD) host=$(hostname) targets=${T:-none}"
sha256sum $YAML $SMILIFY_SMAL_FILE ${T:-/dev/null}
rm -rf $R ${R}_hier
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M $HIER_BASE $HX --seed $SEED --results_dir ${R}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M --yaml_src $YAML $MX \
  --init_from ${R}_hier/H2_joint.npz --seed $SEED --results_dir $R
echo "[$(date +%H:%M:%S)] $ARM s$SEED done"
