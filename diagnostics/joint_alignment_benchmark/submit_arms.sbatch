#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=JAB_arms
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=diagnostics/joint_alignment_benchmark/sbatch_logs/arm_%A_%a.log
#SBATCH --error=diagnostics/joint_alignment_benchmark/sbatch_logs/arm_%A_%a.log
# Joint-Alignment Benchmark: one array task per strategy arm, seeds 0,1,2 inside.
# Every arm differs from A in exactly the factor listed in PREREGISTRATION.md section 6.
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0

B=diagnostics/joint_alignment_benchmark
M=/hpcwork/nao48500/jab_stage
R=/hpcwork/nao48500/jab_runs
ARMS=(A_prod B_nohier C_nopartition D_nolimits E_anteriorlimits F_noscalecap G_allometric \
      H_learnedinit I_cse J_nooffnorm K_posefrozen L_splitdistal)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
MODEL=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
[ "$ARM" = E_anteriorlimits ] && MODEL=3D_model_prep/OmniAnt_25PCs_anterior_limited.pkl
export SMILIFY_SMAL_FILE=$MODEL

HIER_BASE="--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3"
HIER_EXTRA=""; MOON_EXTRA=""; YAML=$B/cfg/A_prod.yaml; DO_HIER=1
case $ARM in
  B_nohier)       DO_HIER=0 ;;
  C_nopartition)  HIER_EXTRA="--no_partition" ;;
  D_nolimits)     HIER_BASE="--midline 2.0 --beta_prior 0.0 --limit 0.0 --offset 30.0 --skip_h3"; YAML=$B/cfg/D_nolimits.yaml ;;
  F_noscalecap)   YAML=$B/cfg/F_noscalecap.yaml ;;
  G_allometric)   YAML=$B/cfg/G_allometric.yaml ;;
  H_learnedinit)  INIT=$R/inputs/learned_init.npz
                  mkdir -p $R/inputs
                  [ -f $INIT ] || python -u $B/tools/make_learned_init.py $M $INIT
                  HIER_EXTRA="--init_joint_rot_from $INIT" ;;
  I_cse)          CSE=$R/inputs/cse_all.npz
                  mkdir -p $R/inputs
                  [ -f $CSE ] || python -u diagnostics/anatomical_pose_init/generate_cse_correspondence_20260827.py \
                      --mesh_dir $M --normalise --filter cycle --keep_frac 0.5 --segments all --out $CSE
                  HIER_EXTRA="--cse_correspondence_from $CSE"; MOON_EXTRA="--cse_correspondence_from $CSE" ;;
  J_nooffnorm)    YAML=$B/cfg/J_nooffnorm.yaml ;;
  K_posefrozen)   YAML=$B/cfg/K_posefrozen.yaml ;;
  L_splitdistal)  HIER_EXTRA="--split_distal" ;;
esac

echo "arm=$ARM model=$MODEL yaml=$YAML host=$(hostname) head=$(git rev-parse HEAD)"
sha256sum $YAML $MODEL
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

for SEED in 0 1 2; do
  T=$R/${ARM}_s${SEED}
  rm -rf ${T} ${T}_hier
  echo "[$(date +%H:%M:%S)] $ARM seed $SEED"
  if [ $DO_HIER = 1 ]; then
    python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M $HIER_BASE $HIER_EXTRA \
      --seed $SEED --results_dir ${T}_hier
    python -u -m fitter_3d.optimise_moonshot --mesh_dir $M --yaml_src $YAML $MOON_EXTRA \
      --init_from ${T}_hier/H2_joint.npz --seed $SEED --results_dir $T
  else
    python -u -m fitter_3d.optimise_moonshot --mesh_dir $M --yaml_src $YAML $MOON_EXTRA \
      --seed $SEED --results_dir $T
  fi
  echo "[$(date +%H:%M:%S)] $ARM seed $SEED done"
done
echo "[$(date +%H:%M:%S)] ARM $ARM COMPLETE"
