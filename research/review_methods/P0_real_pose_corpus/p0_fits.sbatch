#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=RM_P0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=research/review_methods/P0_real_pose_corpus/sbatch_logs/p0_%A_%a.log
#SBATCH --error=research/review_methods/P0_real_pose_corpus/sbatch_logs/p0_%A_%a.log
# P0 (PREREGISTRATION.md): D1_PROD (JAB A_prod recipe, unchanged) on worker_ALT minus all JAB genera.
# tasks 0-13: chunk cNN, seed 0.  tasks 14-15: stability subset stab48, seeds 1 and 2.
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
O=/hpcwork/nao48500/review_methods/P0
YAML=diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml
HIER_BASE="--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3"
T=$SLURM_ARRAY_TASK_ID
if [ $T -lt 14 ]; then C=$(printf "c%02d" $T); SEED=0; else C=stab48; SEED=$((T - 13)); fi
M=$O/chunks/$C; D=$O/runs/${C}_s${SEED}
echo "chunk=$C seed=$SEED n=$(ls $M | wc -l) head=$(git rev-parse HEAD) host=$(hostname)"
rm -rf $D ${D}_hier
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M $HIER_BASE --seed $SEED --results_dir ${D}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M --yaml_src $YAML --init_from ${D}_hier/H2_joint.npz \
  --seed $SEED --results_dir $D
echo "[$(date +%H:%M:%S)] $C s$SEED done"
