#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=RM_Q3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-50
#SBATCH --output=research/review_methods/Q3d_identity_weight_sweep/sbatch_logs/q3d_%A_%a.log
#SBATCH --error=research/review_methods/Q3d_identity_weight_sweep/sbatch_logs/q3d_%A_%a.log
# Q3d (PREREGISTRATION.md): lambda x target-quality sweep, D1_PROD recipe, targets in all stages.
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
CFG=()
for Q in rho0.0 rho0.2 rho0.4; do for L in 0.1 0.25 0.5 2; do CFG+=("$Q:$L"); done; done
for L in 0.1 0.25 0.5 1 2; do CFG+=("nonleg0.2:$L"); done
C=${CFG[$((SLURM_ARRAY_TASK_ID / 3))]}; S=$((SLURM_ARRAY_TASK_ID % 3))
Q=${C%%:*}; L=${C##*:}
M=diagnostics/moonshot/synth_power48; O=/hpcwork/nao48500/review_methods/Q3d
T=/hpcwork/nao48500/review_methods/Q3b_S1/targets_${Q}.npz
YAML=diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml
HIER_BASE="--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3"
R=$O/runs/${Q}_lam${L}_s${S}
echo "quality=$Q lambda=$L seed=$S head=$(git rev-parse HEAD) host=$(hostname)"; sha256sum $T
rm -rf $R ${R}_hier
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M $HIER_BASE --cse_correspondence_from $T \
  --w_cse_correspondence $L --seed $S --results_dir ${R}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M --yaml_src $YAML --cse_correspondence_from $T \
  --w_cse_correspondence $L --init_from ${R}_hier/H2_joint.npz --seed $S --results_dir $R
