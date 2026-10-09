#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=RM_Q2_base
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=research/review_methods/Q2_oracle_budget/sbatch_logs/base_%A_%a.log
#SBATCH --error=research/review_methods/Q2_oracle_budget/sbatch_logs/base_%A_%a.log
# Q2 B0 (D1_PROD) and B1 (D1_PROD + deployed CSE targets in all stages, as JAB I_cse).
# Recipe byte-identical to JAB A_prod / I_cse. 2 regimes x 2 arms x 3 seeds.
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
O=/hpcwork/nao48500/review_methods/Q2
YAML=diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml
HIER_BASE="--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3"
T=$SLURM_ARRAY_TASK_ID
REGIMES=(REALPOSE REALSCALE); ARMS=(B0 B1)
R=${REGIMES[$((T / 6))]}; ARM=${ARMS[$(((T / 3) % 2))]}; SEED=$((T % 3))
M=$O/$R/meshes; X=""
[ $ARM = B1 ] && X="--cse_correspondence_from $O/$R/targets_deployed.npz"
D=$O/$R/runs/${ARM}_s${SEED}
echo "regime=$R arm=$ARM seed=$SEED head=$(git rev-parse HEAD) host=$(hostname)"
rm -rf $D ${D}_hier
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M $HIER_BASE $X --seed $SEED --results_dir ${D}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M --yaml_src $YAML $X \
  --init_from ${D}_hier/H2_joint.npz --seed $SEED --results_dir $D
echo "[$(date +%H:%M:%S)] $R $ARM s$SEED done"
