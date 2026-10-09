#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=fig_regstages_cpu1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=1-20
#SBATCH --output=diagnostics/fig_registration_stages/sbatch_logs/run_cpu1_%A_%a.log
#SBATCH --error=diagnostics/fig_registration_stages/sbatch_logs/run_cpu1_%A_%a.log
# DEVIATIONS.md D2: same D1_PROD commands as submit.sbatch, one specimen per array task (batch 1)
# on CPU, because the 20-batch CPU fit is single-thread bound (~4 of 48 cores busy).
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

S=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
M=/hpcwork/nao48500/fig_registration_stages/single_meshes/$S
R=/hpcwork/nao48500/fig_registration_stages/D1_s0_cpu1/$S
YAML=diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml

echo "host=$(hostname) head=$(git rev-parse HEAD) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} specimen=$S"
md5sum $SMILIFY_SMAL_FILE $YAML $M/$S.obj
sha256sum diagnostics/fig_registration_stages/PREREGISTRATION.md
git diff fitter_3d/ smal_model/ | sha256sum | sed 's/-$/code_patch(fitter_3d+smal_model)/'

rm -rf $R; mkdir -p $R
echo "[$(date +%H:%M:%S)] hierarchical"
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3 \
  --seed 0 --results_dir $R/hier
echo "[$(date +%H:%M:%S)] moonshot"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M --yaml_src $YAML \
  --init_from $R/hier/H2_joint.npz --seed 0 --results_dir $R/moon --eval
echo "[$(date +%H:%M:%S)] DONE"
