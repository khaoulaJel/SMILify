#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=fig_regstages_cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=diagnostics/fig_registration_stages/sbatch_logs/run_cpu_%j.log
#SBATCH --error=diagnostics/fig_registration_stages/sbatch_logs/run_cpu_%j.log
# Task 2: D1_PROD on the 20 Atta scans, seed 0, exactly as
# PREREGISTRATION.md section 2. Figure specimen is 13.obj (pre-registered).
# DEVIATIONS.md D1: CPU execution (c23ms, account default); recipe unchanged.
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

B=diagnostics/fig_registration_stages
M=/hpcwork/nao48500/atta20
R=/hpcwork/nao48500/fig_registration_stages
YAML=diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml

echo "host=$(hostname) head=$(git rev-parse HEAD) job=$SLURM_JOB_ID"
md5sum $SMILIFY_SMAL_FILE $YAML diagnostics/moonshot/cfg/D1_PROD.yaml
sha256sum $B/PREREGISTRATION.md
git diff fitter_3d/ smal_model/ | sha256sum | sed 's/-$/code_patch(fitter_3d+smal_model)/'
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
echo "meshes: $(ls $M/*.obj | wc -l)"

rm -rf $R/D1_s0_cpu $R/D1_s0_cpu_hier
echo "[$(date +%H:%M:%S)] hierarchical"
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3 \
  --seed 0 --results_dir $R/D1_s0_cpu_hier
echo "[$(date +%H:%M:%S)] moonshot"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M --yaml_src $YAML \
  --init_from $R/D1_s0_cpu_hier/H2_joint.npz --seed 0 --results_dir $R/D1_s0_cpu --eval
echo "[$(date +%H:%M:%S)] DONE"
ls -la $R/D1_s0_cpu_hier $R/D1_s0_cpu
