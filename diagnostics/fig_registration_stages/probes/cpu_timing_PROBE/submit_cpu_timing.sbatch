#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --job-name=fig_cpu_timing
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/fig_registration_stages/probes/cpu_timing_PROBE/timing_%j.log
#SBATCH --error=diagnostics/fig_registration_stages/probes/cpu_timing_PROBE/timing_%j.log
# DEVIATIONS.md D1, step 1. Timing probe ONLY: 10 iterations per stage, D1_PROD otherwise unchanged.
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH" SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
P=diagnostics/fig_registration_stages/probes/cpu_timing_PROBE
R=/hpcwork/nao48500/fig_registration_stages/cpu_timing_PROBE
echo "host=$(hostname) head=$(git rev-parse HEAD) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
lscpu | grep "Model name"
rm -rf $R; python -c "import torch;print('threads',torch.get_num_threads(),'cuda',torch.cuda.is_available())"
t0=$(date +%s)
python -u -m fitter_3d.optimise_hierarchical --mesh_dir /hpcwork/nao48500/atta20 \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3 \
  --body_its 10 --leg_its 10 --joint_its 10 --seed 0 --results_dir $R/hier
t1=$(date +%s)
python -u -m fitter_3d.optimise_moonshot --mesh_dir /hpcwork/nao48500/atta20 --yaml_src $P/A_prod_nits10.yaml \
  --init_from $R/hier/H2_joint.npz --seed 0 --results_dir $R/moon --eval
t2=$(date +%s)
echo "TIMING hier_30its_s=$((t1-t0)) moon_20its_plus_eval_s=$((t2-t1))"
