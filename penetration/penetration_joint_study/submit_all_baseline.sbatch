#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_pen_all_baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=penetration_joint_study_results/slurm_baseline_%j.log
#SBATCH --error=penetration_joint_study_results/slurm_baseline_%j.log
#
# Penetration x unfrozen-pose joint experiment -- BASELINE arm (w_penetration=0,
# scheme:'all' in the deform stages). Runs on the real bench50_clean specimen set.
# See fitter_3d/ants_cfg_all_baseline.yaml for the full stage/weight config.
#
# Submit FROM THE REPO ROOT (the #SBATCH --output path is resolved at submit
# time, so the results dir must exist before sbatch):
#     mkdir -p penetration_joint_study_results
#     sbatch scripts/penetration_joint_study/submit_all_baseline.sbatch
# Monitor:
#     squeue --me
#     tail -f penetration_joint_study_results/slurm_baseline_<jobid>.log
# Cancel:
#     scancel <jobid>
#
# NOTE: intentionally no set -u -- conda's activate hooks reference unset vars
# (e.g. MKL_INTERFACE_LAYER) and would abort under set -u.
set -eo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
REPO_DIR="${REPO_DIR:-.}"
CONDA_SH="${CONDA_SH:-/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-pytorch3d}"

echo "Job $SLURM_JOB_ID on $(hostname) -- $(date)"

# ---- avoid the OpenBLAS/pthread_create thread-storm seen on the login node --
# fitter_3d/utils.py's plot_meshes() spawns one multiprocessing worker per
# specimen (up to cpu_count()); dc-gpu nodes report cpu_count()=256 same as
# the login node. Each worker reimporting numpy/torch would otherwise try to
# spin up ~128 OpenBLAS threads of its own -- with several workers doing that
# concurrently the per-user pthread_create budget (ulimit -u) can be
# exhausted even under a SLURM allocation. Capping each process to a single
# BLAS/OMP thread removes the failure mode outright, independent of how many
# workers get spawned. This is standard practice whenever numpy/torch is
# combined with multiprocessing on a many-core node.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p penetration_joint_study_results

python -m fitter_3d.optimise --yaml_src fitter_3d/ants_cfg_all_baseline.yaml

echo "Done -- $(date)"
