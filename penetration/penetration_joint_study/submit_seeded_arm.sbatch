#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_pen_seeded
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=fit3d_results_seeded/slurm_%j.log
#SBATCH --error=fit3d_results_seeded/slurm_%j.log
#
# Reseeded rerun of the step-1 pose-freeze ablation (see FINDINGS.md), with
# two corrections folded in vs. the original run: w_limit now active in
# Stage_2/Stage_3 (not just Stage_1), and --seed for reproducible multi-seed
# variance estimates. One ARM x SEED combination per job.
#
# Submit FROM THE REPO ROOT, once per (arm, seed) combination:
#     mkdir -p fit3d_results_seeded
#     for arm in baseline gentle; do
#       for seed in 0 1 2; do
#         ARM=$arm SEED=$seed sbatch --export=ALL,ARM,SEED \
#           scripts/penetration_joint_study/submit_seeded_arm.sbatch
#       done
#     done
# Monitor:
#     squeue --me
#     tail -f fit3d_results_seeded/slurm_<jobid>.log
# Cancel:
#     scancel <jobid>
#
# NOTE: intentionally no set -u -- conda's activate hooks reference unset vars
# (e.g. MKL_INTERFACE_LAYER) and would abort under set -u.
set -eo pipefail

: "${ARM:?Set ARM=baseline or ARM=gentle via --export}"
: "${SEED:?Set SEED=<int> via --export}"

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
REPO_DIR="${REPO_DIR:-.}"
CONDA_SH="${CONDA_SH:-/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-pytorch3d}"

case "$ARM" in
  baseline) YAML="fitter_3d/ants_cfg_all_baseline.yaml" ;;
  gentle)   YAML="fitter_3d/ants_cfg_all_penetration_gentle.yaml" ;;
  *) echo "ARM must be 'baseline' or 'gentle', got '$ARM'" >&2; exit 1 ;;
esac
RESULTS_DIR="fit3d_results_seeded/${ARM}_seed${SEED}"

echo "Job $SLURM_JOB_ID on $(hostname) -- $(date) -- ARM=$ARM SEED=$SEED -> $RESULTS_DIR"

# See submit_all_baseline.sbatch for the full explanation.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Multiple seeded jobs run concurrently against this same shared conda env --
# plot_meshes()'s multiprocessing workers reimporting torch/numpy/matplotlib
# via 'spawn' under that concurrent load was observed to race on the shared
# filesystem and hang (partially-initialized-module ImportErrors, never
# returning). This rerun doesn't need the mesh PNGs, so skip them entirely.
export SMIL_DISABLE_PLOTTING=1

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p "$RESULTS_DIR"

python -m fitter_3d.optimise --yaml_src "$YAML" --results_dir "$RESULTS_DIR" --seed "$SEED" --require_seed

echo "Done -- $(date)"
