#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_neighbor_drag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=neighbor_drag_results/slurm_%j.log
#SBATCH --error=neighbor_drag_results/slurm_%j.log
#
# Neighbor-drag hypothesis test: local (k=2 hop, factor=0.1) vs. global-control
# (matched average strength) down-weighting of edge/laplacian smoothness around
# currently-penetrating vertices. 3 specimens (2 with the strongest observed
# neighbor-drag signal from the per-pair persistence check, 1 minority-pattern
# contrast case), full Stage_0-3 schedule matching
# fitter_3d/ants_cfg_all_penetration_gentle.yaml. See
# fitter_3d/run_neighbor_drag_experiment.py for the full experiment design.
#
# Submit FROM THE REPO ROOT:
#     mkdir -p neighbor_drag_results
#     sbatch scripts/penetration_joint_study/submit_neighbor_drag_experiment.sbatch
# Monitor:
#     squeue --me
#     tail -f neighbor_drag_results/slurm_<jobid>.log
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

# See submit_all_baseline.sbatch for the full explanation of why this is needed
# (plot_meshes()'s multiprocessing workers + OpenBLAS on a many-core node).
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p neighbor_drag_results

python -m fitter_3d.run_neighbor_drag_experiment

echo "Done -- $(date)"
