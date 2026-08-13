#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=landmark_scoring
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=diagnostics/moonshot/hull/landmark_scoring_slurm_%j.log
#SBATCH --error=diagnostics/moonshot/hull/landmark_scoring_slurm_%j.log
set -eo pipefail

# Execution plan §4.5: landmark-restricted correspondence scoring, on synth_clean. CPU-only
# (HKS is a scipy cotangent-Laplacian eigendecomposition, no GPU involved) -- would ideally run
# on a CPU partition to avoid tying up a GPU node for CPU-only work, but this account (ias-7)
# has no dc-cpu association (confirmed via `sacctmgr show assoc user=jellal1`, GPU partitions
# only) -- no --gres requested, so this shouldn't block a concurrent GPU job on the same node
# if the scheduler allows sharing; if not, it queues like any other dc-gpu job.
# --run defaults to AB_A_SYN, the tertiary-E6 job's arm-A tag; if that fit hasn't landed yet
# the 'fitted' row is simply skipped (within_part_signal.py's own graceful-skip behaviour),
# HKS/spatial/random rows are unaffected since they don't need a fit.

CONDA_SH=/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
REPO_DIR=/p/home/jusers/jellal1/jureca/SMILify
source "$CONDA_SH"
conda activate pytorch3d
cd "$REPO_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

python -u diagnostics/moonshot/hull/within_part_signal_landmarks.py --n 6 --n_eig 120 --run AB_A_SYN
