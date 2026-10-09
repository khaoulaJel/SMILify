#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=Z8_full757
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=diagnostics/full_corpus/sbatch_logs/Z8_%j.log
#SBATCH --error=diagnostics/full_corpus/sbatch_logs/Z8_%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

# Z8 -- fit ALL 757 workers under the PRODUCTION recipe, then measure trait repeatability at
# full power. Z7 ranked traits from 25 conspecific pairs; the full corpus has 191 species with
# >=2 specimens (484 specimens), which turns a usable ranking into a usable estimate of each R.
#
# Uses diagnostics/morphometrics/run_m1_fit_all.sh -- the production fitter chain the
# morphometrics report itself is built on (hierarchical H0-H2 init, then moonshot with
# D1_PROD.yaml), NOT the bare optimise_moonshot call the Z2-Z6 diagnostics used. Repeatability
# has to be measured on the pipeline whose outputs the report actually uses.
#
# SKIP_CLEAN=1: ALL_ANTS_CLEAN is not on this cluster and is not the corpus of interest.

echo "[$(date +%H:%M:%S)] fitting 757 workers, production recipe, 12 chunks over 2 GPUs"
CONDA_SH=/home/nao48500/miniforge3/etc/profile.d/conda.sh \
REPO_DIR=$SLURM_SUBMIT_DIR \
W=/hpcwork/nao48500/worker_alt_data \
CLEAN=/dev/null \
SKIP_CLEAN=1 \
TAG_PREFIX=Z8_ \
STAGE=diagnostics/full_corpus/stage \
bash diagnostics/morphometrics/run_m1_fit_all.sh

echo "[$(date +%H:%M:%S)] fits complete; scoring repeatability at full power"
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
python -u diagnostics/full_corpus/z8_repeatability_full.py
echo "[$(date +%H:%M:%S)] Z8 done"
