#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=ab_b_scalecap
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/morphometrics/ab_scale_cap/ab_b_slurm_%j.log
#SBATCH --error=diagnostics/morphometrics/ab_scale_cap/ab_b_slurm_%j.log
set -eo pipefail

# Arm B of the §4 full-corpus scale_cap A/B (execution plan, SCHEMA.md): D1_PROD_SCALECAP.yaml,
# candidate, w_scale: 0.052. Same corpus/provenance as arm A (submit_ab_a.sbatch) -- only the
# yaml_src and tag prefix differ, so any difference in output is attributable to scale_cap
# alone, not a corpus or scheduling confound.
#
# Same 6h QOS wall-time cap as arm A (ias-7/dc-gpu, "nocont" QOS) -- if this hits the limit,
# just `sbatch` it again; see submit_ab_a.sbatch's comment for why that's safe (recipe-
# fingerprint skip-if-exists resumes from the last completed chunk).
cd "$SLURM_SUBMIT_DIR"

CONDA_SH=/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh \
REPO_DIR="$SLURM_SUBMIT_DIR" \
W=/p/scratch/cias-7/jellal1/SMILify_DATA/custom_processing/antscan_proofread_castes/worker_ALT \
CLEAN=/p/scratch/cias-7/jellal1/SMILify_DATA/ALL_ANTS_CLEAN \
STAGE=diagnostics/morphometrics/ab_scale_cap/stage_B \
TAG_PREFIX=AB_B_ \
YAML_SRC=diagnostics/moonshot/cfg/D1_PROD_SCALECAP.yaml \
FORCE_REFIT=1 \
bash diagnostics/morphometrics/run_m1_fit_all.sh
