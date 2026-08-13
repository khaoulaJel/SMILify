#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=ab_a_baseline
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/morphometrics/ab_scale_cap/ab_a_slurm_%j.log
#SBATCH --error=diagnostics/morphometrics/ab_scale_cap/ab_a_slurm_%j.log
set -eo pipefail

# Arm A of the §4 full-corpus scale_cap A/B (execution plan, SCHEMA.md): D1_PROD.yaml,
# baseline, no scale_cap. 757 worker_ALT + 81 ALL_ANTS_CLEAN, synced from the shared
# UM6P_2026 Google Drive to /p/scratch/cias-7/jellal1/SMILify_DATA 2026-08-13 (see
# SCHEMA.md's "Precondition" section for provenance/verification).
#
# QOS on this account/partition (ias-7/dc-gpu, "nocont") caps wall-time at 6h -- confirmed via
# `sacctmgr show qos`, not assumed. This job likely will NOT finish 838 specimens x
# hierarchical+moonshot in one 6h slot (rough estimate: several hours per arm). If it hits the
# wall-time limit: just `sbatch` this same script again. run_m1_fit_all.sh's fit() stamps a
# recipe fingerprint per completed chunk and skips it on rerun, so a resubmit resumes from
# the last fully-completed chunk rather than restarting from scratch (only the chunk that was
# mid-flight when the walltime hit gets redone).
cd "$SLURM_SUBMIT_DIR"

CONDA_SH=/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh \
REPO_DIR="$SLURM_SUBMIT_DIR" \
W=/p/scratch/cias-7/jellal1/SMILify_DATA/custom_processing/antscan_proofread_castes/worker_ALT \
CLEAN=/p/scratch/cias-7/jellal1/SMILify_DATA/ALL_ANTS_CLEAN \
STAGE=diagnostics/morphometrics/ab_scale_cap/stage_A \
TAG_PREFIX=AB_A_ \
YAML_SRC=diagnostics/moonshot/cfg/D1_PROD.yaml \
FORCE_REFIT=1 \
bash diagnostics/morphometrics/run_m1_fit_all.sh
