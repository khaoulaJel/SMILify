#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$HOME/SMILify"

RUN_DIR="$REPO_DIR/comparison_output/bench50_clean_20260814_125245_Glc7oS"
MANIFEST="$RUN_DIR/manifest.tsv"
BLENDER="$HOME/software/blender-4.2.23-linux-x64/blender"

FAILED_TASKS="11,12,13,14,15,20,22,23,24,25,26,28,31,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49"

sbatch \
    --array="$FAILED_TASKS%8" \
    --mem=5G \
    --time=30:00 \
    --output="$RUN_DIR/slurm_logs/retry_%A_%a.out" \
    --error="$RUN_DIR/slurm_logs/retry_%A_%a.err" \
    --job-name="smil_retry" \
    --export="ALL,RUN_DIR=$RUN_DIR,MANIFEST=$MANIFEST,BLENDER=$BLENDER" \
    "$REPO_DIR/run_comparison_array_task.sh"
