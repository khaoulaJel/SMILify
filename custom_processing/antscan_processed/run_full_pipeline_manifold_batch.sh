#!/usr/bin/env bash
# Runs the FULL integrated pipeline (raw STL -> clean -> decimate -> external Manifold repair ->
# final export) via prepare_antscan_data_for_mesh_fitting_manifold.py's "manifold_only" mode, on
# all 10 specimens with raw STL available under custom_processing/antscan_data/. One combined log.
set -uo pipefail

REPO_ROOT="/home/khaoula/SMILify"
STL_ROOT="${REPO_ROOT}/custom_processing/antscan_data"
OUT_DIR="${REPO_ROOT}/custom_processing/antscan_processed/manifold_external_full_pipeline"
LOG="${OUT_DIR}/batch_log.txt"
PIPELINE_SCRIPT="${REPO_ROOT}/custom_processing/prepare_antscan_data_for_mesh_fitting_manifold.py"

mkdir -p "$OUT_DIR"
: > "$LOG"

echo "FULL PIPELINE (manifold_external) BATCH RUN - started $(date -Iseconds)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

cd "$REPO_ROOT"

SUMMARY_ROWS=()

for stl_dir in "$STL_ROOT"/*/; do
    name=$(basename "$stl_dir")
    stl_file="${stl_dir}${name}.stl"
    if [ ! -f "$stl_file" ]; then
        continue
    fi

    specimen_out_dir="${OUT_DIR}/${name}"
    mkdir -p "$specimen_out_dir"

    echo "=================================================================" | tee -a "$LOG"
    echo "SPECIMEN: $name" | tee -a "$LOG"
    echo "  stl: $stl_file ($(du -h "$stl_file" | cut -f1))" | tee -a "$LOG"
    echo "  started: $(date -Iseconds)" | tee -a "$LOG"
    echo "=================================================================" | tee -a "$LOG"

    t0=$(date +%s)
    blender --background --python "$PIPELINE_SCRIPT" -- "$stl_file" "$specimen_out_dir" manifold_only \
        2>&1 | grep -v "^Progress\|Color management\|non-existent directory" | tee -a "$LOG"
    t1=$(date +%s)
    elapsed=$((t1-t0))
    echo "elapsed=${elapsed}s" | tee -a "$LOG"

    final_line=$(grep "^FINAL_SUMMARY: specimen=${name} " "$LOG" | tail -1)
    if [ -n "$final_line" ]; then
        SUMMARY_ROWS+=("$name | OK | elapsed=${elapsed}s | $final_line")
    else
        SUMMARY_ROWS+=("$name | FAILED (no FINAL_SUMMARY line) | elapsed=${elapsed}s")
    fi
    echo "" | tee -a "$LOG"
done

echo "=================================================================" | tee -a "$LOG"
echo "BATCH SUMMARY - finished $(date -Iseconds)" | tee -a "$LOG"
echo "=================================================================" | tee -a "$LOG"
for row in "${SUMMARY_ROWS[@]}"; do
    echo "$row" | tee -a "$LOG"
done

echo "Done. Combined log: $LOG"
