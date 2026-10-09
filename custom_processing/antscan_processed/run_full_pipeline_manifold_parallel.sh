#!/usr/bin/env bash
# Parallel variant of run_full_pipeline_manifold_batch.sh: processes remaining specimens
# concurrently (default 3 workers - machine has 20 cores / ~10GB free RAM observed idle, and
# each Blender process runs mostly single-threaded at ~1-2 cores / <2GB steady-state, so 3
# workers leaves headroom for the largest raw-STL import/ray-cast peaks). Each specimen writes
# its own log; merge_logs.sh (or the trailing cat below) combines them into one file after.
set -uo pipefail

REPO_ROOT="/home/khaoula/SMILify"
STL_ROOT="${REPO_ROOT}/custom_processing/antscan_data"
OUT_DIR="${REPO_ROOT}/custom_processing/antscan_processed/manifold_external_full_pipeline"
PIPELINE_SCRIPT="${REPO_ROOT}/custom_processing/prepare_antscan_data_for_mesh_fitting_manifold.py"
PARALLEL_JOBS="${1:-3}"
shift || true
SPECIMENS=("$@")

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

run_one() {
    local name="$1"
    local stl_file="${STL_ROOT}/${name}/${name}.stl"
    local specimen_out_dir="${OUT_DIR}/${name}"
    local specimen_log="${OUT_DIR}/${name}.log"
    mkdir -p "$specimen_out_dir"

    {
        echo "================================================================="
        echo "SPECIMEN: $name"
        echo "  stl: $stl_file ($(du -h "$stl_file" | cut -f1))"
        echo "  started: $(date -Iseconds)"
        echo "================================================================="
    } > "$specimen_log"

    t0=$(date +%s)
    blender --background --python "$PIPELINE_SCRIPT" -- "$stl_file" "$specimen_out_dir" manifold_only \
        >> "$specimen_log" 2>&1
    t1=$(date +%s)
    echo "elapsed=$((t1-t0))s" >> "$specimen_log"
    echo "DONE: $name (elapsed=$((t1-t0))s)"
}
export -f run_one
export STL_ROOT OUT_DIR PIPELINE_SCRIPT

printf "%s\n" "${SPECIMENS[@]}" | xargs -P "$PARALLEL_JOBS" -I{} bash -c 'run_one "$@"' _ {}

echo "All parallel jobs finished."
