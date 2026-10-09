#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/khaoula/SMILify"
STL_ROOT="${REPO_ROOT}/custom_processing/antscan_data"
OUT_DIR="${REPO_ROOT}/custom_processing/antscan_processed/manifold_external_reprocess_v2"
PIPELINE_SCRIPT="${REPO_ROOT}/custom_processing/prepare_antscan_data_for_mesh_fitting_manifold.py"

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

run_one() {
    name="$1"
    stl_file="${STL_ROOT}/${name}/${name}.stl"
    specimen_out_dir="${OUT_DIR}/${name}"
    log="${OUT_DIR}/${name}.log"
    mkdir -p "$specimen_out_dir"

    echo "SPECIMEN: $name started $(date -Iseconds)" > "$log"
    t0=$(date +%s)
    blender --background --python "$PIPELINE_SCRIPT" -- "$stl_file" "$specimen_out_dir" manifold_only \
        2>&1 | grep -v "^Progress\|Color management\|non-existent directory" >> "$log"
    t1=$(date +%s)
    echo "elapsed=$((t1-t0))s" >> "$log"
    echo "DONE: $name (elapsed=$((t1-t0))s)" >> "${OUT_DIR}/driver.log"
}

run_one "Cephalotes_depressus_CASENT0744569" &
run_one "Syllophopsis_sechellensis_OKENT0105212" &
wait
echo "All reprocessing jobs finished $(date -Iseconds)" >> "${OUT_DIR}/driver.log"
