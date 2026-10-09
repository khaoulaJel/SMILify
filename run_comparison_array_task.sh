#!/usr/bin/env bash
set -uo pipefail

# ------------------------------------------------------------
# One specimen = one SLURM array task.
#
# Environment supplied by submit_comparison_array.sh:
#   RUN_DIR
#   MANIFEST
#   BLENDER
# ------------------------------------------------------------

if [[ -z "${RUN_DIR:-}" || -z "${MANIFEST:-}" || -z "${BLENDER:-}" ]]; then
    echo "ERROR: RUN_DIR, MANIFEST and BLENDER must be set."
    exit 2
fi

REPO_DIR="$HOME/SMILify"

ORIGINAL_SCRIPT="$REPO_DIR/custom_processing/prepare_antscan_data_for_mesh_fitting.py"
ALPHAWRAP_SCRIPT="$REPO_DIR/custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py"

# Blender's embedded Python does not automatically see ~/.local.
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------
# Identify this specimen
# ------------------------------------------------------------

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

INPUT_PATH="$(sed -n "${TASK_ID}p" "$MANIFEST")"

if [[ -z "$INPUT_PATH" ]]; then
    echo "ERROR: No specimen found for array task $TASK_ID"
    exit 2
fi

if [[ ! -f "$INPUT_PATH" ]]; then
    echo "ERROR: Input does not exist:"
    echo "       $INPUT_PATH"
    exit 2
fi

NAME="$(basename "$INPUT_PATH")"
NAME="${NAME%.*}"

# Make the specimen directory filesystem-safe.
# The original filename is retained in the manifest, so this is
# only for the output directory name.
SAFE_NAME="$(printf '%s' "$NAME" | sed 's/[^A-Za-z0-9_.-]/_/g')"

SPECIMEN_DIR="$RUN_DIR/specimens/${TASK_ID}_${SAFE_NAME}"

ORIGINAL_OUT="$SPECIMEN_DIR/original"
ALPHAWRAP_OUT="$SPECIMEN_DIR/alphawrap"
LOG_DIR="$SPECIMEN_DIR/logs"

mkdir -p "$ORIGINAL_OUT" "$ALPHAWRAP_OUT" "$LOG_DIR"

# ------------------------------------------------------------
# Completely isolated temporary directory
# ------------------------------------------------------------

TASK_TMP="$RUN_DIR/tmp/task_${TASK_ID}"
mkdir -p "$TASK_TMP"

export TMPDIR="$TASK_TMP"
export TEMP="$TASK_TMP"
export TMP="$TASK_TMP"

# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

{
    echo "============================================================"
    echo "SMILify original vs AlphaWrap comparison"
    echo "============================================================"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
    echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unknown}"
    echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
    echo "HOSTNAME=$(hostname)"
    echo "DATE=$(date --iso-8601=seconds)"
    echo "INPUT_PATH=$INPUT_PATH"
    echo "NAME=$NAME"
    echo "BLENDER=$BLENDER"
    echo "PYTHONPATH=$PYTHONPATH"
    echo "TMPDIR=$TMPDIR"
    echo "============================================================"
} > "$SPECIMEN_DIR/run_metadata.txt"

# Record exact repository state.
git -C "$REPO_DIR" rev-parse HEAD \
    >> "$SPECIMEN_DIR/run_metadata.txt" 2>&1 || true

git -C "$REPO_DIR" status --short \
    >> "$SPECIMEN_DIR/git_status.txt" 2>&1 || true

# Record executable versions.
"$BLENDER" --version > "$SPECIMEN_DIR/blender_version.txt" 2>&1 || true

"$REPO_DIR/custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap" \
    > "$SPECIMEN_DIR/alphawrap_version_or_help.txt" 2>&1 || true

"$REPO_DIR/custom_processing/external/ManifoldPlus/build/manifold" \
    > "$SPECIMEN_DIR/manifoldplus_version_or_help.txt" 2>&1 || true

# ------------------------------------------------------------
# Run ORIGINAL
# ------------------------------------------------------------

echo "============================================================"
echo "[$TASK_ID] ORIGINAL"
echo "Specimen: $NAME"
echo "Input:    $INPUT_PATH"
echo "Started:  $(date --iso-8601=seconds)"
echo "============================================================"

ORIGINAL_START=$(date +%s)

"$BLENDER" \
    --background \
    --python-exit-code 1 \
    --python "$ORIGINAL_SCRIPT" \
    -- "$INPUT_PATH" "$ORIGINAL_OUT" \
    > "$LOG_DIR/original.log" 2>&1

ORIGINAL_RC=$?

ORIGINAL_END=$(date +%s)
ORIGINAL_SECONDS=$((ORIGINAL_END - ORIGINAL_START))

echo "ORIGINAL_EXIT_CODE=$ORIGINAL_RC" \
    | tee "$LOG_DIR/original_status.txt"

echo "ORIGINAL_SECONDS=$ORIGINAL_SECONDS" \
    | tee -a "$LOG_DIR/original_status.txt"

# ------------------------------------------------------------
# Run ALPHAWRAP
# ------------------------------------------------------------

echo "============================================================"
echo "[$TASK_ID] ALPHAWRAP"
echo "Specimen: $NAME"
echo "Input:    $INPUT_PATH"
echo "Started:  $(date --iso-8601=seconds)"
echo "============================================================"

ALPHA_START=$(date +%s)

"$BLENDER" \
    --background \
    --python-exit-code 1 \
    --python "$ALPHAWRAP_SCRIPT" \
    -- "$INPUT_PATH" "$ALPHAWRAP_OUT" alpha_wrap_only \
    > "$LOG_DIR/alphawrap.log" 2>&1

ALPHA_RC=$?

ALPHA_END=$(date +%s)
ALPHA_SECONDS=$((ALPHA_END - ALPHA_START))

echo "ALPHAWRAP_EXIT_CODE=$ALPHA_RC" \
    | tee "$LOG_DIR/alphawrap_status.txt"

echo "ALPHAWRAP_SECONDS=$ALPHA_SECONDS" \
    | tee -a "$LOG_DIR/alphawrap_status.txt"

# ------------------------------------------------------------
# Final task summary
# ------------------------------------------------------------

{
    echo "FINAL_TASK_SUMMARY:"
    echo "task_id=$TASK_ID"
    echo "specimen=$NAME"
    echo "input=$INPUT_PATH"
    echo "original_exit_code=$ORIGINAL_RC"
    echo "original_seconds=$ORIGINAL_SECONDS"
    echo "alphawrap_exit_code=$ALPHA_RC"
    echo "alphawrap_seconds=$ALPHA_SECONDS"
    echo "specimen_dir=$SPECIMEN_DIR"
} | tee "$SPECIMEN_DIR/FINAL_TASK_SUMMARY.txt"

# We deliberately return failure if either method failed.
# The outputs/logs are retained regardless.
if [[ "$ORIGINAL_RC" -ne 0 || "$ALPHA_RC" -ne 0 ]]; then
    exit 1
fi

exit 0
