#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$HOME/SMILify"

BLENDER="${BLENDER:-$HOME/software/blender-4.2.23-linux-x64/blender}"
DATA_DIR="${DATA_DIR:-$HOME/SMILify/diagnostics/moonshot/bench50_clean}"

# Maximum number of specimens running simultaneously.
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/comparison_output}"

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

if [[ ! -x "$BLENDER" ]]; then
    echo "ERROR: Blender not executable:"
    echo "       $BLENDER"
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: DATA_DIR does not exist:"
    echo "       $DATA_DIR"
    exit 1
fi

if [[ ! -x "$REPO_DIR/custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap" ]]; then
    echo "ERROR: AlphaWrap binary not found/executable."
    exit 1
fi

if [[ ! -x "$REPO_DIR/custom_processing/external/ManifoldPlus/build/manifold" ]]; then
    echo "ERROR: ManifoldPlus binary not found/executable."
    exit 1
fi

# ------------------------------------------------------------
# Create a genuinely unique run directory.
# mktemp prevents accidental collisions.
# ------------------------------------------------------------

mkdir -p "$OUTPUT_ROOT"

RUN_DIR="$(mktemp -d "$OUTPUT_ROOT/bench50_clean_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

MANIFEST="$RUN_DIR/manifest.tsv"

mkdir -p "$RUN_DIR/specimens"
mkdir -p "$RUN_DIR/tmp"
mkdir -p "$RUN_DIR/slurm_logs"

# ------------------------------------------------------------
# Build deterministic specimen manifest.
# ------------------------------------------------------------

find "$DATA_DIR" -type f \
    \( -iname "*.stl" -o -iname "*.obj" \) \
    | sort > "$MANIFEST"

N=$(wc -l < "$MANIFEST")

if [[ "$N" -eq 0 ]]; then
    echo "ERROR: No .stl/.obj files found under:"
    echo "       $DATA_DIR"
    rm -rf "$RUN_DIR"
    exit 1
fi

# ------------------------------------------------------------
# Record experiment metadata before submission.
# ------------------------------------------------------------

{
    echo "SMILify original vs AlphaWrap comparison"
    echo
    echo "run_dir=$RUN_DIR"
    echo "data_dir=$DATA_DIR"
    echo "blender=$BLENDER"
    echo "repo=$REPO_DIR"
    echo "specimen_count=$N"
    echo "max_concurrent=$MAX_CONCURRENT"
    echo
    echo "git_commit:"
    git -C "$REPO_DIR" rev-parse HEAD
    echo
    echo "git_status:"
    git -C "$REPO_DIR" status --short
    echo
    echo "blender:"
    "$BLENDER" --version
    echo
    echo "alphawrap_binary:"
    ls -lh "$REPO_DIR/custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap"
    echo
    echo "manifoldplus_binary:"
    ls -lh "$REPO_DIR/custom_processing/external/ManifoldPlus/build/manifold"
    echo
    echo "created=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
} > "$RUN_DIR/experiment_metadata.txt"

# ------------------------------------------------------------
# Show exactly what will run.
# ------------------------------------------------------------

echo
echo "============================================================"
echo "SMILify comparison experiment"
echo "============================================================"
echo "Run directory:"
echo "  $RUN_DIR"
echo
echo "Data:"
echo "  $DATA_DIR"
echo
echo "Specimens:"
echo "  $N"
echo
echo "Concurrent tasks:"
echo "  $MAX_CONCURRENT"
echo
echo "Blender:"
echo "  $BLENDER"
echo
echo "Manifest:"
echo "  $MANIFEST"
echo "============================================================"
echo

echo "First specimens:"
head -10 "$MANIFEST"
echo

# ------------------------------------------------------------
# Submit array.
#
# One task = one specimen.
# %MAX_CONCURRENT limits simultaneous tasks.
# ------------------------------------------------------------

JOB_ID=$(
    sbatch \
        --parsable \
        --array="1-${N}%${MAX_CONCURRENT}" \
        --mem=8G \
        --output="$RUN_DIR/slurm_logs/%A_%a.out" \
        --error="$RUN_DIR/slurm_logs/%A_%a.err" \
        --job-name="smil_compare" \
        --export="ALL,RUN_DIR=$RUN_DIR,MANIFEST=$MANIFEST,BLENDER=$BLENDER" \
        "$REPO_DIR/run_comparison_array_task.sh"
)

echo "Submitted job array: $JOB_ID"
echo
echo "Run directory:"
echo "$RUN_DIR"
echo
echo "Check with:"
echo "  squeue -j $JOB_ID"
echo
echo "After completion:"
echo "  cat $RUN_DIR/experiment_metadata.txt"
echo "  find $RUN_DIR/specimens -name FINAL_TASK_SUMMARY.txt -print"
echo
