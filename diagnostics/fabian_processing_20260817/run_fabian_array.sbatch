#!/usr/bin/env bash
#SBATCH --job-name=fabian_processing50
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/fabian_processing_20260817/logs/%a.out
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/fabian_processing_20260817/logs/%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-50%10

set -eo pipefail

REPO=/rwthfs/rz/cluster/home/nao48500/SMILify
BLENDER=/home/nao48500/software/blender-4.2.23-linux-x64/blender
SCRIPT="$REPO/custom_processing/fabian_processing.py"
FABIAN_DIR="$REPO/diagnostics/fabian_processing_20260817"
STL_DIR="$REPO/diagnostics/fabian_stl_inputs_20260817"

NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$FABIAN_DIR/manifest.txt")
INPUT="$STL_DIR/${NAME}.stl"
OUT_DIR="$FABIAN_DIR/specimens/$NAME"
mkdir -p "$OUT_DIR"

TASK_TMP="$FABIAN_DIR/tmp/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TASK_TMP"
export TMPDIR="$TASK_TMP"

echo "task=$SLURM_ARRAY_TASK_ID specimen=$NAME input=$INPUT"
START=$(date +%s)
"$BLENDER" --background --python-exit-code 1 --python "$SCRIPT" \
    -- "$INPUT" "$OUT_DIR" \
    > "$FABIAN_DIR/logs/${NAME}.log" 2>&1
RC=$?
END=$(date +%s)
echo "specimen=$NAME exit_code=$RC seconds=$((END-START))" >> "$FABIAN_DIR/timing_summary.txt"
exit $RC
