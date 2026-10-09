#!/usr/bin/env bash
#SBATCH --job-name=offset_fix_rerun22
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_fix_rerun_20260816/logs/%a.out
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_fix_rerun_20260816/logs/%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-22%6

set -eo pipefail

REPO=/rwthfs/rz/cluster/home/nao48500/SMILify
BLENDER=/home/nao48500/software/blender-4.2.23-linux-x64/blender
SCRIPT="$REPO/custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py"
RERUN_DIR="$REPO/diagnostics/offset_fix_rerun_20260816"
MANIFEST="$RERUN_DIR/manifest.tsv"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
NAME=$(echo "$LINE" | cut -f1)
INPUT=$(echo "$LINE" | cut -f2)

OUT_DIR="$RERUN_DIR/specimens/$NAME"
mkdir -p "$OUT_DIR"

TASK_TMP="$RERUN_DIR/tmp/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TASK_TMP"
export TMPDIR="$TASK_TMP"

echo "task=$SLURM_ARRAY_TASK_ID specimen=$NAME input=$INPUT"
START=$(date +%s)
"$BLENDER" --background --python-exit-code 1 --python "$SCRIPT" \
    -- "$INPUT" "$OUT_DIR" alpha_wrap_only \
    > "$RERUN_DIR/logs/${NAME}.log" 2>&1
RC=$?
END=$(date +%s)
{
    echo "specimen=$NAME exit_code=$RC seconds=$((END-START))"
} >> "$RERUN_DIR/timing_summary.txt"
exit $RC
