#!/usr/bin/env bash
#SBATCH --job-name=issue95_worker
#SBATCH --partition=c23ms
#SBATCH --account=default
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/issue95_fullrun/slurm_logs/%a.out
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/issue95_fullrun/slurm_logs/%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=183-272%25

set -o pipefail

REPO=/rwthfs/rz/cluster/home/nao48500/SMILify
RUNDIR="$REPO/diagnostics/issue95_fullrun"
RCLONE=/rwthfs/rz/cluster/home/nao48500/software/bin/rclone
BLENDER=/rwthfs/rz/cluster/home/nao48500/software/blender-4.2.23-linux-x64/blender
SCRIPT="$REPO/custom_processing/prepare_antscan_data_for_mesh_fitting.py"
export PYTHONPATH="/home/nao48500/.local/lib/python3.11/site-packages"

DRIVE_RAW="gdrive:UM6P_2026/DATA/mesh_registration/custom_processing/antscan_data"
DRIVE_OUT="gdrive:UM6P_2026/DATA/mesh_registration/custom_processing/antscan_processed_issue95"

NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUNDIR/manifest.txt")
if [ -z "$NAME" ]; then
    echo "No specimen name for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

LOGFILE="$RUNDIR/logs/${NAME}.log"
OUT_DIR="$RUNDIR/specimens/$NAME"
mkdir -p "$OUT_DIR"

if "$RCLONE" lsf "$DRIVE_OUT/${NAME}_processed.obj" 2>/dev/null | grep -q .; then
    echo "specimen=$NAME already on Drive, skipping" >> "$LOGFILE"
    echo "specimen=$NAME status=skipped_already_done" >> "$RUNDIR/timing_summary.txt"
    exit 0
fi

TASK_TMP="$RUNDIR/tmp/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TASK_TMP"
export TMPDIR="$TASK_TMP"

STL_PATH="$TASK_TMP/${NAME}.stl"
echo "=== [$NAME] downloading raw STL ===" > "$LOGFILE"
"$RCLONE" copy "$DRIVE_RAW/$NAME/" "$TASK_TMP/" --include "*.stl" --include "*.json" >> "$LOGFILE" 2>&1

if [ ! -f "$STL_PATH" ]; then
    echo "ERROR: raw STL not found on Drive for $NAME after download attempt" >> "$LOGFILE"
    echo "specimen=$NAME status=raw_stl_missing" >> "$RUNDIR/timing_summary.txt"
    rm -rf "$TASK_TMP"
    exit 1
fi

echo "=== [$NAME] running fixed pipeline (issue #95) ===" >> "$LOGFILE"
START=$(date +%s)
"$BLENDER" --background --python-exit-code 1 --python "$SCRIPT" \
    -- "$STL_PATH" "$OUT_DIR" >> "$LOGFILE" 2>&1
RC=$?
END=$(date +%s)
echo "specimen=$NAME exit_code=$RC seconds=$((END-START))" >> "$RUNDIR/timing_summary.txt"

OUT_OBJ="$OUT_DIR/${NAME}_processed.obj"
if [ $RC -eq 0 ] && [ -f "$OUT_OBJ" ]; then
    echo "=== [$NAME] uploading result + log to Drive ===" >> "$LOGFILE"
    "$RCLONE" copy "$OUT_OBJ" "$DRIVE_OUT/" >> "$LOGFILE" 2>&1
    if [ -f "$OUT_DIR/${NAME}.json" ]; then
        "$RCLONE" copy "$OUT_DIR/${NAME}.json" "$DRIVE_OUT/" >> "$LOGFILE" 2>&1
    fi
    "$RCLONE" copy "$LOGFILE" "$DRIVE_OUT/logs/" >> "$LOGFILE" 2>&1
    echo "specimen=$NAME status=uploaded" >> "$RUNDIR/timing_summary.txt"
else
    echo "WARNING: [$NAME] failed (exit=$RC) or no output produced - uploading log for diagnosis" >> "$LOGFILE"
    "$RCLONE" copy "$LOGFILE" "$DRIVE_OUT/logs/failed/" >> "$LOGFILE" 2>&1
    echo "specimen=$NAME status=failed_exit${RC}" >> "$RUNDIR/timing_summary.txt"
fi

rm -rf "$TASK_TMP"
exit $RC
