#!/bin/bash
# Handles all Google Drive traffic for the antscan pipeline. Run this on
# the JURECA LOGIN node (not submitted as a batch job) -- compute nodes on
# the dc-gpu partition have no internet route, so all rclone calls have to
# happen here instead. Meant to be run inside `tmux`/`screen` so it keeps
# running after you disconnect:
#
#   tmux new -s antscan-sync
#   bash sync_antscan_drive.sh
#   [Ctrl-b d to detach; tmux attach -t antscan-sync to come back]
#
# It does one initial full sync down, then repeatedly pushes newly
# processed outputs back up to Drive every $SYNC_INTERVAL seconds so
# progress from the running/queued batch job is continuously backed up.
# Stop it with Ctrl-C once the batch job has finished.

export PATH="$HOME/bin:$PATH"   # rclone lives here

RCLONE_REMOTE="gdrive"
REMOTE_DATA_PATH="UM6P_2026/DATA/mesh_registration/custom_processing/antscan_data"
REMOTE_OUT_PATH="UM6P_2026/DATA/mesh_registration/custom_processing/antscan_processed"

WORK_DIR="/p/scratch/cias-7/jellal1/antscan"
DATA_DIR="$WORK_DIR/antscan_data"
OUT_DIR="$WORK_DIR/antscan_processed"
SYNC_INTERVAL=300   # seconds between upload passes

mkdir -p "$DATA_DIR" "$OUT_DIR"

echo "Syncing input STL files from Google Drive..."
rclone copy "$RCLONE_REMOTE:$REMOTE_DATA_PATH" "$DATA_DIR" --progress

echo "Syncing already-processed outputs from Google Drive (so the batch job can skip them)..."
rclone copy "$RCLONE_REMOTE:$REMOTE_OUT_PATH" "$OUT_DIR" --progress

echo "Initial sync done. Now watching for new outputs to push back every ${SYNC_INTERVAL}s (Ctrl-C to stop)..."
while true; do
    sleep "$SYNC_INTERVAL"
    echo "$(date): syncing processed outputs back to Google Drive..."
    rclone copy "$OUT_DIR" "$RCLONE_REMOTE:$REMOTE_OUT_PATH"
done
