#!/bin/bash
# Handles Google Drive traffic for the ENHANCED antscan pipeline
# (run_antscan_enhanced.sh / prepare_antscan_data_for_mesh_fitting_enhanced.py).
# Run this on the JURECA LOGIN node (not submitted as a batch job) --
# compute nodes on the dc-gpu partition have no internet route. Meant to be
# run inside `tmux`/`screen` so it keeps running after you disconnect:
#
#   tmux new -s antscan-sync-enhanced
#   bash sync_antscan_drive_enhanced.sh
#   [Ctrl-b d to detach; tmux attach -t antscan-sync-enhanced to come back]
#
# Input STL files are shared with the original pipeline (same antscan_data
# on scratch); this script still does its own initial pull in case the
# original sync_antscan_drive.sh isn't already running. Output goes to a
# SEPARATE Drive folder (antscan_processed_enhanced) rather than
# antscan_processed, since the enhanced algorithm's results aren't a
# drop-in match for what the old script already produced there.

export PATH="$HOME/bin:$PATH"   # rclone lives here

RCLONE_REMOTE="gdrive"
REMOTE_DATA_PATH="UM6P_2026/DATA/mesh_registration/custom_processing/antscan_data"
REMOTE_OUT_PATH="UM6P_2026/DATA/mesh_registration/custom_processing/antscan_processed_enhanced"

WORK_DIR="/p/scratch/cias-7/jellal1/antscan"
DATA_DIR="$WORK_DIR/antscan_data"
OUT_DIR="$WORK_DIR/antscan_processed_enhanced"
SYNC_INTERVAL=300   # seconds between upload passes

mkdir -p "$DATA_DIR" "$OUT_DIR"

echo "Syncing input STL files from Google Drive..."
rclone copy "$RCLONE_REMOTE:$REMOTE_DATA_PATH" "$DATA_DIR" --progress

echo "Syncing already-processed enhanced outputs from Google Drive (so the batch job can skip them)..."
rclone copy "$RCLONE_REMOTE:$REMOTE_OUT_PATH" "$OUT_DIR" --progress

echo "Initial sync done. Now watching for new outputs to push back every ${SYNC_INTERVAL}s (Ctrl-C to stop)..."
while true; do
    sleep "$SYNC_INTERVAL"
    echo "$(date): syncing processed outputs back to Google Drive..."
    rclone copy "$OUT_DIR" "$RCLONE_REMOTE:$REMOTE_OUT_PATH"
done
