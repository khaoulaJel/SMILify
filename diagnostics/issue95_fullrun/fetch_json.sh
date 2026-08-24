#!/usr/bin/env bash
RUNDIR=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/issue95_fullrun
RCLONE=/usr/bin/rclone
for i in 1 2 3 4 5; do
    "$RCLONE" copy "gdrive:UM6P_2026/DATA/mesh_registration/custom_processing/antscan_data/" "$RUNDIR/json_probe/" \
        --include "*.json" --max-depth 2 -q --transfers 16 --checkers 16 >> "$RUNDIR/fetch_json.log" 2>&1
    n=$(find "$RUNDIR/json_probe" -name "*.json" | wc -l)
    echo "$(date): pass $i done, count=$n" >> "$RUNDIR/fetch_json.log"
    if [ "$n" -ge 780 ]; then break; fi
done
echo "$(date): fetch_json finished" >> "$RUNDIR/fetch_json.log"
