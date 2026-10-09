#!/usr/bin/env bash
#SBATCH --job-name=offset_fix_sanity_dorylus
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_fix_sanity_20260816/logs/%x_%j.out
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_fix_sanity_20260816/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -eo pipefail

REPO=/rwthfs/rz/cluster/home/nao48500/SMILify
BLENDER=/home/nao48500/software/blender-4.2.23-linux-x64/blender
SCRIPT="$REPO/custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py"
SANITY_DIR="$REPO/diagnostics/offset_fix_sanity_20260816"

NAME="Dorylus_sp._CASENT0744703"
INPUT="$REPO/custom_processing/antscan_data/Dorylus_sp._CASENT0744703/Dorylus_sp._CASENT0744703.stl"
OUT_DIR="$SANITY_DIR/$NAME"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "specimen=$NAME input=$INPUT"
START=$(date +%s)
"$BLENDER" --background --python-exit-code 1 --python "$SCRIPT" \
    -- "$INPUT" "$OUT_DIR" alpha_wrap_only \
    > "$SANITY_DIR/logs/${NAME}.log" 2>&1
RC=$?
END=$(date +%s)
echo "specimen=$NAME exit_code=$RC seconds=$((END-START))" >> "$SANITY_DIR/timing_summary.txt"
