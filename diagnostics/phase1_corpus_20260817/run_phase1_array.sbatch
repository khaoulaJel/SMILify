#!/usr/bin/env bash
#SBATCH --job-name=phase1_corpus50
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/phase1_corpus_20260817/logs/%a.out
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/phase1_corpus_20260817/logs/%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-50%8

set -eo pipefail

REPO=/rwthfs/rz/cluster/home/nao48500/SMILify
BLENDER=/home/nao48500/software/blender-4.2.23-linux-x64/blender
SCRIPT="$REPO/custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py"
RUN_DIR="$REPO/diagnostics/phase1_corpus_20260817"
MANIFEST="$RUN_DIR/manifest.tsv"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
NAME=$(echo "$LINE" | cut -f1)
INPUT=$(echo "$LINE" | cut -f2)

OUT_DIR="$RUN_DIR/specimens/$NAME"
mkdir -p "$OUT_DIR"

TASK_TMP="$RUN_DIR/tmp/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TASK_TMP"
export TMPDIR="$TASK_TMP"

echo "task=$SLURM_ARRAY_TASK_ID specimen=$NAME input=$INPUT"
START=$(date +%s)
"$BLENDER" --background --python-exit-code 1 --python "$SCRIPT" \
    -- "$INPUT" "$OUT_DIR" alpha_wrap_only \
    > "$RUN_DIR/logs/${NAME}.log" 2>&1
RC=$?
END=$(date +%s)
{
    echo "specimen=$NAME exit_code=$RC seconds=$((END-START))"
} >> "$RUN_DIR/timing_summary.txt"
exit $RC
