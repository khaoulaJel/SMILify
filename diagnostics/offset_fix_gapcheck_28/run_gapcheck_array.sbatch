#!/usr/bin/env bash
#SBATCH --job-name=gapcheck28
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_fix_gapcheck_28/logs/%a.out
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_fix_gapcheck_28/logs/%a.err
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-27%8

set -eo pipefail

GAPDIR=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_fix_gapcheck_28
BLENDER=/home/nao48500/software/blender-4.2.23-linux-x64/blender

python3 - "$SLURM_ARRAY_TASK_ID" << 'PYEOF' > /tmp/gapcheck_args_${SLURM_ARRAY_TASK_ID}.txt
import json, sys
idx = int(sys.argv[1])
with open("/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_fix_gapcheck_28/manifest_full.json") as f:
    m = json.load(f)
name, alpha, prerecon = m[idx]
print(f"{name}\t{alpha}\t{prerecon}")
PYEOF

IFS=$'\t' read -r NAME ALPHA PRERECON < /tmp/gapcheck_args_${SLURM_ARRAY_TASK_ID}.txt

echo "specimen=$NAME alpha=$ALPHA"
"$BLENDER" --background --python-exit-code 1 --python "$GAPDIR/gapcheck_blender2.py" \
    -- "$NAME" "$ALPHA" "$PRERECON" "$GAPDIR/results/${NAME}.json"
