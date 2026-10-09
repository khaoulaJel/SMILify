#!/usr/bin/env bash
# run_one.sh <version: V0|V1|V2> <specimen>  -- one raw AntScan STL through one script version.
set -o pipefail
V=$1; S=$2
REPO=/rwthfs/rz/cluster/home/nao48500/SMILify; D=$REPO/holotype_paper/task1_weld_fix
BLENDER=/home/nao48500/software/blender-4.2.23-linux-x64/blender
case $V in V0) SCRIPT=$D/versions/V0_master.py;; V1) SCRIPT=$D/versions/V1_fixbranch.py;; V2) SCRIPT=$D/versions/V2_investigation.py;; esac
IN=${RAW_ROOT:-/hpcwork/nao48500/antscan_data}/$S/$S.stl
OUT=/hpcwork/nao48500/holotype_task1/$V/$S; mkdir -p $OUT
export TMPDIR=/hpcwork/nao48500/holotype_task1/tmp/${V}_${S}; mkdir -p $TMPDIR
cd $D/versions; export PYTHONPATH=$D/versions:$PYTHONPATH   # V2 imports its sibling module
T0=$(date +%s)
"$BLENDER" --background --python-exit-code 1 --python "$SCRIPT" -- "$IN" "$OUT" > "$OUT/run.log" 2>&1
RC=$?
echo "version=$V specimen=$S exit=$RC seconds=$(( $(date +%s) - T0 ))" | tee -a /hpcwork/nao48500/holotype_task1/timing.txt
