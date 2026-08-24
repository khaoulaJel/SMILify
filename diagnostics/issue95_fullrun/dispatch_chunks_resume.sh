#!/usr/bin/env bash
RUNDIR=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/issue95_fullrun
TEMPLATE="$RUNDIR/run_issue95_array.sbatch"
CHUNK_SIZE=90
START=363
TOTAL=602
LOG="$RUNDIR/dispatch.log"
echo "$(date): dispatcher started, start=$START total=$TOTAL chunk_size=$CHUNK_SIZE" >> "$LOG"
for ((lo=START; lo<=TOTAL; lo+=CHUNK_SIZE)); do
    hi=$((lo + CHUNK_SIZE - 1))
    if [ "$hi" -gt "$TOTAL" ]; then hi=$TOTAL; fi
    CHUNK_SBATCH="$RUNDIR/tmp_chunk_${lo}_${hi}.sbatch"
    sed "s/--array=1-602%25/--array=${lo}-${hi}%25/" "$TEMPLATE" > "$CHUNK_SBATCH"
    echo "$(date): submitting chunk ${lo}-${hi}" >> "$LOG"
    JOBID=$(sbatch --parsable "$CHUNK_SBATCH" 2>>"$LOG")
    if [ -z "$JOBID" ]; then
        echo "$(date): FAILED to submit chunk ${lo}-${hi}, aborting dispatcher" >> "$LOG"
        exit 1
    fi
    echo "$(date): chunk ${lo}-${hi} submitted as job $JOBID" >> "$LOG"
    while squeue --job "$JOBID" -h 2>/dev/null | grep -q .; do
        sleep 60
    done
    echo "$(date): chunk ${lo}-${hi} (job $JOBID) fully drained from queue" >> "$LOG"
done
echo "$(date): dispatcher finished all chunks" >> "$LOG"
