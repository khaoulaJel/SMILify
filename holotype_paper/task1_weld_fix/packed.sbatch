#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=c23ms
#SBATCH --job-name=T1_weld
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=230G
#SBATCH --time=08:00:00
#SBATCH --output=holotype_paper/task1_weld_fix/sbatch_logs/packed_%j.log
#SBATCH --error=holotype_paper/task1_weld_fix/sbatch_logs/packed_%j.log
# Task 1 (PROTOCOL.md), one allocation. CPU-only Blender => default/c23ms (user-approved scoped exception).
#  1. smoke: first specimen x V0/V1/V2; abort the job if any version fails
#  2. remaining 39 x 3, largest STL first, 16 at a time, 2 h timeout per run
#  3. determinism: V0 re-run on 3 specimens into a separate dir (outputs compared afterwards)
#  4. measurements (t1_measure.py), 8 specimens at a time
set -o pipefail
cd "$SLURM_SUBMIT_DIR"
D=holotype_paper/task1_weld_fix; O=/hpcwork/nao48500/holotype_task1
mkdir -p $O; : > $O/timing.txt
export OMP_NUM_THREADS=2
echo "start $(date -Is) host=$(hostname) head=$(git rev-parse HEAD)"; sha256sum $D/versions/*.py

FIRST=$(head -1 $D/specimens.txt)
for V in V0 V1 V2; do timeout 2h bash $D/run_one.sh $V $FIRST & done; wait
if [ "$(grep -c "specimen=$FIRST exit=0" $O/timing.txt)" -ne 3 ]; then
  echo "SMOKE FAILED -- aborting"; cat $O/timing.txt; exit 1
fi
echo "smoke ok $(date -Is)"

tail -n +2 $D/specimens.txt | while read S; do echo "$(stat -c %s /hpcwork/nao48500/antscan_data/$S/$S.stl) $S"; done \
  | sort -rn | awk '{print $2}' | while read S; do for V in V0 V1 V2; do echo "$V $S"; done; done \
  | xargs -P 16 -L 1 bash -c 'timeout 2h bash '"$D"'/run_one.sh "$0" "$1" || echo "version=$0 specimen=$1 exit=TIMEOUT_OR_FAIL" >> '"$O"'/timing.txt'
echo "all runs done $(date -Is)"

for S in $(head -3 $D/specimens.txt); do
  mkdir -p $O/V0_repeat/$S
  ( cd $D/versions && /home/nao48500/software/blender-4.2.23-linux-x64/blender --background --python-exit-code 1 \
      --python V0_master.py -- /hpcwork/nao48500/antscan_data/$S/$S.stl $O/V0_repeat/$S > $O/V0_repeat/$S/run.log 2>&1 ) &
done; wait
echo "determinism reruns done $(date -Is)"

source /home/nao48500/miniforge3/etc/profile.d/conda.sh && conda activate pytorch3d
cat $D/specimens.txt | xargs -P 8 -n 1 python -u $D/t1_measure.py > $D/out/measure_all.log 2>&1
echo "measurements done $(date -Is)"
