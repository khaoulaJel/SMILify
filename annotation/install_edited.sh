#!/usr/bin/env bash
# Install edited annotations from annotation/incoming/ into annotation/gt_expert/.
# QC runs FIRST and installs nothing if it finds an unplaced point or an ordering violation.
set -eo pipefail
cd "$(dirname "$0")/.."
N=$(find annotation/incoming -name '*_joints.json' | wc -l)
[ "$N" -eq 0 ] && { echo "nothing in annotation/incoming/"; exit 1; }
echo "$N incoming files"
source /home/nao48500/miniforge3/etc/profile.d/conda.sh && conda activate pytorch3d
TMP=$(mktemp -d); cp -r annotation/gt_expert "$TMP/gt_expert_backup"
find annotation/incoming -name '*_joints.json' -exec cp {} annotation/gt_expert/ \;
if python annotation/qc_annotations.py | tee /dev/stderr | grep -q 'ISSUE'; then
  echo; echo "QC found issues above. The old files are backed up at $TMP/gt_expert_backup"
  echo "Review, then either fix the source or accept. 'git checkout -- annotation/gt_expert' reverts."
else
  echo; echo "QC clean. Installed."
fi
