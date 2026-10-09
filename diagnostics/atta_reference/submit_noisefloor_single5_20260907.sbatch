#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=nf5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/sbatch_logs/noisefloor_run5_single_%j.log
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/sbatch_logs/noisefloor_run5_single_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

WT=/hpcwork/nao48500/atta20_master
REPO=/rwthfs/rz/cluster/home/nao48500/SMILify
cd "$WT"

python -u -m fitter_3d.optimise \
  --mesh_dir /hpcwork/nao48500/atta20 \
  --yaml_src $REPO/diagnostics/atta_reference/cfg_noisefloor_run5_20260907.yaml

python -u - <<PY
import numpy as np, glob, os
d = "/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/runs/noisefloor_run5_20260907"
z = np.load(os.path.join(d, "Stage_3_deform_fine.npz"), allow_pickle=True)
assert z["verts"].shape[0] == 20
lab = sorted(str(x).replace(".obj", "") for x in z["labels"])
assert lab == [f"{j:02d}" for j in range(1, 21)]
print("OK run 5")
PY
