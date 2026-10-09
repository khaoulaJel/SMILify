#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=nf4_c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/sbatch_logs/noisefloor_run4_c23g_%j.log
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/sbatch_logs/noisefloor_run4_c23g_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
WT=/hpcwork/nao48500/atta20_master
REPO=/rwthfs/rz/cluster/home/nao48500/SMILify
cd "$WT"
python -u -m fitter_3d.optimise \
  --mesh_dir /hpcwork/nao48500/atta20 \
  --yaml_src $REPO/diagnostics/atta_reference/cfg_noisefloor_run4_20260907.yaml
python -u - <<PY
import numpy as np, os
d = "/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/runs/noisefloor_run4_20260907"
z = np.load(os.path.join(d, "Stage_3_deform_fine.npz"), allow_pickle=True)
assert z["verts"].shape[0] == 20
print("OK run 4 (c23g)")
PY
