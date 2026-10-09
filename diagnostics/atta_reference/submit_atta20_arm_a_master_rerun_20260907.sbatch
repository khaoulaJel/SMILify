#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=atta20_armA_rerun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/sbatch_logs/atta20_armA_rerun_%j.log
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/sbatch_logs/atta20_armA_rerun_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

# X2 Task 1 determinism rerun -- identical recipe to submit_atta20_arm_a_master.sbatch
# (same worktree /hpcwork/nao48500/atta20_master @ b5bf9565, same config, same 20 meshes),
# only results_dir changed, to check whether the fitter reproduces Stage_3_deform_fine.npz
# byte-for-byte on a fresh run.

WT=/hpcwork/nao48500/atta20_master
REPO=/rwthfs/rz/cluster/home/nao48500/SMILify
cd "$WT"

echo "[$(date +%H:%M:%S)] worktree HEAD: $(git rev-parse --short HEAD)"
echo "[$(date +%H:%M:%S)] model: $(grep -m1 '^SMAL_FILE' config.py)"
echo "[$(date +%H:%M:%S)] meshes: $(ls /hpcwork/nao48500/atta20/*.obj | wc -l)"

python -u -m fitter_3d.optimise \
  --mesh_dir /hpcwork/nao48500/atta20 \
  --yaml_src $REPO/diagnostics/atta_reference/cfg_arm_a_master_rerun_20260907.yaml

echo "[$(date +%H:%M:%S)] verifying npz"
python -u - <<'PY'
import numpy as np, glob, os
d = "/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/runs/ARM_A_master_rerun_20260907"
print("files:", sorted(os.path.basename(f) for f in glob.glob(d + "/*.npz")))
z = np.load(os.path.join(d, "Stage_3_deform_fine.npz"), allow_pickle=True)
print("keys:", sorted(z.files))
print("verts:", z["verts"].shape)
print("labels:", list(z["labels"]))
assert z["verts"].shape[0] == 20, f"expected 20 fits, got {z['verts'].shape[0]}"
lab = sorted(str(x).replace(".obj", "") for x in z["labels"])
assert lab == [f"{i:02d}" for i in range(1, 21)], f"labels do not match 01..20: {lab}"
print("OK")
PY
echo "[$(date +%H:%M:%S)] ARM A rerun done"
