#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=atta20_armA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/sbatch_logs/atta20_armA_%j.log
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/sbatch_logs/atta20_armA_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

# ARM A -- the STOCK fitter as it exists on origin/master (b5bf9565), for the head-to-head
# against ARM B (the investigation recipe, submit_atta20_fit.sbatch). Neither recipe was
# tuned on Fabian's reference CSVs, so those act as an independent target.
#
# WHY A WORKTREE. The moonshot pipeline does not exist on master and master's fitter does not
# exist in the investigation tree in runnable form (config.py differs). Running arm A from a
# detached worktree at origin/master is the only way to get the genuine stock behaviour rather
# than a reconstruction of it.
#
# THE ONLY TWO LOCAL EDITS in that worktree, both to hold the MODEL constant so the FITTER is
# the sole variable between arms:
#   1. 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl symlinked in (untracked on master).
#   2. config.py SMAL_FILE repointed from SMIL_OmniAnt.pkl to that model -- master has no
#      SMILIFY_SMAL_FILE env override, so it must be edited. Verified by sha256 that both
#      arms load a byte-identical model file.
# The optimisation config is master's own fitter_3d/ants_cfg.yaml verbatim except results_dir
# (its `args:` block overwrites CLI flags, so results_dir cannot be passed on the command line).

WT=/hpcwork/nao48500/atta20_master
REPO=/rwthfs/rz/cluster/home/nao48500/SMILify
cd "$WT"

echo "[$(date +%H:%M:%S)] worktree HEAD: $(git rev-parse --short HEAD)"
echo "[$(date +%H:%M:%S)] model: $(grep -m1 '^SMAL_FILE' config.py)"
echo "[$(date +%H:%M:%S)] meshes: $(ls /hpcwork/nao48500/atta20/*.obj | wc -l)"

python -u -m fitter_3d.optimise \
  --mesh_dir /hpcwork/nao48500/atta20 \
  --yaml_src $REPO/diagnostics/atta_reference/cfg_arm_a_master.yaml

echo "[$(date +%H:%M:%S)] verifying npz"
python -u - <<'PY'
import numpy as np, glob, os
d = "/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/atta_reference/runs/ARM_A_master"
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
echo "[$(date +%H:%M:%S)] ARM A done"
