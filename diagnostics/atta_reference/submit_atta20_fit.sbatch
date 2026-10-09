#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=atta20_fit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=diagnostics/atta_reference/sbatch_logs/atta20_%j.log
#SBATCH --error=diagnostics/atta_reference/sbatch_logs/atta20_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

# Fit the 20 Atta size-series specimens, to replicate Fabian's original morphometric
# export (diagnostics/atta_reference/measurements/) with the current model.
#
# RECIPE. Identical to the production M1 recipe in
# diagnostics/morphometrics/run_m1_fit_all.sh: D1_PROD.yaml, hierarchical H0-H2 first,
# then moonshot initialised from H2_joint.npz. Deliberately NOT the moonshot-only path
# used by the Y-series probes -- the reference CSVs are a production artifact, so the
# replication must use the production recipe or a difference in output cannot be
# attributed to the model rather than the recipe.
#
# w_scale (scale_cap) is NOT enabled, matching run_m1_fit_all.sh's default. See that
# script's header: the candidate has not cleared its pre-registered full-corpus A/B.
#
# SCOPE. 20 meshes only (01.obj..20.obj), per Fabian's instruction to fit "just the 20
# Atta specimens". The same Drive folder holds all 81 ALL_ANTS_CLEAN meshes, but the
# other 61 have no reference body length and would export unscaled.
#
# OUTPUT. diagnostics/moonshot/runs/ATTA20/Stage_3_deform_fine.npz, which carries
# `verts` (20, nverts, 3) and `labels` ("01".."20"). That is what the Blender addon's
# create_shapekeys() consumes when shapekeys_from_PCA is OFF -- one shape key per
# specimen, named to match the Shape column of ant_body_lengths.csv.

M=/hpcwork/nao48500/atta20
R=diagnostics/moonshot/runs
HIER_FLAGS="--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3"

echo "[$(date +%H:%M:%S)] meshes: $(ls $M/*.obj | wc -l)"

echo "[$(date +%H:%M:%S)] stage 1/2 -- hierarchical H0-H2"
python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir $M $HIER_FLAGS --seed 0 \
  --results_dir $R/ATTA20_hier

echo "[$(date +%H:%M:%S)] stage 2/2 -- moonshot"
python -u -m fitter_3d.optimise_moonshot \
  --mesh_dir $M \
  --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml \
  --seed 0 \
  --init_from $R/ATTA20_hier/H2_joint.npz \
  --results_dir $R/ATTA20

echo "[$(date +%H:%M:%S)] verifying npz"
python -u - <<'PY'
import numpy as np
d = np.load("diagnostics/moonshot/runs/ATTA20/Stage_3_deform_fine.npz", allow_pickle=True)
print("keys:", sorted(d.files))
print("verts:", d["verts"].shape)
print("labels:", list(d["labels"]))
assert d["verts"].shape[0] == 20, f"expected 20 fits, got {d['verts'].shape[0]}"
assert sorted(str(x) for x in d["labels"]) == [f"{i:02d}.obj" for i in range(1, 21)] or \
       sorted(str(x) for x in d["labels"]) == [f"{i:02d}" for i in range(1, 21)], \
       f"labels do not match 01..20: {list(d['labels'])}"
print("OK")
PY
echo "[$(date +%H:%M:%S)] ATTA20 done"
