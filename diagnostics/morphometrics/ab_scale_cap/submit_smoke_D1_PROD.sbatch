#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smoke_D1_PROD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/morphometrics/ab_scale_cap/smoke_D1_PROD_slurm_%j.log
#SBATCH --error=diagnostics/morphometrics/ab_scale_cap/smoke_D1_PROD_slurm_%j.log
set -eo pipefail

# Task C: smoke-test the FINAL, w_scale-carrying D1_PROD.yaml with SHIPPED_RECIPE.md's exact
# documented command, on 5 real ALL_ANTS_CLEAN specimens, before trusting it for full refits.
CONDA_SH=/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
REPO_DIR=/p/home/jusers/jellal1/jureca/SMILify
source "$CONDA_SH"
conda activate pytorch3d
cd "$REPO_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

M=diagnostics/morphometrics/ab_scale_cap/smoke_D1_PROD_mesh
R=diagnostics/morphometrics/ab_scale_cap/smoke_D1_PROD_out
t=smoke

python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3 \
  --results_dir $R/${t}_hier

python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
  --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml \
  --init_from $R/${t}_hier/H2_joint.npz \
  --results_dir $R/$t

echo "[$(date +%H:%M:%S)] smoke test complete"
python -c "
import numpy as np
d = np.load('$R/$t/Stage_3_deform_fine.npz')
print('verts:', d['verts'].shape, 'labels:', list(d['labels']))
print('log_beta_scales finite:', np.isfinite(d['log_beta_scales']).all())
"
