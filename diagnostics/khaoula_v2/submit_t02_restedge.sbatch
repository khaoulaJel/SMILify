#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=t02_restedge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/khaoula_v2/runs/t02_slurm_%j.log
#SBATCH --error=diagnostics/khaoula_v2/runs/t02_slurm_%j.log
set -eo pipefail

source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/khaoula_v2/runs
D=diagnostics/moonshot/synth_clean
CFG=diagnostics/khaoula_v2/cfg
t=T02_restedge

python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --results_dir $R/${t}_hier >$R/$t.log 2>&1 || { echo "[$t] HIER FAILED"; exit 1; }
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
  --yaml_src $CFG/T02_restedge.yaml --init_from $R/${t}_hier/H2_joint.npz \
  --results_dir $R/$t >>$R/$t.log 2>&1 || { echo "[$t] MOONSHOT FAILED"; exit 1; }
echo "[$(date +%H:%M:%S)] $t done"
