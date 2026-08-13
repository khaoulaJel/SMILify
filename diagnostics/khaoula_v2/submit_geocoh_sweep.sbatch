#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=geocoh_sweep
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/khaoula_v2/runs/geocoh_slurm_%j.log
#SBATCH --error=diagnostics/khaoula_v2/runs/geocoh_slurm_%j.log
set -eo pipefail

# Lever B: geodesic motion-coherence sweep (w_geocoh in {0.1, 1.0, 10.0}), on synth_clean,
# same hierarchical->moonshot chain as every other Tier-0/Tier-1 GPU run. Uses
# optimise_geocoherence.py (fork of optimise_moonshot.py) so trainer_moonshot.py stays
# untouched.
source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/khaoula_v2/runs
D=diagnostics/moonshot/synth_clean
CFG=diagnostics/khaoula_v2/cfg

arm () {
  local g=$1 t=$2 cfg=$3
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
    --results_dir $R/${t}_hier >$R/$t.log 2>&1 || { echo "[$t] HIER FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/khaoula_v2/optimise_geocoherence.py --mesh_dir $D \
    --yaml_src $CFG/${cfg}.yaml --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1 || { echo "[$t] MOONSHOT FAILED"; return 1; }
  echo "[$(date +%H:%M:%S)] $t done"
}

arm 0 GEOCOH_01 GEOCOH_01 &
arm 1 GEOCOH_1  GEOCOH_1 &
arm 2 GEOCOH_10 GEOCOH_10 &
wait
echo "[$(date +%H:%M:%S)] geocoh sweep complete"
