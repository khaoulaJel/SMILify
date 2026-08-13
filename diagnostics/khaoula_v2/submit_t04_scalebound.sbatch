#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=t04_scalebound
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/khaoula_v2/runs/t04_slurm_%j.log
#SBATCH --error=diagnostics/khaoula_v2/runs/t04_slurm_%j.log
set -eo pipefail

# T0.4 -- validate beta-scale/translation bounding (joint_limits.py scale_barrier/trans_barrier)
# through the same E6 harness as T0.1/T0.2. Two SEPARATE arms, not combined: scale_cap has one
# externally-calibrated candidate weight (w_scale=0.052, from diagnostics/moonshot/cfg/M7_scale1x.yaml);
# trans_cap has NO calibrated candidate anywhere in the prior work, so w_trans=1.0 here is an
# uncalibrated exploratory probe, not a validated value -- keep that distinction in the readout.
# Each arm reruns its own hierarchical stage rather than reusing a T0.1/T0.2 H2_joint.npz
# checkpoint: those checkpoints were confirmed NOT byte-identical across arms despite identical
# flags (md5 differs across T01_off2_hier/T01_off10_hier/T01_off20_hier/T02_restedge_hier), so
# the hierarchical stage is not deterministic here and reusing a foreign checkpoint would be an
# uncontrolled confound.
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
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
    --yaml_src $CFG/${cfg}.yaml --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1 || { echo "[$t] MOONSHOT FAILED"; return 1; }
  echo "[$(date +%H:%M:%S)] $t done"
}

arm 0 T04_scale T04_scale &
arm 1 T04_trans T04_trans &
wait
echo "[$(date +%H:%M:%S)] T0.4 scale/trans bounding sweep complete"
