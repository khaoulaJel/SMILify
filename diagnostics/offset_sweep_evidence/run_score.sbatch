#!/bin/bash
#SBATCH --job-name=offset_sweep_score
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/offset_sweep_evidence/logs/score_%j.out

set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify

source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

python -u diagnostics/moonshot/score_synth_roundtrip.py \
  --pairs SYN_clean_w2_5:synth_clean SYN_clean_w5:synth_clean SYN_clean_w10:synth_clean SYN_clean_w20:synth_clean \
          SYN_noisy_w2_5:synth_noisy SYN_noisy_w5:synth_noisy SYN_noisy_w10:synth_noisy SYN_noisy_w20:synth_noisy \
  2>&1 | tee diagnostics/offset_sweep_evidence/logs/score_synth_roundtrip.log

echo "SCORE DONE"
