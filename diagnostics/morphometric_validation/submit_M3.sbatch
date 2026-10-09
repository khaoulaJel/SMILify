#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=M3_repeat
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/morphometric_validation/sbatch_logs/M3_%j.log
#SBATCH --error=diagnostics/morphometric_validation/sbatch_logs/M3_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
D=diagnostics/moonshot/synth_power48
C=diagnostics/morphometric_validation
for s in 1 2; do
  echo "[$(date +%H:%M:%S)] M3 seed $s"
  python -u -m fitter_3d.optimise_moonshot --mesh_dir $D --yaml_src $C/M3_seed$s.yaml --seed $s
done
echo "[$(date +%H:%M:%S)] M3 fits done"
