#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=A3_offnorm
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/morphometric_validation/sbatch_logs/A3_%j.log
#SBATCH --error=diagnostics/morphometric_validation/sbatch_logs/A3_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
D=diagnostics/moonshot/synth_power48
C=diagnostics/morphometric_validation
echo "[$(date +%H:%M:%S)] arm A -- D1_PROD as shipped"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D --yaml_src $C/A3_prod.yaml --seed 0
echo "[$(date +%H:%M:%S)] arm B -- w_offset=0 w_normal=0"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D --yaml_src $C/A3_nooffnorm.yaml --seed 0
echo "[$(date +%H:%M:%S)] analysis"
python -u $C/a3_analyse.py
echo "[$(date +%H:%M:%S)] A3 done"
