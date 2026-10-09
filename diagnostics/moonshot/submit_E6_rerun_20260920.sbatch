#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c25g
#SBATCH --job-name=E6_rerun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/moonshot/sbatch_logs/E6_rerun_%j.log
#SBATCH --error=diagnostics/moonshot/sbatch_logs/E6_rerun_%j.log
set -eo pipefail
# Re-run of E6's adopted-recipe arm (SYN_D1) on the 12-specimen ground-truth corpus, ONLY to
# regenerate fit geometry for a slide visual (slide 25a). The original SYN_D1 run directory was
# purged; only ground_truth.npz, the corpus meshes and two labelled PNGs survive. Chain and
# arguments are copied from diagnostics/moonshot/run_e8b_synth.sh (arm SYN_D1, D1_SYN.yaml, --offset 30).
# NOTE: fitter_3d/trainer_moonshot.py etc. carry uncommitted edits since E6, so this reproduces
# the recipe on the CURRENT code -- re-measure correctness here, do not quote E6's 6.69% for it.
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
T=SYN_D1_rerun
echo "[$(date +%H:%M:%S)] hierarchical stage"
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --results_dir $R/${T}_hier
echo "[$(date +%H:%M:%S)] moonshot stages"
python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
    --yaml_src diagnostics/moonshot/cfg/D1_SYN.yaml --init_from $R/${T}_hier/H2_joint.npz \
    --results_dir $R/$T
echo "[$(date +%H:%M:%S)] E6 rerun done -> $R/$T"
