#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=t04_bench50
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/khaoula_v2/runs/t04_bench50_slurm_%j.log
#SBATCH --error=diagnostics/khaoula_v2/runs/t04_bench50_slurm_%j.log
set -eo pipefail

# T0.4 follow-up, REAL corpus -- the actual test of scale_cap's design target.
# synth_clean cannot reproduce the 138x/389x/889x anterior-scale pathology
# FINAL_REPORT measured (it's generated from the model's own well-behaved geometry),
# so that check (diagnostics/khaoula_v2/runs/T04_baseline vs T04_scale) was
# inconclusive by construction. bench50_clean is real ethanol-preserved worker scans
# -- the regime the pathology was actually measured in. No E6 ground truth needed:
# this is a log_beta_scales magnitude read, not a correspondence metric.
source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "$SLURM_SUBMIT_DIR"

export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/khaoula_v2/runs
D=diagnostics/moonshot/bench50_clean
CFG=diagnostics/khaoula_v2/cfg

arm () {
  local g=$1 t=$2 cfg=$3
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
    --results_dir $R/${t}_hier >$R/$t.log 2>&1 || { echo "[$t] HIER FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
    --yaml_src ${cfg} --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1 || { echo "[$t] MOONSHOT FAILED"; return 1; }
  echo "[$(date +%H:%M:%S)] $t done"
}

arm 0 T04_bench_baseline diagnostics/moonshot/cfg/D1_SYN.yaml &
arm 1 T04_bench_scale    $CFG/T04_scale.yaml &
wait
echo "[$(date +%H:%M:%S)] T0.4 bench50 anterior check complete"
