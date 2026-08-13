#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=ab_e6_tertiary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=diagnostics/morphometrics/ab_scale_cap/e6_tertiary_slurm_%j.log
#SBATCH --error=diagnostics/morphometrics/ab_scale_cap/e6_tertiary_slurm_%j.log
set -eo pipefail

# Tertiary endpoint of the §4 A/B (execution plan §4.1/§4.2): E6 correct%/median_err on full
# synth_clean, one run per arm's config. SCHEMA.md flags this as runnable independently of the
# primary/secondary full-corpus fits since synth_clean needs no external corpus access -- run
# here in parallel with the primary A/B (jobs ab_a_baseline/ab_b_scalecap) while it's compute,
# not because it changes the ship decision (§4.4's rule is genus enrichment on the real corpus).
#
# Fits synth_clean (the same hierarchical->moonshot chain run_m1_fit_all.sh uses) under both
# arm configs, tagged AB_A_SYN / AB_B_SYN so they sit in the same diagnostics/moonshot/runs/
# namespace as the primary A/B's AB_A_*/AB_B_* tags without colliding (SYN suffix, not W*/CLEAN).

CONDA_SH=/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
REPO_DIR=/p/home/jusers/jellal1/jureca/SMILify
source "$CONDA_SH"
conda activate pytorch3d
cd "$REPO_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

R=diagnostics/moonshot/runs
M=diagnostics/moonshot/synth_clean
HIER_FLAGS="--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3"

fit () { # fit <yaml_src> <tag>
  local yaml=$1 t=$2
  echo "[$(date +%H:%M:%S)] $t: hierarchical stage"
  python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
    $HIER_FLAGS \
    --results_dir $R/${t}_hier
  echo "[$(date +%H:%M:%S)] $t: moonshot stage"
  python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
    --yaml_src $yaml --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t
  echo "[$(date +%H:%M:%S)] $t done"
}

fit diagnostics/moonshot/cfg/D1_PROD.yaml          AB_A_SYN
fit diagnostics/moonshot/cfg/D1_PROD_SCALECAP.yaml AB_B_SYN

echo "[$(date +%H:%M:%S)] scoring E6 on both arms"
python -u diagnostics/moonshot/score_synth_roundtrip.py \
  --pairs AB_A_SYN:synth_clean AB_B_SYN:synth_clean --render_n 3 \
  | tee diagnostics/morphometrics/ab_scale_cap/e6_synth_clean_summary.txt

echo "[$(date +%H:%M:%S)] tertiary E6 run complete"
