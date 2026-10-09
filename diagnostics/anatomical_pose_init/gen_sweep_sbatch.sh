#!/usr/bin/env bash
# Generates one sbatch script per pose-noise-sweep arm + the capacity-ceiling arm, and submits
# them all. Run from repo root: bash diagnostics/anatomical_pose_init/gen_sweep_sbatch.sh
set -eo pipefail
cd "$(dirname "$0")/../.."
OUT=diagnostics/anatomical_pose_init/out_ceiling_20260820
SB=diagnostics/anatomical_pose_init/sweep_sbatch
mkdir -p "$SB" "$OUT/runs_log"

gen_pose_job () {
  local t=$1 init=$2
  cat > "$SB/submit_${t}.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=${t}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=$OUT/runs_log/${t}_slurm_%j.log
#SBATCH --error=$OUT/runs_log/${t}_slurm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "\$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
python -u -m fitter_3d.optimise_hierarchical --mesh_dir \$D \\
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 \\
  --init_joint_rot_from ${init} \\
  --results_dir \$R/${t}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir \$D \\
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from \$R/${t}_hier/H2_joint.npz \\
  --results_dir \$R/${t}
echo "[\$(date +%H:%M:%S)] ${t} done"
EOF
  sbatch "$SB/submit_${t}.sbatch"
}

# Primary: pose-error sweep, GT + fixed-magnitude leg-joint noise at each level.
for L in 5 10 15 20 25 30; do
  gen_pose_job "ACI_noise${L}" "$OUT/pose_noise/noise${L}.npz"
done

# Secondary: capacity ceiling. GT-pose init (exact ground_truth.npz) + scale_cap=0.052
# (D1_PROD.yaml, matches the shipped scale_cap barrier -- "the strongest capacity setting
# already identified", diagnostics/moonshot/cfg/D1_PROD.yaml). Compared against the EXISTING
# SYN_clean_pose25_gtinit_w5 run (GT-pose + free/uncapped segment scale, D1_low.yaml) -- not
# re-run. See RESULTS_v2.md for why this mapping (free-scale-already-exists, need only the
# capped-scale comparison arm).
cat > "$SB/submit_ACI_gtcapacitycap.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --account=rwth2151
#SBATCH --partition=c23g
#SBATCH --job-name=ACI_gtcap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=$OUT/runs_log/ACI_gtcapacitycap_slurm_%j.log
#SBATCH --error=$OUT/runs_log/ACI_gtcapacitycap_slurm_%j.log
set -eo pipefail
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
cd "\$SLURM_SUBMIT_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
t=ACI_gtcapacitycap
python -u -m fitter_3d.optimise_hierarchical --mesh_dir \$D \\
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --deform_its 0 --scale_cap 0.052 \\
  --init_joint_rot_from \$D/ground_truth.npz \\
  --results_dir \$R/\${t}_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir \$D \\
  --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml --init_from \$R/\${t}_hier/H2_joint.npz \\
  --results_dir \$R/\$t
echo "[\$(date +%H:%M:%S)] \$t done"
EOF
sbatch "$SB/submit_ACI_gtcapacitycap.sbatch"

echo "submitted 7 jobs"
