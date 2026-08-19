#!/bin/bash
# Intervention A evaluation driver. Run after run_A_splitdistal_fit.sbatch (job array 0-3)
# completes -- check with `sacct -j <jobid>` or that
# diagnostics/moonshot/runs/SYN_clean_{pose0,pose25,drop30,drop60}_splitdistal_w5/
# Stage_3_deform_fine.npz all exist.
#
# Metric A (leg_distal R/bias/SNR): calib_features.py, exact Task 3/6 methodology, baseline vs
#   split_distal, same corpus each pair.
# Metric B/joint localization: reuses probe_1a_joint_localization.py (appendage_evidence) via
#   PROBE_RUN/PROBE_CORPUS env vars.
# Metric C correspondence: extend probe_d2b's `conditions` list (edited in place, see NOTE) --
#   run manually, see below, since split_distal is a snapshot audit at leg-granularity that
#   requires no code change to accept new run names.
set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

OUT=diagnostics/registration_interventions/out
mkdir -p "$OUT"

echo "=== Metric A: leg_distal R/bias/SNR, baseline vs split_distal ==="
declare -A PAIRS=(
  [pose0]="SYN_clean_pose0_w5 SYN_clean_pose0_splitdistal_w5 synth_clean_pose0"
  [pose25]="SYN_clean_w5 SYN_clean_pose25_splitdistal_w5 synth_clean"
  [drop30]="SYN_clean_drop30_w5 SYN_clean_drop30_splitdistal_w5 synth_clean_drop30"
  [drop60]="SYN_clean_drop60_w5 SYN_clean_drop60_splitdistal_w5 synth_clean_drop60"
)
for cond in pose0 pose25 drop30 drop60; do
  read -r base split corpus <<< "${PAIRS[$cond]}"
  echo "--- $cond ($corpus) ---"
  python diagnostics/absolute_scale/calib_features.py --runs "$base" "$split" --primary "$base" --corpus "$corpus" \
    2>&1 | tee "$OUT/calib_${cond}.log"
done

echo "=== Metric B: joint localization, baseline vs split_distal ==="
for cond in pose0 pose25 drop30 drop60; do
  read -r base split corpus <<< "${PAIRS[$cond]}"
  PROBE_RUN="$base" PROBE_CORPUS="$corpus" python diagnostics/appendage_evidence/probe_1a_joint_localization.py \
    2>&1 | tee "$OUT/joint_${cond}_baseline.log"
  PROBE_RUN="$split" PROBE_CORPUS="$corpus" python diagnostics/appendage_evidence/probe_1a_joint_localization.py \
    2>&1 | tee "$OUT/joint_${cond}_splitdistal.log"
done

echo "=== Metric C: correspondence audit (cross-leg / within-leg), baseline vs split_distal ==="
python diagnostics/registration_interventions/probe_A_d2b_correspondence.py 2>&1 | tee "$OUT/A_d2b_correspondence.log"

echo "=== DONE. See $OUT/ for logs and JSON. ==="
