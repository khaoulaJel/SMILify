#!/bin/bash
# Intervention C (Phase C1) evaluation driver. Run after run_C_stratified_fit.sbatch (job
# 3054996, array 0-5) completes. Reuses the exact Task 3/6/Intervention-A methodology.
set -e
cd /rwthfs/rz/cluster/home/nao48500/SMILify
source /home/nao48500/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

OUT=diagnostics/registration_interventions/out
mkdir -p "$OUT"

echo "=== Metric D (final reliability): leg_distal R/bias/SNR, baseline vs each quota ==="
declare -A ARMS=(
  [pose0_q05]="SYN_clean_pose0_w5 SYN_clean_pose0_stratq05_w5 synth_clean_pose0"
  [pose0_q10]="SYN_clean_pose0_w5 SYN_clean_pose0_stratq10_w5 synth_clean_pose0"
  [pose0_q20]="SYN_clean_pose0_w5 SYN_clean_pose0_stratq20_w5 synth_clean_pose0"
  [pose25_q05]="SYN_clean_w5 SYN_clean_pose25_stratq05_w5 synth_clean"
  [pose25_q10]="SYN_clean_w5 SYN_clean_pose25_stratq10_w5 synth_clean"
  [pose25_q20]="SYN_clean_w5 SYN_clean_pose25_stratq20_w5 synth_clean"
)
for cond in pose0_q05 pose0_q10 pose0_q20 pose25_q05 pose25_q10 pose25_q20; do
  read -r base strat corpus <<< "${ARMS[$cond]}"
  echo "--- $cond ($corpus) ---"
  python diagnostics/absolute_scale/calib_features.py --runs "$base" "$strat" --primary "$base" --corpus "$corpus" \
    2>&1 | tee "$OUT/calibC_${cond}.log"
done

echo "=== Metric C (joint localization): baseline vs each quota ==="
for cond in pose0_q05 pose0_q10 pose0_q20 pose25_q05 pose25_q10 pose25_q20; do
  read -r base strat corpus <<< "${ARMS[$cond]}"
  PROBE_RUN="$strat" PROBE_CORPUS="$corpus" python diagnostics/appendage_evidence/probe_1a_joint_localization.py \
    2>&1 | tee "$OUT/jointC_${cond}_strat.log"
done

echo "=== Metric B (correspondence): baseline vs each quota, via probe_A_d2b_correspondence.py's machinery ==="
python diagnostics/registration_interventions/probe_C_d2b_correspondence.py 2>&1 | tee "$OUT/C_d2b_correspondence.log"

echo "=== DONE. See $OUT/ for logs and JSON. ==="
