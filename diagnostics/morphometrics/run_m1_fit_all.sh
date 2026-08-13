#!/bin/bash
# M1 — refit BOTH corpora under one consistent recipe, so morphometrics is never comparing
# specimens that were fitted differently. DO NOT EDIT WHILE RUNNING.
#
# WHY REFIT AT ALL. The fits already on disk are a patchwork: `CLEAN_M7` (81 ALL_ANTS_CLEAN)
# used the M7 recipe, `D1_low_MERGED` (128 gated workers) used D1. The headline morphometric
# question -- do the same genera give the same proportions in the two corpora? -- is only
# answerable if the recipe is held constant, otherwise a corpus difference and a recipe
# difference are perfectly confounded.
#
# RECIPE. D1_PROD (diagnostics/moonshot/cfg/D1_PROD.yaml), i.e. D1 (handoff `w_offset`
# 5.0/2.0 -- E8b measured this as the best available setting on the direct ground-truth
# round trip, 6.69% correct vs the control's 4.81%, median vertex error 3.21% vs 3.48%,
# REPORT_E8.md §4 adopts it as the default). Previously this pointed at D1_low.yaml, which
# has a known defect (joint limits silently off in the handoff stage, `w_limit` absent) --
# D1_PROD is D1_SYN.yaml (the corrected recipe used throughout the 2026-08-11 validation
# work), not D1_low with a patch bolted on.
#
# `w_scale: 0.052` (scale_cap) is DELIBERATELY NOT in this default recipe. It is validated
# on a magnitude metric and directional-only genus signal (diagnostics/V2_CHANGELOG.md /
# diagnostics/khaoula_v2/REPORT_V2.md, 2026-08-11) but has not cleared the pre-registered
# full-corpus A/B (execution plan §4, diagnostics/morphometrics/ab_scale_cap/SCHEMA.md).
# That candidate lives in diagnostics/moonshot/cfg/D1_PROD_SCALECAP.yaml (arm B of that A/B)
# -- do not point --yaml_src at it here until the A/B passes its decision rule.
#
# NOTE ON `--offset` (H0-H2 hierarchical stage): H3_deform's npz is not read by the moonshot
# handoff (REPORT_E8.md §3.1), so this flag would otherwise be inert. `--skip_h3` below makes
# that explicit (0 iterations) instead of relying on `--offset`'s value being harmless.
#
# SCOPE. ALL 757 workers, not the 379 that pass the fittability gate. Two reasons: taxonomic
# coverage of the shared genera is what limits the cross-corpus test, and keeping the ungated
# half lets the gate itself be evaluated as a morphometric quality filter post hoc rather than
# assumed. Gating is a column in the output table, not a precondition of the run.
# CONDA_SH/REPO_DIR are overridable: this script originally assumed Fabian's own machine.
# The 757+81 corpus turned out to be reachable after all -- not at /media/fabi/Data (that
# path is genuinely local-machine-only) but on the shared UM6P_2026 Google Drive
# (DATA/mesh_registration/{ALL_ANTS_CLEAN,custom_processing/antscan_proofread_castes/worker_ALT}),
# synced to this HPC cluster via rclone 2026-08-13 (81 + 757 files, exact count match, see
# diagnostics/morphometrics/ab_scale_cap/SCHEMA.md for the sync provenance). Defaults below
# still match Fabian's machine for backward compatibility; override for this cluster with:
#   CONDA_SH=/p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh \
#   REPO_DIR=/p/home/jusers/jellal1/jureca/SMILify \
#   W=/p/scratch/cias-7/jellal1/SMILify_DATA/custom_processing/antscan_proofread_castes/worker_ALT \
#   CLEAN=/p/scratch/cias-7/jellal1/SMILify_DATA/ALL_ANTS_CLEAN \
#   ./run_m1_fit_all.sh
CONDA_SH=${CONDA_SH:-/home/fabi/mambaforge/etc/profile.d/conda.sh}
REPO_DIR=${REPO_DIR:-/home/fabi/dev/SMILify}
source "$CONDA_SH"
conda activate pytorch3d
cd "$REPO_DIR"
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
# YAML_SRC and TAG_PREFIX are overridable so the §4 A/B (diagnostics/morphometrics/ab_scale_cap/)
# can run arm A and arm B through this same script. R must NOT be overridden to a separate
# root: diagnostics/morphometrics/measure.py's measure_run() hardcodes
# diagnostics/moonshot/runs/<tag>/Stage_3_deform_fine.npz (MOON = diagnostics/moonshot,
# confirmed by reading measure.py directly, not assumed) -- analyse.py/genus_table.py can
# only address a run by TAG NAME under that fixed root, not by an arbitrary directory. Arm
# identity goes in the tag prefix instead, e.g.:
#   TAG_PREFIX=AB_A_ YAML_SRC=diagnostics/moonshot/cfg/D1_PROD.yaml FORCE_REFIT=1 \
#     STAGE=diagnostics/morphometrics/ab_scale_cap/stage_A ./run_m1_fit_all.sh
#   TAG_PREFIX=AB_B_ YAML_SRC=diagnostics/moonshot/cfg/D1_PROD_SCALECAP.yaml FORCE_REFIT=1 \
#     STAGE=diagnostics/morphometrics/ab_scale_cap/stage_B ./run_m1_fit_all.sh
# (STAGE must differ between concurrent invocations -- see its own definition below for why.)
# which produces AB_A_W0.. / AB_A_CLEAN and AB_B_W0.. / AB_B_CLEAN side by side under the same
# diagnostics/moonshot/runs/, and can then be passed to analyse.py as
# --worker_runs AB_A_W0 AB_A_W1 ... --clean_run AB_A_CLEAN (and the AB_B_ equivalents).
R=diagnostics/moonshot/runs
TAG_PREFIX=${TAG_PREFIX:-MORPH_}
# STAGE is overridable too: two invocations sharing the default staging dir would race on the
# same symlink directory if run concurrently (e.g. the §4 A/B's two arms) -- the chunking
# script below deletes and recreates symlinks in $STAGE/w*, and a fit() call's --mesh_dir glob
# could transiently see a partial chunk mid-restage from the OTHER arm's invocation. Give
# concurrent invocations distinct STAGE dirs.
STAGE=${STAGE:-diagnostics/morphometrics/stage}
# Note the Drive-synced folder is named "worker_ALT", not "worker" -- confirmed by the user
# (2026-08-13) as the correct corpus despite the name differing from this script's original
# /media/fabi/Data/.../worker path; file count matches exactly (757) so this is treated as
# the same corpus under a different folder name, not a substitute.
W=${W:-/media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker}
CLEAN=${CLEAN:-/media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN}
CHUNK=${CHUNK:-64}

python - "$W" "$CLEAN" "$STAGE" "$CHUNK" <<'PY'
import os, sys, glob, math
W, CLEAN, STAGE, chunk = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
work = sorted(f for f in os.listdir(W) if f.endswith(".obj"))
n = math.ceil(len(work) / chunk)
for i in range(n):
    d = f"{STAGE}/w{i}"
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/*.obj"):
        os.unlink(f)
    for nm in work[i * chunk:(i + 1) * chunk]:
        os.symlink(os.path.join(W, nm), os.path.join(d, nm))
print(f"[m1] {len(work)} workers in {n} chunks of <= {chunk}")
d = f"{STAGE}/clean"
os.makedirs(d, exist_ok=True)
for f in glob.glob(f"{d}/*.obj"):
    os.unlink(f)
cl = sorted(f for f in os.listdir(CLEAN) if f.endswith(".obj"))
for nm in cl:
    os.symlink(os.path.join(CLEAN, nm), os.path.join(d, nm))
print(f"[m1] {len(cl)} ALL_ANTS_CLEAN meshes in 1 chunk")
open(f"{STAGE}/nchunks.txt", "w").write(str(n))
PY

N=$(cat $STAGE/nchunks.txt)

# Skip-if-exists is RECIPE-AWARE, not just existence-based (this class of bug -- silently
# reusing stale output from a prior recipe with no error -- has already caused one near-miss
# in this investigation, see diagnostics/V2_CHANGELOG.md, "operational gotcha" entry,
# 2026-08-11: the D1_low -> D1_PROD switch). A fingerprint of the yaml_src config + the fixed
# hierarchical CLI flags is stamped into $R/$t/RECIPE_ID.txt on a successful fit. `fit()` only
# skips when Stage_3_deform_fine.npz exists AND its stamp matches the CURRENT recipe; a
# mismatch (or missing stamp, e.g. pre-dating this mechanism) triggers a warning and a refit,
# not a silent stale reuse. Set FORCE_REFIT=1 to always refit regardless of the stamp (e.g. for
# the diagnostics/morphometrics/ab_scale_cap/ A/B in the execution plan, which requires
# force_refit: true so arm A and arm B are never confused for each other's cached output).
YAML_SRC=${YAML_SRC:-diagnostics/moonshot/cfg/D1_PROD.yaml}
HIER_FLAGS="--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3"
RECIPE_ID=$( { cat "$YAML_SRC"; echo "$HIER_FLAGS"; } | sha256sum | cut -c1-16)
FORCE_REFIT=${FORCE_REFIT:-0}

fit () { # fit <gpu> <mesh_dir> <tag>
  local g=$1 M=$2 t=$3
  local stamp="$R/$t/RECIPE_ID.txt"
  if [ -f "$R/$t/Stage_3_deform_fine.npz" ]; then
    if [ "$FORCE_REFIT" != "1" ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$RECIPE_ID" ]; then
      echo "[$t] already done under the current recipe ($RECIPE_ID), skipping"
      return 0
    fi
    echo "[$t] WARNING: existing output is stale (recipe changed or FORCE_REFIT=1) -- refitting, not reusing"
  fi
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
    $HIER_FLAGS \
    --results_dir $R/${t}_hier >$R/$t.log 2>&1 || { echo "[$t] HIER FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
    --yaml_src $YAML_SRC --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1 || { echo "[$t] MOONSHOT FAILED"; return 1; }
  echo "$RECIPE_ID" > "$stamp"
  echo "[$(date +%H:%M:%S)] $t done"
}

# GPU0 takes even worker chunks plus the clean corpus, GPU1 the odd ones
( fit 0 $STAGE/clean ${TAG_PREFIX}CLEAN
  for ((i=0;i<N;i+=2)); do fit 0 $STAGE/w$i ${TAG_PREFIX}W$i; done ) &
( for ((i=1;i<N;i+=2)); do fit 1 $STAGE/w$i ${TAG_PREFIX}W$i; done ) &
wait
echo "[$(date +%H:%M:%S)] M1 fits complete"
