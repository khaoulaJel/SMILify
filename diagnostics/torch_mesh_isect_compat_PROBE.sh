#!/usr/bin/env bash
# torch_mesh_isect_compat_PROBE.sh
#
# Answers one question before any torch-mesh-isect-based penetration-loss arm
# gets designed: does the compiled BVH collision extension (bvh_cuda) build
# and run correctly against THIS environment's pinned stack (Python 3.10,
# torch 2.3.1, CUDA 11.8)? The original repo (vchoutas/torch-mesh-isect) was
# tested against torch 1.0/CUDA 10.0; the maintained fork
# (EthanFifle/torch-mesh-intersection) against torch 2.0/CUDA 11.7 -- neither
# is an exact match, and compiled-CUDA-op ABI mismatches are often invisible
# on import, only surfacing at kernel launch. This is a compile+run test, not
# an import-only check.
#
# Tries the maintained fork first (bundles its own CUDA-sample headers,
# no CUDA_SAMPLES_INC needed per its setup.py). Falls back to the original
# vchoutas repo only if the fork's build fails -- the original additionally
# needs $CUDA_SAMPLES_INC pointing at helper_math.h from NVIDIA's cuda-samples
# repo, which this script does NOT fetch automatically (kept as a manual,
# visible step if the fallback path is ever reached, rather than silently
# vendoring a second external dependency).
#
# Clones into custom_processing/external/, matching this project's existing
# convention for compiled external deps (Manifold, ManifoldPlus,
# cgal_alpha_wrap already live there, gitignored / left untracked for review).
#
# Builds IN PLACE (build_ext --inplace), not `pip install`, so nothing is
# written into the shared pytorch3d conda env's site-packages -- only this
# script's own PYTHONPATH addition makes the extension importable, and
# deleting the cloned directory fully undoes it.
#
# Run from the repo root:
#     bash diagnostics/torch_mesh_isect_compat_PROBE.sh
#
# All build logs and the final probe output are preserved under diagnostics/
# per this project's diagnostic-artifact convention -- nothing is cleaned up
# automatically.
# NOTE: intentionally no `-u` -- conda's activate/deactivate hooks reference
# unset vars (e.g. CONDA_BACKUP_CXX, MKL_INTERFACE_LAYER) and abort under it.
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXT_ROOT="custom_processing/external"
FORK_DIR="$EXT_ROOT/torch-mesh-intersection"
ORIG_DIR="$EXT_ROOT/torch-mesh-isect"
OUT_DIR="diagnostics"
SUMMARY="$OUT_DIR/torch_mesh_isect_compat_PROBE_out.txt"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-pytorch3d}"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

mkdir -p "$EXT_ROOT" "$OUT_DIR"

: > "$SUMMARY"
log() { echo "$@" | tee -a "$SUMMARY"; }

log "=== torch-mesh-isect compatibility probe -- $(date) ==="
log "python: $(python --version 2>&1)"
log "torch:  $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
log "gpu:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")')"

build_variant() {
    local dir="$1" url="$2" name="$3" logfile="$4"
    if [ ! -d "$dir" ]; then
        log "[$name] cloning $url -> $dir"
        git clone --depth 1 "$url" "$dir" >>"$logfile" 2>&1 || { log "[$name] clone FAILED, see $logfile"; return 1; }
    else
        log "[$name] $dir already present, reusing (not re-cloning)"
    fi
    log "[$name] building in place (build_ext --inplace)"
    (cd "$dir" && python setup.py build_ext --inplace) >>"$logfile" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        log "[$name] BUILD FAILED (exit $status), see $logfile"
        return 1
    fi
    log "[$name] build succeeded"
    return 0
}

VARIANT=""

FORK_LOG="$OUT_DIR/torch_mesh_isect_compat_PROBE_fork_build_out.txt"
: > "$FORK_LOG"
if build_variant "$FORK_DIR" "https://github.com/EthanFifle/torch-mesh-intersection" "fork" "$FORK_LOG"; then
    VARIANT="fork"
    export PYTHONPATH="$REPO_ROOT/$FORK_DIR:${PYTHONPATH:-}"
else
    log ""
    log "Fork build failed -- falling back to original vchoutas/torch-mesh-isect."
    log "NOTE: the original repo needs \$CUDA_SAMPLES_INC pointing at helper_math.h"
    log "from https://github.com/NVIDIA/cuda-samples (Common/ dir). Not fetched"
    log "automatically by this script -- set CUDA_SAMPLES_INC yourself and re-run"
    log "the fallback block manually if this also fails for that reason (check"
    log "$OUT_DIR/torch_mesh_isect_compat_PROBE_orig_build_out.txt for the actual error)."
    ORIG_LOG="$OUT_DIR/torch_mesh_isect_compat_PROBE_orig_build_out.txt"
    : > "$ORIG_LOG"
    if build_variant "$ORIG_DIR" "https://github.com/vchoutas/torch-mesh-isect" "original" "$ORIG_LOG"; then
        VARIANT="original"
        export PYTHONPATH="$REPO_ROOT/$ORIG_DIR:${PYTHONPATH:-}"
    fi
fi

if [ -z "$VARIANT" ]; then
    log ""
    log "=== OVERALL: BUILD FAILED (neither fork nor original compiled) ==="
    exit 1
fi

log ""
log "=== running collision probe against the '$VARIANT' build ==="
python diagnostics/torch_mesh_isect_compat_probe.py 2>&1 | tee -a "$SUMMARY"
PROBE_STATUS=${PIPESTATUS[0]}

log ""
log "variant used: $VARIANT"
if [ $PROBE_STATUS -eq 0 ]; then
    log "=== OVERALL: PASS ==="
else
    log "=== OVERALL: FAIL (probe exited $PROBE_STATUS) ==="
fi
exit $PROBE_STATUS
