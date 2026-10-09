#!/usr/bin/env bash
# Clones and builds hjwdzh/ManifoldPlus (the 2020 successor to hjwdzh/Manifold, already vendored
# alongside this script - see MANIFOLD_SETUP.md) into
# custom_processing/external/ManifoldPlus/build/manifold. Used as the fallback reconstruction
# method in prepare_antscan_data_for_mesh_fitting_alphawrap.py when a CGAL alpha-wrap build
# (setup_alpha_wrap.sh) is impractical. Not vendored in git (see .gitignore) - run once per
# machine. Requires cmake, a C++ toolchain, and Eigen3 (`sudo apt install libeigen3-dev` on
# Ubuntu 24.04) - a much lighter dependency footprint than CGAL's.
#
# UNVERIFIED on this machine as of 2026-08-13 (no internet access in the session that wrote this
# script) - see ALPHA_WRAP_SETUP.md.
#
# Usage: bash custom_processing/external/setup_manifoldplus.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/ManifoldPlus"

if [ -x "${TARGET_DIR}/build/manifold" ]; then
    echo "ManifoldPlus already built at ${TARGET_DIR}/build/manifold - nothing to do."
    exit 0
fi

if [ ! -d "${TARGET_DIR}" ]; then
    git clone --recursive https://github.com/hjwdzh/ManifoldPlus.git "${TARGET_DIR}"
fi

mkdir -p "${TARGET_DIR}/build"
cd "${TARGET_DIR}/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

if [ ! -x "${TARGET_DIR}/build/manifold" ]; then
    echo "ERROR: build finished but ${TARGET_DIR}/build/manifold was not produced - check this" >&2
    echo "repo's actual CMakeLists.txt output binary name/path, it may differ from what this" >&2
    echo "script assumes." >&2
    exit 1
fi

echo "Built: ${TARGET_DIR}/build/manifold"
echo "Run '${TARGET_DIR}/build/manifold --help' and compare its actual CLI flags against" \
     "close_holes_via_manifold_plus()'s subprocess call in" \
     "prepare_antscan_data_for_mesh_fitting_alphawrap.py (--input/--output/--depth assumed)" \
     "before trusting it on real specimens."
