#!/usr/bin/env bash
# Clones and builds the external Manifold mesh-repair tool (github.com/hjwdzh/Manifold) into
# custom_processing/external/Manifold/build/manifold. Not vendored in git (see .gitignore) -
# run this once per machine. Requires cmake and a C++ build toolchain.
#
# Usage: bash custom_processing/external/setup_manifold.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/Manifold"

if [ -x "${TARGET_DIR}/build/manifold" ]; then
    echo "Manifold already built at ${TARGET_DIR}/build/manifold - nothing to do."
    exit 0
fi

if [ ! -d "${TARGET_DIR}" ]; then
    git clone https://github.com/hjwdzh/Manifold.git "${TARGET_DIR}"
fi

mkdir -p "${TARGET_DIR}/build"
cd "${TARGET_DIR}/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

echo "Built: ${TARGET_DIR}/build/manifold"
