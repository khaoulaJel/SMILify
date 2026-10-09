#!/usr/bin/env bash
# Clones CGAL and compiles the Alpha_wrap_3 triangle_soup_wrap example into
# custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap. Not vendored in git (see
# .gitignore) - run this once per machine. Requires cmake, a C++17 toolchain, Boost, GMP/MPFR,
# and Eigen3 (CGAL's usual dependency set - `sudo apt install libboost-dev libgmp-dev libmpfr-dev
# libeigen3-dev` on Ubuntu 24.04 covers all of them; CGAL itself is fetched via git clone below
# rather than assuming a system libcgal-dev package, since the example .cpp files are not shipped
# in that package).
#
# UNVERIFIED on this machine as of 2026-08-13 (no internet access / heavy deps unavailable in the
# session that wrote this script) - see ALPHA_WRAP_SETUP.md. Run this, read the actual build
# output, and fix whatever it complains about before trusting it blindly.
#
# Usage: bash custom_processing/external/setup_alpha_wrap.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CGAL_DIR="${SCRIPT_DIR}/cgal_alpha_wrap/CGAL"
BUILD_DIR="${SCRIPT_DIR}/cgal_alpha_wrap/build"
EXAMPLE_SRC_SUBDIR="Alpha_wrap_3/examples/Alpha_wrap_3"

if [ -x "${BUILD_DIR}/triangle_soup_wrap" ]; then
    echo "triangle_soup_wrap already built at ${BUILD_DIR}/triangle_soup_wrap - nothing to do."
    exit 0
fi

if [ ! -d "${CGAL_DIR}" ]; then
    # --depth 1 --branch: pin a recent release tag rather than tracking master, once one has
    # actually been build-tested against this script - left as `master` until that validation
    # pass happens (see ALPHA_WRAP_SETUP.md step 1).
    git clone --depth 1 https://github.com/CGAL/cgal.git "${CGAL_DIR}"
fi

EXAMPLE_DIR="${CGAL_DIR}/${EXAMPLE_SRC_SUBDIR}"
if [ ! -d "${EXAMPLE_DIR}" ]; then
    echo "ERROR: expected example directory not found at ${EXAMPLE_DIR}." >&2
    echo "CGAL's repo layout may have changed since this script was written - locate the" >&2
    echo "Alpha_wrap_3 'triangle_soup_wrap' example manually and adjust EXAMPLE_SRC_SUBDIR above." >&2
    exit 1
fi

if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    # A prior failed run's cache pins CMAKE_CXX_COMPILER to whatever was active then (the conda
    # env's compiler, in the failure this comment documents) - CMake does not cleanly support
    # switching compilers on an existing cache, so wipe it rather than risk a silently-stale
    # configuration. Cheap to regenerate; nothing here is hand-edited.
    echo "Removing stale build cache at ${BUILD_DIR} (compiler selection changed)..."
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
# CGAL_DIR points find_package(CGAL) at the git checkout directly: CGAL ships a CGALConfig.cmake
# at its repo root specifically so it can be used straight from source, without a separate
# `cmake --install` step - confirmed necessary here (first run without this failed with "Could
# not find a package configuration file provided by CGAL"; no system libcgal-dev was installed).
#
# Force the SYSTEM compiler, not whatever conda env happens to be active: confirmed necessary
# here (a run inside the `pytorch3d` conda env failed with "boost/config.hpp: No such file or
# directory" even with libboost-dev apt-installed) - conda-forge's gcc/g++ build with a sysroot
# scoped to the conda env itself, so it never sees /usr/include at all regardless of what's
# apt-installed there. This external tool is meant to be a standalone system build (same as
# Manifold/ManifoldPlus, which only happened to avoid this because they vendor their own
# dependencies rather than relying on system Boost/GMP/MPFR).
if [ ! -x /usr/bin/g++ ] || [ ! -x /usr/bin/gcc ]; then
    echo "ERROR: /usr/bin/gcc or /usr/bin/g++ not found - install the system build toolchain" >&2
    echo "(e.g. 'sudo apt install build-essential') before running this script." >&2
    exit 1
fi
cmake "${EXAMPLE_DIR}" -DCMAKE_BUILD_TYPE=Release -DCGAL_DIR="${CGAL_DIR}" \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++
make -j"$(nproc)" triangle_soup_wrap

if [ ! -x "${BUILD_DIR}/triangle_soup_wrap" ]; then
    echo "ERROR: build finished but ${BUILD_DIR}/triangle_soup_wrap was not produced - check the" >&2
    echo "cmake target name above against 'cmake --build . --target help' output." >&2
    exit 1
fi

echo "Built: ${BUILD_DIR}/triangle_soup_wrap"
echo "Run '${BUILD_DIR}/triangle_soup_wrap --help' (or with no args) and compare its actual CLI" \
     "against close_holes_via_alpha_wrap()'s subprocess call in" \
     "prepare_antscan_data_for_mesh_fitting_alphawrap.py before trusting it on real specimens."
