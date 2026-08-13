#!/bin/bash
module load Stages/2025 GCCcore/.13.3.0 Blender/4.3.2-linux-x86_64-CUDA-12 Xvfb/21.1.14

export DISPLAY=:87
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe

pkill -u $USER -f "Xvfb :87" 2>/dev/null
rm -f /tmp/.X87-lock 2>/dev/null
Xvfb :87 -screen 0 1024x768x24 & 2>/dev/null
sleep 2

REPO_DIR="/p/home/jusers/jellal1/jureca/SMILify"
TARGET_DIR="$REPO_DIR/custom_processing/external/blender_python_packages"
OUT_DIR="/p/scratch/cias-7/jellal1/antscan/antscan_processed_enhanced"

SCRIPT_PATH="$REPO_DIR/custom_processing/prepare_antscan_data_for_mesh_fitting_enhanced.py"

for f in Philidris_sp._CASENT0878052 Centromyrmex_brachycola_CASENT0744052; do
  echo "=========================================="
  echo "Running manifold repair for: $f"
  echo "=========================================="
  
  blender --background --python-exit-code 1 \
    --python-expr "import sys; sys.path.insert(0, '$TARGET_DIR')" \
    --python "$SCRIPT_PATH" \
    -- "/p/scratch/cias-7/jellal1/antscan/antscan_data/$f/$f.stl" \
       "$OUT_DIR"
done
