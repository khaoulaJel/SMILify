#!/bin/bash
module load Stages/2025 GCCcore/.13.3.0 Blender/4.3.2-linux-x86_64-CUDA-12 Xvfb/21.1.14

export DISPLAY=:86
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe

pkill -u $USER -f "Xvfb :86" 2>/dev/null
rm -f /tmp/.X86-lock 2>/dev/null
Xvfb :86 -screen 0 1024x768x24 & 2>/dev/null
sleep 2

mkdir -p diagnostics

cat << 'PYEOF' > /tmp/import_test.py
import sys
print("STEP1", flush=True)
sys.path.insert(0, "/p/home/jusers/jellal1/jureca/SMILify/custom_processing/external/blender_python_packages")
print("STEP2", flush=True)
import importlib.util
print("STEP3", flush=True)
spec = importlib.util.spec_from_file_location(
    "enhanced_mod",
    "/p/home/jusers/jellal1/jureca/SMILify/custom_processing/prepare_antscan_data_for_mesh_fitting_enhanced.py")
print("STEP4", flush=True)
mod = importlib.util.module_from_spec(spec)
print("STEP5", flush=True)
spec.loader.exec_module(mod)
print("IMPORT_OK", flush=True)
print("has report_bad_edges:", hasattr(mod, "report_bad_edges"), flush=True)
PYEOF

blender --background --python-exit-code 1 --python /tmp/import_test.py 2>&1 | tee diagnostics/import_test.txt
