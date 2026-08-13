#!/bin/bash
#SBATCH --job-name=antscan
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:59:00
#SBATCH --signal=B:SIGUSR1@600
#SBATCH --output=/p/scratch/cias-7/jellal1/antscan/logs/antscan_%j.out
#SBATCH --error=/p/scratch/cias-7/jellal1/antscan/logs/antscan_%j.err

# Pure local compute: reads STL files already staged on scratch, writes
# processed .obj files back to scratch. No Google Drive / network calls
# here on purpose -- the dc-gpu partition's compute nodes have no route to
# the internet (verified: DNS resolves, but every connect to Google's IPs
# fails "No route to host"). All Drive syncing happens separately on the
# login node via sync_antscan_drive.sh, which does have internet access.

# --- environment -------------------------------------------------------
source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d

module load Stages/2025 GCCcore/.13.3.0 Blender/4.3.2-linux-x86_64-CUDA-12 Xvfb/21.1.14

# Under `sbatch`, the script runs from a spool-directory copy, so
# ${BASH_SOURCE[0]} would resolve to the wrong place. $SLURM_SUBMIT_DIR
# (the directory `sbatch` was invoked from) is correct there; fall back to
# the script's own location when run directly with `bash run_antscan.sh`.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Blender's GHOST windowing layer needs a display even in --background mode
# on this build (fails with "unable to open a display" otherwise). JURECA
# nodes have no X server, so start a virtual one (Xvfb) and force software
# rendering (the node's video device is a BMC chip, not a real GPU, and
# Mesa hangs trying to probe it for hardware acceleration).
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
Xvfb "$DISPLAY" -screen 0 1024x768x24 &
XVFB_PID=$!
trap 'kill "$XVFB_PID" 2>/dev/null' EXIT
sleep 2

# --- local working copy on scratch (populated by sync_antscan_drive.sh) --
WORK_DIR="/p/scratch/cias-7/jellal1/antscan"
DATA_DIR="$WORK_DIR/antscan_data"
OUT_DIR="$WORK_DIR/antscan_processed"

mkdir -p "$DATA_DIR" "$OUT_DIR"

# SLURM sends SIGUSR1 10 minutes before the 24h time limit (--signal above).
# Bash defers trap execution until the current foreground command (a single
# blender call) finishes, so this cleanly stops between specimens rather
# than killing one mid-write, then resubmits itself so the whole dataset
# gets processed across multiple 24h jobs without needing anyone to notice
# a TIMEOUT and manually resubmit.
RESUBMIT_NEEDED=0
trap 'echo "Time limit approaching -- will resubmit after the current specimen."; RESUBMIT_NEEDED=1' SIGUSR1

# Process substitution (not a pipe) so the loop runs in this shell, not a
# subshell -- otherwise RESUBMIT_NEEDED set by the trap wouldn't be visible
# here.
while read -r stl_file; do
    # Extract base filename without extension
    filename=$(basename "$stl_file" .stl)
    out_file="$OUT_DIR/${filename}_processed.obj"

    # Skip if output file already exists
    if [ -f "$out_file" ]; then
        echo "Skipping (already processed): $filename"
        continue
    fi

    if [ "$RESUBMIT_NEEDED" -eq 1 ]; then
        echo "Stopping early to resubmit before hitting the time limit."
        break
    fi

    echo "Processing specimen: $stl_file"
    blender --background --python "$REPO_DIR/custom_processing/prepare_antscan_data_for_mesh_fitting.py" -- "$stl_file" "$OUT_DIR"

    # If Blender encounters an I/O glitch, log warning and keep processing the rest
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed processing $stl_file. Continuing to next..."
    fi
done < <(find "$DATA_DIR" -type f -name "*.stl")

REMAINING=$(comm -23 \
    <(find "$DATA_DIR" -type f -name "*.stl" -exec basename {} .stl \; | sort) \
    <(find "$OUT_DIR" -type f -name "*_processed.obj" -exec basename {} _processed.obj \; | sort) \
    | wc -l)

if [ "$REMAINING" -gt 0 ]; then
    echo "$REMAINING specimen(s) still unprocessed -- resubmitting to continue."
    sbatch "$REPO_DIR/run_antscan.sh"
else
    echo "All specimens processed. Not resubmitting."
fi
