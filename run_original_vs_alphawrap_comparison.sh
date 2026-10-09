#!/usr/bin/env bash
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BLENDER="${BLENDER:-$HOME/software/blender-4.2.23-linux-x64/blender}"
DATA_DIR="${DATA_DIR:-$HOME/SMILify/diagnostics/moonshot/bench50_clean}"

# Blender 4.2 uses its own Python and does not automatically see ~/.local.
export PYTHONPATH="${PYTHONPATH:-$HOME/.local/lib/python3.11/site-packages}"

ORIGINAL_SCRIPT="$REPO_DIR/custom_processing/prepare_antscan_data_for_mesh_fitting.py"
ALPHAWRAP_SCRIPT="$REPO_DIR/custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/comparison_output}"
LOG_DIR="$OUTPUT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)


mkdir -p "$LOG_DIR" "$OUTPUT_DIR/original" "$OUTPUT_DIR/alphawrap"

mapfile -t SPECIMENS < <(find "$DATA_DIR" -type f \( -iname "*.stl" -o -iname "*.obj" \))
echo "Found ${#SPECIMENS[@]} specimens under $DATA_DIR"

for input_path in "${SPECIMENS[@]}"; do
    name=$(basename "$input_path")
    name="${name%.*}"

    echo "=== [$name] original ==="
    "$BLENDER" --background --python-exit-code 1 \
        --python "$ORIGINAL_SCRIPT" \
        -- "$input_path" "$OUTPUT_DIR/original" \
        > "$LOG_DIR/${name}_original_${TIMESTAMP}.log" 2>&1
    echo "  exit_code=$? log=$LOG_DIR/${name}_original_${TIMESTAMP}.log"

    echo "=== [$name] alphawrap ==="
    "$BLENDER" --background --python-exit-code 1 \
        --python "$ALPHAWRAP_SCRIPT" \
        -- "$input_path" "$OUTPUT_DIR/alphawrap" alpha_wrap_only \
        > "$LOG_DIR/${name}_alphawrap_${TIMESTAMP}.log" 2>&1
    echo "  exit_code=$? log=$LOG_DIR/${name}_alphawrap_${TIMESTAMP}.log"
done

echo ""
echo "Done. FINAL_SUMMARY lines:"
grep -h "^FINAL_SUMMARY:" "$LOG_DIR"/*_"${TIMESTAMP}".log
