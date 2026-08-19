"""Task 1 (absolute physical scale): recompute the existing morphometric measurement set in mm.

Source measurement set: diagnostics/morphometrics/out/morphometrics.csv on
feature/registration_moonshot (mirrored here as morphometrics_source.csv), 757 `worker` rows +
81 `ALL_ANTS_CLEAN` rows. Length columns in that CSV are Mosimann log-shape-ratios: for row i,
column c, `csv[i, c] = log(X[i, c]) - log_size[i]` where `log_size[i] = mean_c(log(X[i, :]))`.
So the raw model-unit length is `X[i, c] = exp(csv[i, c] + log_size[i])` -- see
diagnostics/morphometrics/measure.py:log_shape_ratios on that branch.

Calibration: `custom_processing/calibrate_scale.py` gives `mm_per_model_unit` for a specimen from
its own worker_ALT `_processed.obj` (the exact mesh `load_meshes` would load) and its antscan_data
JSON `voxel_size`. Absolute mm length = `X[i, c] * mm_per_model_unit[i]`.

COVERAGE. Only worker specimens whose raw AntScan source scan (with voxel_size) is present in the
antscan_data mirror get a calibration -- 279 of 757 (128 of 172 genera), confirmed by direct
accession-code search that the other 478 are genuinely absent, not a naming mismatch. ALL_ANTS_CLEAN
(81 specimens) has NO calibration path at all: those meshes are generic-numbered .obj files with no
JSON, no specimen identity, and are already pre-normalised with no recorded original units -- do not
attempt to calibrate them, and do not drop them from the output; they stay proportions-only.
"""

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "custom_processing"))
from calibrate_scale import calibrate_specimen  # noqa: E402

SOURCE_CSV = os.path.join(HERE, "morphometrics_source.csv")
WORKER_ALT_DIR = "/hpcwork/nao48500/worker_ALT"
ANTSCAN_JSON_DIR = "/hpcwork/nao48500/antscan_data_json_only"
OUT_CSV = os.path.join(HERE, "morphometrics_absolute_mm.csv")
CALIB_CACHE_JSON = os.path.join(HERE, "calibration_by_specimen.json")

META_COLS = {"label", "source", "genus", "species", "subfamily", "ecology", "asym_median", "log_size"}


def find_json_path(specimen_name):
    direct = os.path.join(ANTSCAN_JSON_DIR, specimen_name, f"{specimen_name}.json")
    if os.path.exists(direct):
        return direct
    # rclone --include preserves the source tree; fall back to a search in case of nesting quirks
    for root, _, files in os.walk(ANTSCAN_JSON_DIR):
        if f"{specimen_name}.json" in files:
            return os.path.join(root, f"{specimen_name}.json")
    return None


def main():
    with open(SOURCE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    length_cols = [c for c in fieldnames if c not in META_COLS and not c.startswith("shapedev_")]
    print(f"{len(rows)} total rows, {len(length_cols)} length columns to back-transform.")

    calib_by_specimen = {}
    out_rows = []
    n_calibrated = 0
    n_no_source = 0
    n_implausible = 0

    for row in rows:
        label = row["label"]
        specimen = label[: -len("_processed.obj")] if label.endswith("_processed.obj") else label
        out_row = dict(row)

        if row["source"] != "worker":
            # ALL_ANTS_CLEAN: no calibration path exists (no JSON, no raw-scan provenance,
            # pre-normalised with unknown original units). Leave mm columns blank, do not guess.
            for c in length_cols:
                out_row[f"{c}_mm"] = ""
            out_row["mm_per_model_unit"] = ""
            out_row["calibration_status"] = "no_source_corpus"
            out_rows.append(out_row)
            n_no_source += 1
            continue

        obj_path = os.path.join(WORKER_ALT_DIR, label)
        json_path = find_json_path(specimen)

        if not os.path.exists(obj_path) or json_path is None:
            for c in length_cols:
                out_row[f"{c}_mm"] = ""
            out_row["mm_per_model_unit"] = ""
            out_row["calibration_status"] = "no_matching_raw_scan"
            out_rows.append(out_row)
            n_no_source += 1
            continue

        calib = calibrate_specimen(obj_path, json_path)
        calib_by_specimen[specimen] = calib

        if calib["mm_per_model_unit"] is None:
            for c in length_cols:
                out_row[f"{c}_mm"] = ""
            out_row["mm_per_model_unit"] = ""
            out_row["calibration_status"] = "no_voxel_size"
            out_rows.append(out_row)
            n_no_source += 1
            continue

        mm_per_unit = calib["mm_per_model_unit"]
        log_size = float(row["log_size"])
        for c in length_cols:
            z = float(row[c])
            raw_model_units = np.exp(z + log_size)
            out_row[f"{c}_mm"] = f"{raw_model_units * mm_per_unit:.6f}"
        out_row["mm_per_model_unit"] = f"{mm_per_unit:.8f}"
        out_row["calibration_status"] = calib["status"]  # "ok" or "implausible_size"
        if calib["status"] == "implausible_size":
            n_implausible += 1
        else:
            n_calibrated += 1
        out_rows.append(out_row)

    new_fieldnames = fieldnames + [f"{c}_mm" for c in length_cols] + ["mm_per_model_unit", "calibration_status"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    with open(CALIB_CACHE_JSON, "w") as f:
        json.dump(calib_by_specimen, f, indent=2)

    print(f"Calibrated (plausible): {n_calibrated}")
    print(f"Calibrated but flagged implausible size: {n_implausible}")
    print(f"No calibration available (ALL_ANTS_CLEAN or missing raw scan): {n_no_source}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {CALIB_CACHE_JSON}")


if __name__ == "__main__":
    main()
