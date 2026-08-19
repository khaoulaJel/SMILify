"""Recover a specimen-specific model-unit -> millimetre conversion for AntScan meshes.

BACKGROUND
Every AntScan source scan ships a per-specimen `voxel_size` in its JSON sidecar (a scanner
calibration in micrometres/voxel, e.g. "2.44 µm"; it varies per specimen -- never treat it as a
global constant). The raw STL and every mesh derived from it up to and including the
`_processed.obj` used for SMAL fitting (`custom_processing/prepare_antscan_data_for_mesh_fitting*.py`)
carry vertex coordinates in that same *unscaled voxel-index* unit: every `transform_apply` call in
the mesh-prep pipeline passes `scale=False`, so only rotation/translation/topology change --
coordinate magnitudes never do.

The SMAL-fitting target loader (`fitter_3d/utils.py:load_meshes`, on `feature/registration_moonshot`)
then does:

    centre = verts.mean(0)
    verts  = verts - centre
    scale  = max(verts.abs().max(0)[0])
    verts  = verts / scale

`scale` is exactly what turns raw-voxel-unit coordinates into "model units" -- and it is not saved
anywhere, but it is a pure function of that one mesh's own vertices, so it can always be
recomputed offline. Composing the two:

    mm_per_model_unit = scale * voxel_size_um / 1000.0

is therefore a valid, specimen-specific calibration constant requiring only (a) the specimen's own
JSON `voxel_size` and (b) the same mesh file `load_meshes` would have loaded for that specimen.

This module does NOT assume calibration exists elsewhere -- everything here is derived directly
from files already on disk, and `compute_normalization_scale` replicates the `load_meshes` recipe
verbatim so results match whatever the fitter actually did.
"""

import argparse
import json
import os
import re

import numpy as np
import trimesh

VOXEL_SIZE_RE = re.compile(r"([\d.]+)\s*(µm|um|nm|mm|cm)", re.IGNORECASE)

_UNIT_TO_MM = {
    "nm": 1e-6,
    "um": 1e-3,
    "µm": 1e-3,
    "mm": 1.0,
    "cm": 10.0,
}

# Plausible worker-ant body-length range, used only as a sanity gate on calibration output --
# never to derive or adjust the calibration itself.
PLAUSIBLE_MM_RANGE = (0.5, 30.0)


def parse_voxel_size_mm(json_path):
    """Read a specimen JSON sidecar and return its voxel_size in mm/voxel-unit, or None if absent."""
    with open(json_path, "r") as f:
        meta = json.load(f)
    raw = meta.get("voxel_size", "")
    if not raw:
        return None
    m = VOXEL_SIZE_RE.search(raw)
    if not m:
        raise ValueError(f"Could not parse voxel_size string {raw!r} in {json_path}")
    value, unit = float(m.group(1)), m.group(2).lower()
    return value * _UNIT_TO_MM[unit]


def compute_normalization_scale(mesh_path):
    """Recompute the exact `scale` factor `fitter_3d.utils.load_meshes` would apply to this mesh.

    Returns (scale, centre, raw_vertices) in the mesh file's own (unscaled) units.
    """
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    centre = verts.mean(axis=0)
    centred = verts - centre
    scale = float(np.abs(centred).max())
    return scale, centre, verts


def calibrate_specimen(mesh_path, json_path):
    """Return the full calibration record for one specimen."""
    voxel_size_mm = parse_voxel_size_mm(json_path)
    scale, centre, verts = compute_normalization_scale(mesh_path)

    record = {
        "mesh_path": mesh_path,
        "json_path": json_path,
        "voxel_size_mm": voxel_size_mm,
        "normalization_scale": scale,
        "raw_bbox_dims": (verts.max(axis=0) - verts.min(axis=0)).tolist(),
    }

    if voxel_size_mm is None:
        record["mm_per_model_unit"] = None
        record["status"] = "no_voxel_size"
        return record

    mm_per_model_unit = scale * voxel_size_mm
    # A specimen normalized to model units has max|coord| == 1 by construction (that's what
    # `scale` divided out), so its bounding diagonal in mm is bounded by ~2 * sqrt(3) model units.
    raw_max_dim_mm = max(record["raw_bbox_dims"]) * voxel_size_mm

    status = "ok"
    if not (PLAUSIBLE_MM_RANGE[0] <= raw_max_dim_mm <= PLAUSIBLE_MM_RANGE[1]):
        status = "implausible_size"

    record["mm_per_model_unit"] = mm_per_model_unit
    record["raw_max_dim_mm"] = raw_max_dim_mm
    record["status"] = status
    return record


def find_specimen_files(specimen_dir):
    """Locate the (mesh, json) pair for a specimen directory, preferring a processed mesh
    (the one actually fed to load_meshes) over the raw STL if both are present."""
    name = os.path.basename(os.path.normpath(specimen_dir))
    json_path = os.path.join(specimen_dir, f"{name}.json")
    if not os.path.exists(json_path):
        candidates = [f for f in os.listdir(specimen_dir) if f.endswith(".json")]
        json_path = os.path.join(specimen_dir, candidates[0]) if candidates else None

    processed = [f for f in os.listdir(specimen_dir) if f.endswith("_processed.obj")]
    if processed:
        mesh_path = os.path.join(specimen_dir, processed[0])
    else:
        stl_candidates = [f for f in os.listdir(specimen_dir) if f.lower().endswith(".stl")]
        mesh_path = os.path.join(specimen_dir, stl_candidates[0]) if stl_candidates else None

    return mesh_path, json_path


def calibrate_corpus(corpus_dir):
    """Run calibration over every specimen subdirectory in corpus_dir. Returns list of records."""
    records = []
    for name in sorted(os.listdir(corpus_dir)):
        specimen_dir = os.path.join(corpus_dir, name)
        if not os.path.isdir(specimen_dir):
            continue
        mesh_path, json_path = find_specimen_files(specimen_dir)
        if mesh_path is None or json_path is None:
            records.append({"specimen": name, "status": "missing_files"})
            continue
        record = calibrate_specimen(mesh_path, json_path)
        record["specimen"] = name
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", help="Directory of specimen subdirectories (e.g. custom_processing/antscan_data)")
    parser.add_argument("--out", default=None, help="Path to write JSON results (default: print summary only)")
    args = parser.parse_args()

    records = calibrate_corpus(args.corpus_dir)

    ok = [r for r in records if r.get("status") == "ok"]
    implausible = [r for r in records if r.get("status") == "implausible_size"]
    missing = [r for r in records if r.get("status") in ("missing_files", "no_voxel_size")]

    print(f"Calibrated {len(ok)}/{len(records)} specimens.")
    if implausible:
        print(f"WARNING: {len(implausible)} specimens outside plausible size range {PLAUSIBLE_MM_RANGE} mm:")
        for r in implausible:
            print(f"  {r['specimen']}: raw_max_dim_mm={r['raw_max_dim_mm']:.3f}")
    if missing:
        print(f"{len(missing)} specimens skipped (missing mesh/json/voxel_size):")
        for r in missing:
            print(f"  {r['specimen']}: {r['status']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
