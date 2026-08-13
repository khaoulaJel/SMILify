"""Anatomical-region stratification for the integrity dashboard and failure-case gallery
(execution plan §2.3). Diagnostic/stratification tool, NOT a correspondence mechanism --
SDF-as-correspondence-restriction is closed on evidence (FINAL_REPORT.md §3, Phase D of the
execution plan) and this file must not become one. It answers "is deform_mag blowing up on
legs vs. gaster" for a given specimen -- not "which target vertex does this template vertex
match".

TWO paths, and this file is explicit about which one it uses where -- probed before writing,
per CLAUDE.md's "probe before editing" rule, not assumed:

1. EXACT (`region_labels_for_fit`, the path used for every Stage_3_deform_fine.npz fit).
   Confirmed directly on a real fit npz (diagnostics/khaoula_v2/runs/GEOCOH_01) before writing
   this: `verts`/`deform_verts` are (n_specimens, 10235, 3) and `faces` is
   (n_specimens, 20466, 3) -- IDENTICAL to the template's own vertex/face count on every
   specimen. Every fitted mesh in this pipeline is `v_template + deform_verts`, so vertex
   INDEX is template-vertex identity by construction; nothing is resampled or reindexed.
   That means the anatomical region of vertex i is already known exactly from the model's own
   skinning weights via `metrics.template_part_segmentation()` (already in this codebase,
   already used by `eval_run.py`/`part_metrics`) -- zero additional compute, zero
   approximation. This is what the integrity dashboard and the deform_mag failure gallery
   (§2.3 bullets 2-3) should use; SDF is not needed there and would only add a ~97-99%-AUC
   approximation on top of information that is already exact.

2. SDF-APPROXIMATE (`sdf_region_labels`, fallback only). For a mesh NOT in template
   correspondence -- e.g. a raw, un-fitted scan that might enter a future gallery before any
   fit exists for it. Reuses `fitter_3d.SDF_tests` (the module `SDF_batch.py` itself imports
   from, so this is the same computation `SDF_batch.py` runs, not a reimplementation) to get a
   per-vertex Shape Diameter Function value, then assigns each vertex to the anatomical group
   whose TEMPLATE SDF distribution it lands closest to (nearest-median, calibrated once on the
   template via `calibrate_sdf_thresholds`). This is the mechanism
   `diagnostics/moonshot/sdf_prior_PROBE.py` validated: AUC 0.97-0.99 gaster-vs-leg on the
   template, >=87.8% G1 (reproducibility) agreement on real bench50 scans -- good enough for
   triage, not for anything that needs to be exact.

Usage (exact path, the one actually needed today):
  python -m diagnostics.moonshot.sdf_stratify --run_dir <dir with Stage_3_deform_fine.npz> \
      --mesh_dir <target .obj dir, for specimen ordering> --out region_deform.csv

Usage (SDF fallback, for un-fitted meshes):
  python -m diagnostics.moonshot.sdf_stratify --raw_mesh some_scan.obj
"""

import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
import torch
import trimesh

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
import metrics as M  # noqa: E402
from fitter_3d.SDF_tests import compute_sdf, smooth_distances, assign_vertex_sdf  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402


# ---------------------------------------------------------------- 1. exact path


def load_template_dd():
    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def region_labels_for_fit(dd=None):
    """Dense (V,) array of region name per template vertex -- exact, from skinning weights.

    Returns (region_names_per_vertex, region_names_in_order). Every fitted mesh in this
    pipeline shares this exact vertex indexing (see module docstring).

    BUG FOUND 2026-08-13, FIXED HERE: raw `dd["weights"]`/`dd["v_template"]` from the pickle
    are (10229, ...) -- but every fit this pipeline has ever produced (checked GEOCOH_01,
    2026-08-12, and the fresh AB_A/AB_B A/B fits) carries (10235, ...) vertices, because
    `smal_model.smal_torch.SMAL.__init__` symmetrises the template (`align_smal_template_to_
    symmetry_axis`) AFTER loading, which changes vertex count, and the model class's own
    `self.weights` is NOT the raw `dd["weights"]` this function used to read but the
    post-symmetrisation array (verified empirically: instantiating `SMAL()` directly gives
    v_template/weights both at 10235, matching fits). Reading raw `dd["weights"]` silently
    produced a 10229-length region array that IndexErrors (or worse, would have silently
    misaligned without erroring) against every real fit. Fixed by loading through the actual
    model class instead of hand-unpickling.
    """
    from smal_model.smal_torch import SMAL

    m = SMAL(device="cpu", shape_family_id=-1)
    weights = m.weights.detach().cpu().numpy()
    if dd is None:
        dd = load_template_dd()
    part_ids, part_names = M.template_part_segmentation(weights, list(dd["J_names"]))
    n_verts = weights.shape[0]
    region = np.empty(n_verts, dtype=object)
    for ids, name in zip(part_ids, part_names):
        region[ids.numpy()] = name
    return region, part_names


def region_deform_table(run_dir, mesh_dir, stage=None):
    """Per-specimen, per-region deform_mag / edge-logratio stats for the fits in `run_dir`.

    This is the "grouping column" §2.2/§2.3 ask for: rows are (specimen, region), so the
    integrity dashboard can group by region instead of pooling every vertex together, and the
    "top-20 worst deform_mag" gallery can be explained by body region instead of eyeballed one
    mesh at a time.
    """
    region, region_names = region_labels_for_fit()

    stage_files = sorted(glob.glob(os.path.join(run_dir, "*.npz")))
    stage_files = [s for s in stage_files if "_batch_" not in s]
    if not stage_files:
        raise SystemExit(f"no stage npz in {run_dir}")
    if stage:
        stage_files = [s for s in stage_files if os.path.basename(s).startswith(stage)]
    else:
        stage_files = [stage_files[-1]]
    sf = stage_files[-1]
    d = np.load(sf, allow_pickle=True)

    v1 = d["verts"]  # (n, V, 3)
    dv = d["deform_verts"]  # (n, V, 3)
    faces = d["faces"][0].astype(np.int64)  # (F, 3), identical across specimens
    labels = [str(x) for x in d["labels"]]

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edge_region = region[e[:, 0]]  # approximation: an edge's region is its first vertex's region

    rows = []
    dn_all = np.linalg.norm(dv, axis=-1)  # (n, V)
    for i, name in enumerate(labels):
        v1i = v1[i]
        l1 = np.linalg.norm(v1i[e[:, 0]] - v1i[e[:, 1]], axis=-1)
        for reg in region_names:
            vmask = region == reg
            if not vmask.any():
                continue
            emask = edge_region == reg
            row = {
                "specimen": name,
                "region": reg,
                "n_verts": int(vmask.sum()),
                "deform_mag_mean": float(dn_all[i, vmask].mean()) if vmask.any() else float("nan"),
                "deform_mag_p95": float(np.quantile(dn_all[i, vmask], 0.95)) if vmask.any() else float("nan"),
                "deform_mag_max": float(dn_all[i, vmask].max()) if vmask.any() else float("nan"),
                "edge_len_mean": float(l1[emask].mean()) if emask.any() else float("nan"),
            }
            rows.append(row)
    return rows


# ---------------------------------------------------------------- 2. SDF fallback


def calibrate_sdf_thresholds(dd=None, num_rays=25, num_samples=-1):
    """Per-anatomical-group median SDF (relative to bbox diagonal) on the template.

    Reused, not reimplemented: same `compute_sdf`/`smooth_distances` SDF_batch.py itself
    calls. Returns {group_name: median_sdf_over_scale}, for nearest-median assignment on
    meshes that are NOT in template correspondence.
    """
    if dd is None:
        dd = load_template_dd()
    v_template = np.asarray(dd["v_template"], dtype=np.float32)
    f = np.asarray(dd["f"], dtype=np.int64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verts = torch.tensor(v_template, device=device)
    faces = torch.tensor(f, device=device)
    mesh = Meshes(verts=[verts], faces=[faces])

    sample_points, diameters = compute_sdf(mesh, num_samples=num_samples, num_rays=num_rays)
    smoothed = smooth_distances(sample_points, diameters, k=50)
    vertex_sdf = assign_vertex_sdf(verts, sample_points, smoothed, k=10).cpu().numpy()

    scale = float(trimesh.Trimesh(v_template, f, process=False).scale)
    region, region_names = region_labels_for_fit(dd)
    thresholds = {}
    for reg in region_names:
        m = (region == reg) & np.isfinite(vertex_sdf)
        if m.sum():
            thresholds[reg] = float(np.median(vertex_sdf[m]) / scale)
    return thresholds


def sdf_region_labels(mesh_path, thresholds, num_rays=25, num_samples=-1):
    """Per-vertex region label for a mesh with NO known template correspondence.

    Nearest-median SDF assignment against `thresholds` (from `calibrate_sdf_thresholds`).
    Approximate (AUC 0.97-0.99 on the template per sdf_prior_PROBE.py) -- use only when the
    exact path (`region_labels_for_fit`) does not apply, i.e. the mesh has not been fit.
    """
    m = trimesh.load(mesh_path, process=False, force="mesh")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verts = torch.tensor(np.asarray(m.vertices, dtype=np.float32), device=device)
    faces = torch.tensor(np.asarray(m.faces, dtype=np.int64), device=device)
    mesh = Meshes(verts=[verts], faces=[faces])

    sample_points, diameters = compute_sdf(mesh, num_samples=num_samples, num_rays=num_rays)
    smoothed = smooth_distances(sample_points, diameters, k=50)
    vertex_sdf = assign_vertex_sdf(verts, sample_points, smoothed, k=10).cpu().numpy()
    scale = float(m.scale)
    sdf_over_scale = vertex_sdf / scale

    names = list(thresholds.keys())
    meds = np.array([thresholds[n] for n in names])
    labels = np.array(["unknown"] * len(sdf_over_scale), dtype=object)
    finite = np.isfinite(sdf_over_scale)
    nearest = np.argmin(np.abs(sdf_over_scale[finite, None] - meds[None, :]), axis=1)
    labels[finite] = [names[j] for j in nearest]
    return labels, sdf_over_scale


# ---------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="exact path: dir with Stage_*.npz fits")
    ap.add_argument("--mesh_dir", default=None, help="unused except for provenance in --out; kept for eval_run.py-style invocation symmetry")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--raw_mesh", default=None, help="SDF fallback path: a single un-fitted .obj")
    ap.add_argument(
        "--num_samples",
        type=int,
        default=3000,
        help="face-sample budget for the SDF fallback path. -1 samples every face, as "
        "SDF_batch.py's own CLI defaults to -- fine for the ~20k-face template but expensive "
        "on real scans that can carry 5x+ more faces (e.g. bench50_clean: 92704 faces on "
        "Acanthostichus_aff.brevicornis, confirmed by direct count). 3000 matches the density "
        "sdf_prior_PROBE.py already used for real-scan SDF checks (`sample_surface(m, 4000)`).",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.raw_mesh:
        thresholds = calibrate_sdf_thresholds()
        print("template SDF/scale medians by region:", thresholds)
        labels, sdf_vals = sdf_region_labels(args.raw_mesh, thresholds, num_samples=args.num_samples)
        uniq, counts = np.unique(labels, return_counts=True)
        for u, c in zip(uniq, counts):
            print(f"  {u:10s} n={c}")
        if args.out:
            with open(args.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["vertex_id", "region", "sdf_over_scale"])
                for i, (lab, sv) in enumerate(zip(labels, sdf_vals)):
                    w.writerow([i, lab, sv])
        return

    if not args.run_dir:
        raise SystemExit("need --run_dir (exact path) or --raw_mesh (SDF fallback)")
    rows = region_deform_table(args.run_dir, args.mesh_dir, stage=args.stage)
    out = args.out or os.path.join(args.run_dir, "region_deform.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
