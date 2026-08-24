#!/usr/bin/env python3
"""
Retrieval-based leg-pose initializer (raw-geometry, non-learned).
2026-08-20, follow-up to the pose-tolerance sweep.

Motivation
----------
The pose-noise sweep showed D1 tolerates leg-pose error up to ~15 deg with
little cost, and degrades from ~20 deg onward. The PCA/heuristic cheap
initializer (cheap_anatomical_init.py / simple_leg_heuristic.py) produced a
mean leg-joint error of 28.5 deg -- squarely inside the degraded regime,
which is why it hurt correspondence instead of helping it.

This script tests a different, still non-learned strategy before building
anything neural: nearest-neighbor RETRIEVAL from a library of synthetic
(point-cloud descriptor, pose) pairs. No training, no ground truth at
inference time -- just "which library specimen's shape looks most like this
one, and what pose did that specimen have".

Method
------
1. For every specimen in the synthetic corpus, build a canonical-frame,
   pose-sensitive descriptor of its point cloud:
     a. PCA-canonicalize orientation (remove rotation, fix axis signs
        deterministically via 3rd-moment skew so two specimens with the
        same pose produce the same descriptor regardless of how they were
        originally oriented).
     b. Build a coarse cylindrical occupancy histogram (radius x angle x
        height) around the body's long axis. Because legs are the
        pose-varying parts sticking out radially, this histogram is
        sensitive to leg configuration while staying cheap and robust to
        density/noise.
2. Leave-one-out over the corpus: for each query specimen, retrieve the
   nearest OTHER specimen by descriptor distance and copy its ground-truth
   joint_rot as the initializer for the query's legs.
3. Score the retrieved joint_rot against the query's true joint_rot with
   the identical per-leg-joint axis-angle magnitude error metric used in
   pose_noise_sweep.py (leg_chains-derived rows, mean geodesic rotation
   error in degrees), so results land on the same scale as the 5/10/.../30
   deg table and the cheap-init 28.5 deg number.

Repo-specific adapter
----------------------
Only ONE function needs filling in: `load_specimen_points`, which must
return an (N, 3) point cloud for a given specimen name from your synth_clean
corpus. A few common path patterns are tried automatically first; if none
match your layout, edit the function body (marked `# >>> ADAPT`).

Everything else (canonicalization, descriptor, retrieval, scoring, output
format) is self-contained and does not need editing.

Usage
-----
python retrieval_pose_init.py \
    --corpus diagnostics/moonshot/synth_clean \
    --model 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl \
    --out diagnostics/anatomical_pose_init/out_ceiling_20260820/retrieval_init.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

# Reused as-is from pose_noise_sweep.py -- this import already works in your repo.
from fitter_3d.geom_leg_init import leg_chains  # noqa: E402


# ---------------------------------------------------------------------------
# Adapter: point cloud loading (fill in / adjust if the auto-tried patterns
# below don't match your corpus layout)
# ---------------------------------------------------------------------------

def load_specimen_points(corpus_dir: str, name: str) -> np.ndarray:
    """Return an (N, 3) float array of scan points for `name`.

    # >>> ADAPT: if none of these common patterns match your synth_clean
    # layout, replace the body of this function with whatever loads your
    # per-specimen point cloud (e.g. a .ply/.npy/.npz you already write
    # during synthetic generation).
    """
    candidates = [
        Path(corpus_dir) / name / "points.npy",
        Path(corpus_dir) / name / "scan.npy",
        Path(corpus_dir) / f"{name}_points.npy",
        Path(corpus_dir) / name / "points.npz",
    ]
    for c in candidates:
        if c.exists():
            if c.suffix == ".npz":
                d = np.load(c, allow_pickle=True)
                key = "points" if "points" in d else list(d.keys())[0]
                return np.asarray(d[key], dtype=np.float64)
            return np.asarray(np.load(c), dtype=np.float64)

    # Fall back to a .ply if present (uses trimesh if available)
    ply_candidates = [
        Path(corpus_dir) / name / "scan.ply",
        Path(corpus_dir) / f"{name}.ply",
    ]
    for c in ply_candidates:
        if c.exists():
            try:
                import trimesh
                mesh = trimesh.load(str(c), process=False)
                pts = mesh.vertices if hasattr(mesh, "vertices") else mesh
                return np.asarray(pts, dtype=np.float64)
            except ImportError:
                pass

    raise FileNotFoundError(
        f"Could not find a point cloud for specimen '{name}' under {corpus_dir}. "
        f"Edit load_specimen_points() in this script to match your synth_clean "
        f"layout (tried: {[str(c) for c in candidates + ply_candidates]})."
    )


# ---------------------------------------------------------------------------
# PCA canonicalization with deterministic sign resolution
# ---------------------------------------------------------------------------

def pca_canonicalize(points: np.ndarray) -> np.ndarray:
    """Center + rotate points onto their PCA axes (descending variance),
    then fix the sign of each axis via 3rd-moment skew so that repeated
    calls on differently-oriented but same-pose specimens converge to the
    same canonical frame."""
    p = points - points.mean(axis=0, keepdims=True)
    # PCA via SVD on centered points
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    axes = vt  # (3, 3), rows = principal axes, descending variance
    proj = p @ axes.T  # (N, 3) in PCA frame

    # Deterministic sign fix: flip each axis so its 3rd moment (skew) is
    # positive. This is a standard, cheap way to resolve PCA's +/- axis
    # ambiguity without needing any semantic (head/tail) knowledge.
    skew = np.mean(proj ** 3, axis=0)
    signs = np.where(skew < 0, -1.0, 1.0)
    proj = proj * signs

    return proj


# ---------------------------------------------------------------------------
# Pose-sensitive descriptor: cylindrical occupancy histogram around the
# long (first PCA) axis
# ---------------------------------------------------------------------------

def cylindrical_descriptor(
    points_canonical: np.ndarray,
    n_height: int = 10,
    n_angle: int = 16,
    n_radius: int = 6,
) -> np.ndarray:
    """Bin canonical-frame points into (height x angle x radius) cells
    around axis 0 (the longest/body axis), normalized to a unit-sum
    histogram so it's roughly robust to point-count differences."""
    x, y, z = points_canonical[:, 0], points_canonical[:, 1], points_canonical[:, 2]
    r = np.sqrt(y ** 2 + z ** 2)
    theta = np.arctan2(z, y)  # [-pi, pi]

    # Robust range via percentiles to reduce sensitivity to outlier points
    x_lo, x_hi = np.percentile(x, [0.5, 99.5])
    r_hi = np.percentile(r, 99.5) + 1e-8

    x_bins = np.linspace(x_lo, x_hi, n_height + 1)
    r_bins = np.linspace(0.0, r_hi, n_radius + 1)
    theta_bins = np.linspace(-np.pi, np.pi, n_angle + 1)

    hist, _ = np.histogramdd(
        np.stack([x, theta, r], axis=1),
        bins=[x_bins, theta_bins, r_bins],
    )
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.flatten()


def build_descriptor(points: np.ndarray) -> np.ndarray:
    canon = pca_canonicalize(points)
    return cylindrical_descriptor(canon)


# ---------------------------------------------------------------------------
# Leg-joint error metric (identical in spirit to pose_noise_sweep.py's
# sanity check, using scipy for the geodesic rotation distance)
# ---------------------------------------------------------------------------

def leg_rows_from_names(jnames):
    chains = leg_chains(jnames)
    return sorted({idx - 1 for chain in chains.values() for idx in chain})


def leg_joint_error_deg(jr_pred: np.ndarray, jr_gt: np.ndarray, leg_rows) -> float:
    """Mean geodesic rotation error in degrees over the leg-joint rows.
    jr_pred, jr_gt: (54, 3) axis-angle arrays for a single specimen."""
    import scipy.spatial.transform as st

    Ra = st.Rotation.from_rotvec(jr_pred[leg_rows])
    Rb = st.Rotation.from_rotvec(jr_gt[leg_rows])
    err = (Ra.inv() * Rb).magnitude() * 180.0 / np.pi
    return float(err.mean())


# ---------------------------------------------------------------------------
# Main: leave-one-out retrieval + evaluation
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_clean")
    ap.add_argument("--model", default="3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
    ap.add_argument(
        "--out",
        default="diagnostics/anatomical_pose_init/out_ceiling_20260820/retrieval_init.json",
    )
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    jnames = list(dd["J_names"])
    leg_rows = leg_rows_from_names(jnames)

    gt = np.load(os.path.join(args.corpus, "ground_truth.npz"), allow_pickle=True)
    jr_gt_all = gt["joint_rot"]   # (N, 54, 3)
    verts_all = gt["verts"]       # (N, 10235, 3) -- posed mesh vertices, already the point cloud
    names = [str(n) for n in gt["names"]]
    n_spec = len(names)

    print(f"[retrieval_pose_init] building descriptors for {n_spec} specimens ...")
    descriptors = np.stack([build_descriptor(verts_all[i]) for i in range(n_spec)], axis=0)
    per_specimen_err = {}
    retrieved_from = {}

    for i, query_name in enumerate(names):
        # Leave-one-out: exclude the query itself from the library
        dists = np.linalg.norm(descriptors - descriptors[i:i + 1], axis=1)
        dists[i] = np.inf
        j = int(np.argmin(dists))

        jr_pred = jr_gt_all[j]  # legs come from retrieved neighbor's GT pose
        jr_true = jr_gt_all[i]
        err = leg_joint_error_deg(jr_pred, jr_true, leg_rows)

        per_specimen_err[query_name] = err
        retrieved_from[query_name] = names[j]

    errs = np.array(list(per_specimen_err.values()))
    mean_err = float(errs.mean())
    median_err = float(np.median(errs))
    frac_under_15 = float((errs <= 15.0).mean())

    result = {
        "method": "retrieval_pca_cylindrical_histogram_leave_one_out",
        "n_specimens": n_spec,
        "per_specimen_leg_rot_err_deg": per_specimen_err,
        "retrieved_from": retrieved_from,
        "summary": {
            "mean_leg_rot_err_deg": mean_err,
            "median_leg_rot_err_deg": median_err,
            "frac_specimens_under_15deg": frac_under_15,
        },
        "comparison": {
            "zero_init_mean_deg": 23.087,
            "cheap_pca_heuristic_init_mean_deg": 28.520,
            "gt_init_mean_deg": 0.0,
            "robust_basin_upper_bound_deg": 15.0,
        },
        "note": (
            "Leave-one-out nearest-neighbor retrieval over the 12-specimen "
            "synth_clean corpus is a small-library test -- treat this as a "
            "feasibility check, not a claim about retrieval at scale. If "
            "mean error is well under ~15 deg, retrieval is a viable next "
            "step (e.g. grown to a larger synthetic library); if it is "
            "similar to or worse than the cheap heuristic, raw shape "
            "descriptors alone are not separating poses well enough and a "
            "learned initializer becomes the more justified next step."
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[retrieval_pose_init] mean leg-joint error = {mean_err:.3f} deg "
          f"(median {median_err:.3f}), {frac_under_15*100:.0f}% of specimens under 15 deg")
    print(f"[retrieval_pose_init] wrote {args.out}")


if __name__ == "__main__":
    main()