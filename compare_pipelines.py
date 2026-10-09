#!/usr/bin/env python3
"""
Batch comparison: original (Weld-based) vs alpha_wrap pipeline, across all specimens under
a comparison_output/.../specimens/ directory, each containing alphawrap/*.obj and original/*.obj.

Metric suite (grounded in the mesh-reconstruction-evaluation literature, see methodology notes):
  - Topology: watertight, manifold, self-intersection-free, component count (trimesh built-ins)
  - Volume + surface area (sanity: inflation/deflation, not just "did it run")
  - Bidirectional Chamfer Distance (CD) and Hausdorff Distance @ p99.9 between the two outputs
    (standard reconstruction-benchmark pair, e.g. used throughout recent watertight-repair papers)
  - F-score @ threshold = k * specimen's own median edge length (scale-normalized, not a fixed
    global distance - critical across specimens spanning an 8x size range)
  - Normal consistency between nearest-point-matched surface samples
  - Shape Diameter Function (SDF) thin-feature proxy (Shapira, Shamir & Cohen-Or 2008): for each
    of N sampled surface points, cast a ray along -normal and record the distance to the opposite
    surface - a direct, standard estimate of LOCAL THICKNESS. This is the metric that would have
    caught the Pheidole leg-crushing failure automatically: if alpha_wrap's p1 thickness collapses
    relative to original's at the same specimen, that's the smoking gun for "decimation destroyed
    a thin feature," independent of any visual inspection.
  - reconstruction approach actually used per specimen (alpha_wrap vs manifold_plus_fallback_
    from_alpha_wrap) and that specimen's own fidelity_gate_passed verdict, parsed from the
    alphawrap.log FINAL_SUMMARY line if present alongside the specimen's obj files. This matters:
    the two reconstruction paths carry different formal guarantees (alpha_wrap has a strict
    enclosure guarantee, ManifoldPlus does not), so lumping them into one undifferentiated
    "alphawrap" bucket would hide which guarantee actually applies to a given specimen's output.

Statistical aggregation: paired two-sided Wilcoxon signed-rank test per metric across all
specimens (standard for exactly this kind of paired same-subject, non-normal-distributed,
small-to-moderate-N comparison - used throughout recent reconstruction/registration benchmark
papers), not just mean/median differences, so "looks better on average" can be distinguished
from "consistently better vs noisy." Also run separately on the alpha_wrap-only subset, since
that's the reconstruction path with the actual formal guarantee.

Usage:
    pip install trimesh scipy numpy --break-system-packages
    python3 compare_pipelines.py /home/nao48500/SMILify/comparison_output/bench50_clean_20260814_125245_Glc7oS/specimens \
        --out comparison_report

Outputs:
    <out>/per_specimen.csv        - full metric table, one row per specimen
    <out>/summary.json            - aggregate stats + Wilcoxon results per metric
    <out>/flagged_for_review.txt  - specimens where a specific failure signature was detected
                                     (thin-feature collapse, fidelity outlier, topology regression,
                                     ManifoldPlus fallback, fidelity gate failed at generation time)
                                     - THESE are the ones to open visually, not a random sample.
    stdout                        - the actual go/no-go read, printed at the end
"""

import argparse
import csv
import json
import os
import re
import sys
import warnings

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

N_SAMPLE_CD = 20000       # points sampled per surface for Chamfer/Hausdorff/F-score/normal-consistency
N_SAMPLE_SDF = 3000       # points sampled per surface for shape-diameter-function thickness estimate
N_SAMPLE_SDF_CONE = 1000  # points sampled per surface for the cone-averaged SDF validation pass
                          # (fewer than N_SAMPLE_SDF since each point now costs n_rays intersects
                          # instead of 1 - see shape_diameter_thickness_cone, opt-in only)
F_SCORE_EDGE_MULT = 2.0   # F-score threshold = this * median edge length of the ORIGINAL mesh
SDF_THICKNESS_RATIO_FLAG = 0.5   # flag if alphawrap's p1 thickness < this fraction of original's
FIDELITY_OUTLIER_MAD_MULT = 5.0  # flag if a specimen's CD/HD is > this many MADs from the batch median

FINAL_SUMMARY_RE = re.compile(r"FINAL_SUMMARY:\s*(.*)")
FINAL_SUMMARY_KV_RE = re.compile(r"(\w+)=(\S+)")


def find_specimen_pairs(root):
    pairs = []
    for entry in sorted(os.listdir(root)):
        spec_dir = os.path.join(root, entry)
        if not os.path.isdir(spec_dir):
            continue
        aw_dir = os.path.join(spec_dir, "alphawrap")
        orig_dir = os.path.join(spec_dir, "original")
        aw_obj = _first_obj(aw_dir)
        orig_obj = _first_obj(orig_dir)
        if aw_obj and orig_obj:
            pairs.append((entry, orig_obj, aw_obj, spec_dir))
        else:
            print(f"SKIP {entry}: missing obj (alphawrap={bool(aw_obj)}, original={bool(orig_obj)})",
                  file=sys.stderr)
    return pairs


def _first_obj(d):
    if not os.path.isdir(d):
        return None
    for root, _dirs, files in os.walk(d):
        for f in sorted(files):
            if f.lower().endswith(".obj"):
                return os.path.join(root, f)
    return None


def parse_final_summary(spec_dir):
    """Parse the FINAL_SUMMARY line out of logs/alphawrap.log, if present. Returns a dict of
    whatever key=value pairs were on that line (approach, fidelity_gate_passed, etc.), or {}
    if the log/line isn't there - callers must not assume any particular key exists."""
    log_path = os.path.join(spec_dir, "logs", "alphawrap.log")
    if not os.path.isfile(log_path):
        return {}
    last_match = None
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = FINAL_SUMMARY_RE.search(line)
            if m:
                last_match = m.group(1)
    if last_match is None:
        return {}
    kv = dict(FINAL_SUMMARY_KV_RE.findall(last_match))
    out = {}
    for k, v in kv.items():
        if v in ("True", "False"):
            out[k] = (v == "True")
        else:
            try:
                out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                out[k] = v
    return out


def median_edge_length(mesh):
    edges = mesh.edges_unique_length
    return float(np.median(edges)) if len(edges) else float("nan")


def largest_component(mesh):
    """Returns the largest (by face count) connected component of mesh. The 'original' pipeline's
    output is frequently non-watertight and fragmented into dozens of disconnected debris islands
    (confirmed: up to 66 components / 1106 holes on some bench50 specimens - this is the known,
    expected failure mode of the deprecated pipeline, not a script bug). Running SDF thickness
    ray-casting over the whole fragmented mesh lets stray debris islands dominate the low
    percentiles used for thin-feature detection, which is exactly backwards for a metric whose
    job is judging the coherent body's own thin features. Restrict to the largest component so
    the thickness comparison reflects the actual ant body, not debris."""
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        return mesh
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda p: len(p.faces))


def shape_diameter_thickness(mesh, n_samples):
    """Shapira et al. 2008 shape-diameter-function-style local thickness estimate: sample surface
    points, cast a ray along the inward normal, record distance to the opposite surface. Returns
    the array of valid thickness samples (rays that actually hit something). Runs on the mesh's
    largest connected component only - see largest_component().

    KNOWN LIMITATION (flagged 2026-08-17, not yet resolved as of this function): this is a
    single-ray-per-point estimate. The original Shapira et al. paper casts a CONE of rays per
    point (typically ~30deg half-angle) with outlier rejection and averaging, specifically because
    one ray can escape through a small hole/crack in an imperfect mesh (both pipelines' outputs
    can be non-watertight) or clip a locally unrepresentative bit of geometry, producing a wildly
    wrong single-point estimate. This is a concrete, literature-grounded explanation for the
    p1/p5 disagreement noted elsewhere in this file and is a plausible confound in
    thickness_p5_ratio_aw_over_orig at the per-specimen level. See
    shape_diameter_thickness_cone() below for the multi-ray comparison added to validate this
    before trusting single-ray severity rankings any further."""
    mesh = largest_component(mesh)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_idx]
    origins = points - normals * 1e-4  # nudge inward off the surface to avoid self-hit
    directions = -normals
    locations, index_ray, _ = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    if len(locations) == 0:
        return np.array([])
    dists = np.linalg.norm(locations - origins[index_ray], axis=1)
    return dists


def _cone_directions(normal, n_rays, half_angle_rad, rng):
    """n_rays unit directions within half_angle_rad of `normal` (the FIRST direction is always
    exactly `normal` itself, so a cone of n_rays=1 degenerates to the plain single-ray case)."""
    normal = normal / np.linalg.norm(normal)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    dirs = [normal]
    for _ in range(n_rays - 1):
        theta = half_angle_rad * np.sqrt(rng.uniform(0.0, 1.0))  # area-uniform within the cone
        phi = rng.uniform(0.0, 2 * np.pi)
        d = (np.cos(theta) * normal + np.sin(theta) * (np.cos(phi) * u + np.sin(phi) * v))
        dirs.append(d / np.linalg.norm(d))
    return np.array(dirs)


def shape_diameter_thickness_cone(mesh, n_samples, n_rays=7, half_angle_deg=30.0,
                                   outlier_mad_mult=2.0, seed=0):
    """Cone-averaged local thickness estimate, matching Shapira et al. 2008's actual method
    (single-ray sampling above is the simplified/fragile variant): per surface sample point, cast
    n_rays rays within half_angle_deg of the inward normal (not just the normal itself), reject
    hit-distance outliers per point (> outlier_mad_mult median-absolute-deviations from that
    point's own median), and average the survivors into one thickness value per point. A point is
    dropped (not counted) if fewer than 2 of its rays hit anything after outlier rejection - this
    is deliberately conservative rather than falling back to a possibly-bad single ray.

    Returns the array of valid per-point averaged thickness samples (may be shorter than
    n_samples). Runs on the mesh's largest connected component only - see largest_component().
    Added 2026-08-17 to check whether shape_diameter_thickness()'s single-ray estimate is noise-
    dominated at the per-specimen level (it demonstrably is at least once, per the p1/p5
    disagreement) before trusting its severity ranking to guide further reconstruction-level
    engineering effort."""
    mesh = largest_component(mesh)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_idx]
    rng = np.random.default_rng(seed)
    half_angle_rad = np.deg2rad(half_angle_deg)

    all_origins = []
    all_directions = []
    point_id = []
    for i in range(len(points)):
        dirs = _cone_directions(-normals[i], n_rays, half_angle_rad, rng)
        origin = points[i] - normals[i] * 1e-4
        all_origins.append(np.tile(origin, (n_rays, 1)))
        all_directions.append(dirs)
        point_id.append(np.full(n_rays, i))
    all_origins = np.concatenate(all_origins, axis=0)
    all_directions = np.concatenate(all_directions, axis=0)
    point_id = np.concatenate(point_id, axis=0)

    locations, index_ray, _ = mesh.ray.intersects_location(
        all_origins, all_directions, multiple_hits=False
    )
    if len(locations) == 0:
        return np.array([])
    dists = np.linalg.norm(locations - all_origins[index_ray], axis=1)
    hit_point_id = point_id[index_ray]

    out = []
    for i in range(len(points)):
        d = dists[hit_point_id == i]
        if len(d) < 2:
            continue
        med = np.median(d)
        mad = np.median(np.abs(d - med)) or 1e-9
        keep = d[np.abs(d - med) <= outlier_mad_mult * mad]
        if len(keep) < 2:
            continue
        out.append(float(np.mean(keep)))
    return np.array(out)


def compute_pair_metrics(name, orig_path, aw_path, spec_dir):
    orig = trimesh.load(orig_path, process=False)
    aw = trimesh.load(aw_path, process=False)

    row = {"specimen": name}

    final_summary = parse_final_summary(spec_dir)
    row["recon_approach"] = final_summary.get("approach", "unknown")
    row["recon_fidelity_gate_passed"] = final_summary.get("fidelity_gate_passed", None)
    row["recon_worst_fidelity_deviation"] = final_summary.get("worst_fidelity_deviation", float("nan"))

    row["orig_vertices"] = len(orig.vertices)
    row["orig_faces"] = len(orig.faces)
    row["aw_vertices"] = len(aw.vertices)
    row["aw_faces"] = len(aw.faces)

    row["orig_watertight"] = bool(orig.is_watertight)
    row["aw_watertight"] = bool(aw.is_watertight)
    row["orig_winding_consistent"] = bool(orig.is_winding_consistent)
    row["aw_winding_consistent"] = bool(aw.is_winding_consistent)
    row["orig_n_components"] = orig.body_count
    row["aw_n_components"] = aw.body_count

    row["orig_volume"] = float(orig.volume) if orig.is_watertight else float("nan")
    row["aw_volume"] = float(aw.volume) if aw.is_watertight else float("nan")
    row["orig_surface_area"] = float(orig.area)
    row["aw_surface_area"] = float(aw.area)
    if orig.is_watertight and aw.is_watertight and orig.volume > 0:
        row["volume_ratio_aw_over_orig"] = float(aw.volume / orig.volume)
    else:
        row["volume_ratio_aw_over_orig"] = float("nan")

    orig_med_edge = median_edge_length(orig)
    row["orig_median_edge_length"] = orig_med_edge

    # --- Chamfer / Hausdorff / F-score / normal consistency (bidirectional, orig <-> alphawrap) ---
    orig_pts, orig_fidx = trimesh.sample.sample_surface(orig, N_SAMPLE_CD)
    aw_pts, aw_fidx = trimesh.sample.sample_surface(aw, N_SAMPLE_CD)
    orig_normals = orig.face_normals[orig_fidx]
    aw_normals = aw.face_normals[aw_fidx]

    tree_orig = cKDTree(orig_pts)
    tree_aw = cKDTree(aw_pts)
    d_o2a, idx_o2a = tree_aw.query(orig_pts)
    d_a2o, idx_a2o = tree_orig.query(aw_pts)

    row["chamfer_mean"] = float((d_o2a.mean() + d_a2o.mean()) / 2)
    row["hausdorff_p99_9"] = float(max(np.percentile(d_o2a, 99.9), np.percentile(d_a2o, 99.9)))
    row["hausdorff_max"] = float(max(d_o2a.max(), d_a2o.max()))

    f_thresh = F_SCORE_EDGE_MULT * orig_med_edge if np.isfinite(orig_med_edge) and orig_med_edge > 0 else 1.0
    precision = float(np.mean(d_a2o < f_thresh))   # fraction of alphawrap points close to original
    recall = float(np.mean(d_o2a < f_thresh))       # fraction of original points close to alphawrap
    row["f_score_threshold"] = f_thresh
    row["f_score"] = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    nc_o2a = np.abs(np.sum(orig_normals * aw_normals[idx_o2a], axis=1))
    nc_a2o = np.abs(np.sum(aw_normals * orig_normals[idx_a2o], axis=1))
    row["normal_consistency"] = float((nc_o2a.mean() + nc_a2o.mean()) / 2)

    # --- Shape diameter function: thin-feature (leg/antenna) thickness proxy ---
    try:
        orig_thick = shape_diameter_thickness(orig, N_SAMPLE_SDF)
        aw_thick = shape_diameter_thickness(aw, N_SAMPLE_SDF)
        row["orig_thickness_p1"] = float(np.percentile(orig_thick, 1)) if len(orig_thick) else float("nan")
        row["aw_thickness_p1"] = float(np.percentile(aw_thick, 1)) if len(aw_thick) else float("nan")
        row["orig_thickness_p5"] = float(np.percentile(orig_thick, 5)) if len(orig_thick) else float("nan")
        row["aw_thickness_p5"] = float(np.percentile(aw_thick, 5)) if len(aw_thick) else float("nan")
        if row["orig_thickness_p1"] and np.isfinite(row["orig_thickness_p1"]) and row["orig_thickness_p1"] > 0:
            row["thickness_p1_ratio_aw_over_orig"] = float(row["aw_thickness_p1"] / row["orig_thickness_p1"])
        else:
            row["thickness_p1_ratio_aw_over_orig"] = float("nan")
        # p5 is included alongside p1 deliberately: sanity-checking against 'original's non-
        # watertight, hole-riddled output (confirmed on real specimens, see largest_component()
        # docstring) showed p1 is noise-dominated - a handful of rays escaping through a hole in
        # original produce erratic huge/tiny hit distances that flip the p1 ratio's sign entirely
        # relative to p5 on the SAME specimen. p5 is far less sensitive to a few degenerate rays
        # and is the more trustworthy read; p1 is kept for transparency, not as the primary signal.
        if row["orig_thickness_p5"] and np.isfinite(row["orig_thickness_p5"]) and row["orig_thickness_p5"] > 0:
            row["thickness_p5_ratio_aw_over_orig"] = float(row["aw_thickness_p5"] / row["orig_thickness_p5"])
        else:
            row["thickness_p5_ratio_aw_over_orig"] = float("nan")
    except Exception as e:
        row["thickness_p1_ratio_aw_over_orig"] = float("nan")
        row["sdf_error"] = str(e)[:200]

    # --- Cone-averaged thickness (opt-in, COMPARE_PIPELINES_CONE_SDF=1): validates whether the
    # single-ray SDF above is noise-dominated at the per-specimen level - see
    # shape_diameter_thickness_cone() docstring. Env-var gated (not a CLI flag) to avoid touching
    # the multiprocessing worker signature for what is currently a one-off validation pass, not a
    # production metric.
    if os.environ.get("COMPARE_PIPELINES_CONE_SDF") == "1":
        try:
            orig_thick_c = shape_diameter_thickness_cone(orig, N_SAMPLE_SDF_CONE)
            aw_thick_c = shape_diameter_thickness_cone(aw, N_SAMPLE_SDF_CONE)
            row["orig_thickness_p1_cone"] = float(np.percentile(orig_thick_c, 1)) if len(orig_thick_c) else float("nan")
            row["aw_thickness_p1_cone"] = float(np.percentile(aw_thick_c, 1)) if len(aw_thick_c) else float("nan")
            row["orig_thickness_p5_cone"] = float(np.percentile(orig_thick_c, 5)) if len(orig_thick_c) else float("nan")
            row["aw_thickness_p5_cone"] = float(np.percentile(aw_thick_c, 5)) if len(aw_thick_c) else float("nan")
            if row["orig_thickness_p1_cone"] and np.isfinite(row["orig_thickness_p1_cone"]) and row["orig_thickness_p1_cone"] > 0:
                row["thickness_p1_ratio_cone_aw_over_orig"] = float(row["aw_thickness_p1_cone"] / row["orig_thickness_p1_cone"])
            else:
                row["thickness_p1_ratio_cone_aw_over_orig"] = float("nan")
            if row["orig_thickness_p5_cone"] and np.isfinite(row["orig_thickness_p5_cone"]) and row["orig_thickness_p5_cone"] > 0:
                row["thickness_p5_ratio_cone_aw_over_orig"] = float(row["aw_thickness_p5_cone"] / row["orig_thickness_p5_cone"])
            else:
                row["thickness_p5_ratio_cone_aw_over_orig"] = float("nan")
        except Exception as e:
            row["cone_sdf_error"] = str(e)[:200]

    return row


def _wilcoxon_one_sample(vals, label, results, n_min=2):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < n_min:
        return
    try:
        stat, p = wilcoxon(vals - 1.0)
    except ValueError:
        stat, p = float("nan"), float("nan")
    results[label] = {
        "n": int(len(vals)), "median": float(np.median(vals)),
        "wilcoxon_stat": float(stat), "p_value": float(p),
    }


def aggregate(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(out_dir, "per_specimen.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- flags: specific, actionable, not "everything looks fine" ---
    flagged = []
    cd_vals = np.array([r["chamfer_mean"] for r in rows if np.isfinite(r.get("chamfer_mean", np.nan))])
    cd_med = np.median(cd_vals) if len(cd_vals) else 0.0
    cd_mad = np.median(np.abs(cd_vals - cd_med)) if len(cd_vals) else 1.0
    cd_mad = cd_mad if cd_mad > 0 else 1e-9

    for r in rows:
        reasons = []
        if not r.get("aw_watertight", True):
            reasons.append("alphawrap NOT watertight")
        if not r.get("aw_winding_consistent", True):
            reasons.append("alphawrap winding inconsistent")
        if r.get("aw_n_components", 1) != 1:
            reasons.append(f"alphawrap has {r.get('aw_n_components')} components (expected 1)")
        if r.get("orig_n_components", 1) is not None and r.get("orig_n_components", 1) > 5:
            reasons.append(f"original is heavily fragmented ({r.get('orig_n_components')} components) "
                            f"- thickness_p1_ratio below is computed on original's LARGEST component "
                            f"only, not this debris field, but treat with extra caution")
        tr5 = r.get("thickness_p5_ratio_aw_over_orig", float("nan"))
        if np.isfinite(tr5) and tr5 < SDF_THICKNESS_RATIO_FLAG:
            reasons.append(f"thin-feature collapse: p5 thickness ratio={tr5:.3f} "
                            f"(alphawrap thinnest features are <{SDF_THICKNESS_RATIO_FLAG}x original's)")
        tr = r.get("thickness_p1_ratio_aw_over_orig", float("nan"))
        if np.isfinite(tr) and tr < SDF_THICKNESS_RATIO_FLAG and not (np.isfinite(tr5) and tr5 < SDF_THICKNESS_RATIO_FLAG):
            reasons.append(f"thin-feature collapse (p1 only, p5 disagrees - treat p1-only flags as "
                            f"lower confidence): p1 thickness ratio={tr:.3f}")
        cd = r.get("chamfer_mean", float("nan"))
        if np.isfinite(cd) and abs(cd - cd_med) > FIDELITY_OUTLIER_MAD_MULT * cd_mad:
            reasons.append(f"chamfer distance outlier: {cd:.3g} vs batch median {cd_med:.3g} "
                            f"({(cd - cd_med) / cd_mad:.1f} MADs)")
        if r.get("recon_approach") == "manifold_plus_fallback_from_alpha_wrap":
            reasons.append("reconstructed via ManifoldPlus fallback, not alpha_wrap - "
                            "no formal enclosure guarantee for this specimen")
        if r.get("recon_fidelity_gate_passed") is False:
            reasons.append("pipeline's own fidelity_gate_passed=False at generation time "
                            f"(worst_fidelity_deviation={r.get('recon_worst_fidelity_deviation')})")
        if reasons:
            flagged.append((r["specimen"], reasons))

    with open(os.path.join(out_dir, "flagged_for_review.txt"), "w") as f:
        if not flagged:
            f.write("No specimens flagged - still spot-check 2-3 manually before trusting this.\n")
        for name, reasons in flagged:
            f.write(f"{name}:\n")
            for reason in reasons:
                f.write(f"  - {reason}\n")

    # --- paired Wilcoxon signed-rank test per metric where both sides exist ---
    paired_metrics = [
        ("vertices", "orig_vertices", "aw_vertices"),
        ("volume_ratio (aw/orig, should cluster near 1.0 if unbiased)", None, "volume_ratio_aw_over_orig"),
        ("thickness_p1_ratio (aw/orig)", None, "thickness_p1_ratio_aw_over_orig"),
        ("thickness_p5_ratio (aw/orig) - more robust to ray-hit noise, see docstring", None,
         "thickness_p5_ratio_aw_over_orig"),
    ]
    wilcoxon_results = {}
    for label, orig_key, aw_key in paired_metrics:
        if orig_key is None:
            vals = [r[aw_key] for r in rows if np.isfinite(r.get(aw_key, np.nan))]
            _wilcoxon_one_sample(vals, label, wilcoxon_results)
        else:
            a = np.array([r.get(orig_key, np.nan) for r in rows], dtype=float)
            b = np.array([r.get(aw_key, np.nan) for r in rows], dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 2:
                continue
            diffs = b[mask] - a[mask]
            try:
                stat, p = wilcoxon(diffs)
            except ValueError:
                stat, p = float("nan"), float("nan")
            wilcoxon_results[label] = {
                "n": int(mask.sum()), "median_diff_aw_minus_orig": float(np.median(diffs)),
                "wilcoxon_stat": float(stat), "p_value": float(p),
            }

    # Also run the thickness-ratio test restricted to alpha_wrap-only specimens (the
    # reconstruction path with the actual formal enclosure guarantee) - a batch-wide result can
    # be dominated or diluted by the ManifoldPlus-fallback subset, which is a different method
    # with different failure modes and should not be silently averaged in.
    for metric_key, metric_label in [("thickness_p1_ratio_aw_over_orig", "thickness_p1_ratio"),
                                      ("thickness_p5_ratio_aw_over_orig", "thickness_p5_ratio")]:
        aw_only_vals = [r[metric_key] for r in rows if r.get("recon_approach") == "alpha_wrap"]
        _wilcoxon_one_sample(aw_only_vals, f"{metric_label} (aw/orig), alpha_wrap-only subset",
                             wilcoxon_results, n_min=2)
        mp_only_vals = [r[metric_key] for r in rows
                        if r.get("recon_approach") == "manifold_plus_fallback_from_alpha_wrap"]
        _wilcoxon_one_sample(mp_only_vals, f"{metric_label} (aw/orig), manifold_plus-fallback-only subset",
                             wilcoxon_results, n_min=2)

    n_alpha_wrap = sum(1 for r in rows if r.get("recon_approach") == "alpha_wrap")
    n_manifold_plus = sum(1 for r in rows if r.get("recon_approach") == "manifold_plus_fallback_from_alpha_wrap")
    n_gate_failed = sum(1 for r in rows if r.get("recon_fidelity_gate_passed") is False)

    summary = {
        "n_specimens": len(rows),
        "n_flagged": len(flagged),
        "n_recon_approach_alpha_wrap": n_alpha_wrap,
        "n_recon_approach_manifold_plus_fallback": n_manifold_plus,
        "n_recon_fidelity_gate_failed_at_generation_time": n_gate_failed,
        "chamfer_mean_median": float(cd_med),
        "chamfer_mean_mad": float(cd_mad),
        "wilcoxon_tests": wilcoxon_results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary, flagged


def _compute_pair_metrics_worker(args):
    """Top-level wrapper (must be module-level to be picklable for multiprocessing) - runs
    compute_pair_metrics for one specimen and turns any exception into a return value instead of
    letting it kill the worker, so one bad specimen never silently drops from the aggregate."""
    import traceback
    name, orig_path, aw_path, spec_dir = args
    try:
        row = compute_pair_metrics(name, orig_path, aw_path, spec_dir)
        return name, row, None
    except Exception as e:
        tb = traceback.format_exc()
        return name, {"specimen": name, "error": str(e)[:300]}, tb


def _run_pool(pairs, workers, label, jsonl_path, done_names, rows, errors):
    """Runs pairs (list of (name, orig_path, aw_path, spec_dir)) either sequentially (workers<=1)
    or via a spawn-context Pool, appending each result to jsonl_path as it lands so a kill/stall
    never loses already-computed specimens - confirmed necessary: two earlier runs (one fork-based,
    one spawn-based) both deadlocked ~40+ specimens in with ALL results only held in memory, so
    killing the stuck run threw away everything. Skips any name already in done_names (resume)."""
    todo = [p for p in pairs if p[0] not in done_names]
    if not todo:
        return
    print(f"{label}: {len(todo)} specimen(s) to process with {workers} worker(s) "
          f"({len(pairs) - len(todo)} already done, skipped)...")
    with open(jsonl_path, "a") as jf:
        def _record(name, row, tb):
            rows.append(row)
            done_names.add(name)
            jf.write(json.dumps({"specimen": name, "row": row}) + "\n")
            jf.flush()
            if tb:
                print(f"  ERROR: {row.get('error')}", file=sys.stderr)
                errors.append((name, tb))

        if workers <= 1:
            for i, item in enumerate(todo):
                print(f"[{label} {i+1}/{len(todo)}] {item[0]}...")
                name, row, tb = _compute_pair_metrics_worker(item)
                _record(name, row, tb)
        else:
            import multiprocessing as mp
            # 'spawn', not the default 'fork': rtree/libspatialindex (used by trimesh's ray
            # intersector) holds internal locks that can be inherited mid-locked across fork()
            # and deadlock in the child. spawn re-imports cleanly in each worker instead of
            # cloning parent state - confirmed this alone does NOT fully solve it (see
            # LARGE_FILE_MB split below); kept because it is still strictly safer than fork.
            ctx = mp.get_context("spawn")
            done = 0
            with ctx.Pool(processes=workers) as pool:
                for name, row, tb in pool.imap_unordered(_compute_pair_metrics_worker, todo):
                    done += 1
                    print(f"[{label} {done}/{len(todo)}] {name} done.")
                    _record(name, row, tb)


# Specimens whose alphawrap .obj exceeds this size get processed with --large-workers
# concurrency instead of --workers. Confirmed necessary: running 4 workers each on a mesh in the
# 50-170MB / 700K+ vertex range (multiple concurrent large SDF ray-casts + rtree index builds)
# reliably deadlocked TWICE in a row - all workers stuck in futex_do_wait/pipe_read with ZERO
# additional CPU time over 10+ minutes - once under fork, once again under spawn. Since spawn
# didn't fix it, this isn't (only) a fork-safety issue - it's resource contention from several
# huge concurrent ray-casts on a shared, already-busy login node. Capping concurrency specifically
# for the large tail avoids the pile-up while keeping the (safe, fast) full parallelism for the
# many small specimens.
LARGE_FILE_MB = 40.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specimens_dir")
    ap.add_argument("--out", default="comparison_report")
    ap.add_argument("--errors-out", default=None,
                     help="If given, write per-specimen tracebacks for any specimen that errored "
                          "during metric computation to this file, instead of only stderr.")
    ap.add_argument("--workers", type=int, default=1,
                     help="Number of specimens under LARGE_FILE_MB to process in parallel. Be "
                          "considerate of shared/login-node CPU usage when raising this.")
    ap.add_argument("--large-workers", type=int, default=1,
                     help="Concurrency for specimens at/above LARGE_FILE_MB (see docstring above "
                          "LARGE_FILE_MB - concurrent large ray-casts deadlocked at --workers=4).")
    ap.add_argument("--resume-jsonl", default=None,
                     help="Path to a previous run's incremental results jsonl to resume from "
                          "(specimens already in it are skipped). Defaults to <out>/_progress.jsonl.")
    args = ap.parse_args()

    pairs = find_specimen_pairs(args.specimens_dir)
    print(f"Found {len(pairs)} specimen pairs.")

    os.makedirs(args.out, exist_ok=True)
    jsonl_path = args.resume_jsonl or os.path.join(args.out, "_progress.jsonl")

    rows = []
    errors = []
    done_names = set()
    if os.path.isfile(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["specimen"] not in done_names:
                    done_names.add(rec["specimen"])
                    rows.append(rec["row"])
        if done_names:
            print(f"Resuming: {len(done_names)} specimen(s) already in {jsonl_path}, skipping them.")

    small_pairs, large_pairs = [], []
    for p in pairs:
        name, orig_path, aw_path, spec_dir = p
        size_mb = os.path.getsize(aw_path) / 1e6
        (large_pairs if size_mb >= LARGE_FILE_MB else small_pairs).append(p)

    _run_pool(small_pairs, args.workers, "small", jsonl_path, done_names, rows, errors)
    _run_pool(large_pairs, args.large_workers, "large", jsonl_path, done_names, rows, errors)

    if args.errors_out and errors:
        with open(args.errors_out, "w") as f:
            for name, tb in errors:
                f.write(f"===== {name} =====\n{tb}\n\n")

    summary, flagged = aggregate(rows, args.out)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\n{len(flagged)}/{len(rows)} specimens flagged for manual review "
          f"(see {args.out}/flagged_for_review.txt) - open these first, not a random sample.")
    print(f"Full per-specimen table: {args.out}/per_specimen.csv")
    if errors:
        print(f"{len(errors)} specimen(s) ERRORED during metric computation - "
              f"see {args.errors_out or 'stderr'}.")

    print("\n===== READ =====")
    tr_wilcoxon = summary["wilcoxon_tests"].get(
        "thickness_p5_ratio (aw/orig) - more robust to ray-hit noise, see docstring")
    if tr_wilcoxon:
        med = tr_wilcoxon["median_diff_aw_minus_orig"] if "median_diff_aw_minus_orig" in tr_wilcoxon else tr_wilcoxon.get("median")
        p = tr_wilcoxon["p_value"]
        print(f"Thin-feature thickness ratio (aw/orig, p5): median={med:.3f}, p={p:.3g}")
        if p < 0.05 and (tr_wilcoxon.get("median", 1.0) < 0.9):
            print("  -> STATISTICALLY SIGNIFICANT thin-feature loss in alpha_wrap vs original across "
                  "the batch. This is a real signal, not noise - do not proceed without fixing "
                  "decimate_mesh's stopping criterion first.")
        else:
            print("  -> No significant systematic thin-feature loss detected across the batch.")
    print(f"{len(flagged)} specimens flagged individually regardless of the aggregate trend - "
          f"a clean aggregate p-value does not clear a flagged specimen.")
    print(f"Reconstruction approach split: {summary['n_recon_approach_alpha_wrap']} alpha_wrap, "
          f"{summary['n_recon_approach_manifold_plus_fallback']} manifold_plus_fallback (no formal "
          f"enclosure guarantee), {summary['n_recon_fidelity_gate_failed_at_generation_time']} "
          f"failed the pipeline's own fidelity gate at generation time.")


if __name__ == "__main__":
    main()
