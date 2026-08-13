"""
Metric 3 -- Per-vertex displacement magnitude (formalized module)
=====================================================================

See the reasoning in report_section_displacement.md. Summary: deform_verts
is the free-form per-vertex offset unlocked only in Stage_2/Stage_3
(scheme 'deform'), after pose (joint_rot) and shape (betas) are already
fit. Its magnitude is a direct, mechanistic measure of how much the
optimizer had to deviate from a plausible parametric pose -- and unlike
chamfer/F-score, it carries no area bias, since every vertex counts
equally regardless of how much surface area surrounds it.
"""

import numpy as np


def displacement_stats(deform_verts: np.ndarray) -> dict:
    """
    Args:
        deform_verts: (n_verts, 3) array, the saved deform_verts for one
                      specimen (e.g. data["deform_verts"][0] from the npz).

    Returns:
        dict with mean, std, max, p95, p99 of the per-vertex displacement
        norm, plus the full per-vertex magnitude array for further
        analysis (e.g. per-part rollup).
    """
    magnitudes = np.linalg.norm(deform_verts, axis=1)  # (n_verts,)
    return {
        "mean": float(magnitudes.mean()),
        "std": float(magnitudes.std()),
        "max": float(magnitudes.max()),
        "p95": float(np.percentile(magnitudes, 95)),
        "p99": float(np.percentile(magnitudes, 99)),
        "per_vertex_magnitude": magnitudes,
    }


def displacement_stats_by_part(deform_verts: np.ndarray, part_vertex_indices: dict) -> dict:
    """
    Same as displacement_stats, but broken down per anatomical part.
    Useful for confirming WHICH part is absorbing the largest deformation
    on a given specimen, rather than just a single whole-mesh summary.

    Args:
        deform_verts:         (n_verts, 3) array for one specimen.
        part_vertex_indices:  dict part_name -> np.array of vertex indices,
                               from part_groups.get_part_vertex_indices().

    Returns:
        dict: part_name -> {"mean": ..., "std": ..., "max": ..., "p95": ..., "p99": ...}
    """
    magnitudes = np.linalg.norm(deform_verts, axis=1)  # (n_verts,)
    results = {}
    for part_name, idx in part_vertex_indices.items():
        if len(idx) == 0:
            continue
        part_mag = magnitudes[idx]
        results[part_name] = {
            "mean": float(part_mag.mean()),
            "std": float(part_mag.std()),
            "max": float(part_mag.max()),
            "p95": float(np.percentile(part_mag, 95)),
            "p99": float(np.percentile(part_mag, 99)),
        }
    return results
