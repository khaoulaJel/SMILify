"""
Metric 4 -- Per-part precision (extends eval_metrics.py)
==========================================================

Why this metric, at this point:

Both chamfer and F-score sample points proportional to surface area.
We have direct evidence (specimen GAGA-02-08: mid-pack chamfer, 0.000911,
6th/10, but dead-last F-score@1%, 0.7681, 10/10 -- see
experiments/eval_test_v1/master_summary.csv) that this hides damage to
thin anatomical structures behind a strong score from large, easy-to-fit
regions like the gaster.

CORRECTION (checked against current data, see penetration_loss_deliverable/
1_axis_b_per_part_precision_failure/): an earlier version of this docstring
cited specimen 43-36 as an example of "visibly fragmented legs" hidden by a
strong F-score. That claim does NOT hold up against
experiments/eval_test_v1 and eval_test_v2 data for 43-36: per-part leg
precision is 0.9938 (not flagged), self-intersection is 0.0 on every one
of its six leg parts, displacement is among the lowest of the 10
specimens, and a direct render of its fitted mesh shows no visible leg
damage. Do not cite 43-36 as an example of hidden leg damage without
re-verifying against a specific run first -- the claim's origin (which
run, if any, actually showed this) could not be traced.

Per-part precision sidesteps this entirely: instead of sampling points
proportional to area, we use the *exact* template vertices belonging to
each anatomical part (known from the SMIL model's own skinning weights,
via part_groups.py -- no manual annotation needed) and ask, per part:
"how close is this specific anatomical region of the fit to the nearest
actual scan surface?"

Important scope note: this is a ONE-DIRECTIONAL metric (fitted part ->
scan). The scan has no part labels, so we cannot yet ask "what fraction
of the scan's actual leg is covered by the fit" (that would be
per-part recall / F-score, and requires scan-side segmentation --
e.g. via SDF thresholding, a natural next step but out of scope here).
What we CAN measure now, with zero additional data: whether the fitted
part vertices are close to *some* scan surface, which is precisely what
was needed to test whether GAGA-02-08-style leg damage is real and
localized (see correction above re: 43-36).

Known gap: part_groups.get_part_vertex_indices() assigns each vertex to
its argmax-skinning-weight joint. For SMIL_OmniAnt.pkl, the four waist
joints (w_1_r, w_2_r, w_1_l, w_2_l) have zero skinning weight over every
vertex (verified directly against the model file) -- they are pivot-only
joints with no owned mesh region. The "waist" part therefore always
comes back with 0 vertices and is silently skipped below; it will not
appear as a row in downstream tables. This is a property of the model
file, not a bug in this code.
"""

import torch
from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.structures import Meshes


def per_part_precision(
    fitted_verts: torch.Tensor,
    target_meshes: Meshes,
    part_vertex_indices: dict,
    tau_fractions: list[float] = [0.01, 0.02, 0.05],
    n_target_samples: int = 10_000,
) -> dict:
    """
    Compute per-anatomical-part precision: for each part, what fraction
    of that part's fitted vertices are within threshold tau of the
    nearest scan point.

    Args:
        fitted_verts:        Tensor (B, n_verts, 3) -- final fitted vertices
                              (e.g. from smal_3d_fitter() after optimisation).
        target_meshes:       PyTorch3D Meshes, the input scans (B meshes).
        part_vertex_indices: dict part_name -> np.array of vertex indices,
                              from part_groups.get_part_vertex_indices().
        tau_fractions:       threshold fractions of each specimen's bbox diagonal.
        n_target_samples:    points to sample from each scan surface.

    Returns:
        dict: part_name -> {
            "precision": Tensor (B, n_thresholds),
            "n_vertices": int,
        }
        plus "bbox_diag": Tensor (B,) and "thresholds_used": list.
        Parts with 0 vertices (see module docstring) are omitted.
    """
    device = fitted_verts.device
    batch_size = fitted_verts.shape[0]

    # Sample scan points once, reused across all parts
    target_points = sample_points_from_meshes(target_meshes, n_target_samples)  # (B, N, 3)

    t_min = target_points.min(dim=1).values
    t_max = target_points.max(dim=1).values
    bbox_diag = (t_max - t_min).norm(dim=1)  # (B,)

    n_thresholds = len(tau_fractions)
    results = {"bbox_diag": bbox_diag.detach().cpu(), "thresholds_used": tau_fractions}

    for part_name, v_idx in part_vertex_indices.items():
        if len(v_idx) == 0:
            continue

        idx_tensor = torch.as_tensor(v_idx, device=device, dtype=torch.long)
        part_verts = fitted_verts[:, idx_tensor, :]  # (B, n_part_verts, 3)

        # nearest scan point for each part vertex
        d_part_to_scan = knn_points(part_verts, target_points, K=1).dists.squeeze(-1).sqrt()  # (B, n_part_verts)

        precision = torch.zeros(batch_size, n_thresholds, device=device)
        for t_idx, tau_frac in enumerate(tau_fractions):
            tau = tau_frac * bbox_diag  # (B,)
            precision[:, t_idx] = (d_part_to_scan < tau.unsqueeze(1)).float().mean(dim=1)

        results[part_name] = {
            "precision": precision.detach().cpu(),
            "n_vertices": len(v_idx),
        }

    return results
