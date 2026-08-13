"""
Post-hoc evaluation metrics for SMIL template-to-scan fitting.

These are *evaluation* metrics, not training losses. They run after
optimisation completes and measure fit quality in ways the training
losses (chamfer, edge, normal, laplacian) cannot.

Metric 1 -- F-score at threshold tau
--------------------------------------
Given fitted-mesh samples P and scan samples Q, at distance threshold tau:

    precision = |{p in P : min_q ||p - q|| < tau}| / |P|
    recall    = |{q in Q : min_p ||q - p|| < tau}| / |Q|
    F-score   = 2 * precision * recall / (precision + recall + epsilon)

Why this over Chamfer: Chamfer averages all distances into one scalar.
A fit where 95% of the surface is perfect but one leg is 4mm off
will have a reasonable Chamfer but a visibly lower F-score -- and
sweeping tau shows *at what tolerance* the fit breaks down.

What it still does NOT measure: correspondence correctness. A smooth
but semantically wrong fit (head on gaster) can still score high
F-score if the surfaces happen to be nearby. That is addressed by
later metrics (geodesic error, part-IoU).

References:
    - Tatarchenko et al., "What Do Single-View 3D Reconstruction
      Networks Learn?", CVPR 2019 -- introduced F-score for 3D.
    - Knapitsch et al., "Tanks and Temples", ACM TOG 2017.
    - The watertight-remeshing benchmark (CelloCut) uses F-score
      alongside CD, HD, and ANC as its standard evaluation suite.
"""

import torch
from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.structures import Meshes


def f_score(
    pred_meshes: Meshes,
    target_meshes: Meshes,
    tau_fractions: list[float] = [0.01, 0.02, 0.05],
    n_samples: int = 10_000,
) -> dict:
    """
    Compute F-score between predicted (fitted) and target (scan) meshes
    at multiple thresholds, per specimen.

    Args:
        pred_meshes:    PyTorch3D Meshes, the fitted SMIL template after
                        optimisation (batch_size meshes).
        target_meshes:  PyTorch3D Meshes, the input scans (same batch_size).
        tau_fractions:  List of threshold fractions relative to each
                        specimen's bounding-box diagonal. 0.01 = 1%.
        n_samples:      Number of points to sample from each mesh surface.
                        10000 is standard; 5000 is fine for quick checks.

    Returns:
        dict with keys:
            "thresholds_used"  : list of tau fraction values
            "precision"        : Tensor (batch, n_thresholds)
            "recall"           : Tensor (batch, n_thresholds)
            "f_score"          : Tensor (batch, n_thresholds)
            "bbox_diag"        : Tensor (batch,) -- per-specimen BB diagonal
    """

    # --- sample points uniformly from both mesh surfaces ----------------
    pred_points = sample_points_from_meshes(pred_meshes, n_samples)    # (B, N, 3)
    target_points = sample_points_from_meshes(target_meshes, n_samples)  # (B, N, 3)

    batch_size = pred_points.shape[0]

    # --- per-specimen bounding-box diagonal (computed on target scan) ----
    # This normalises the threshold so it is scale-invariant across
    # specimens of different physical size.
    t_min = target_points.min(dim=1).values   # (B, 3)
    t_max = target_points.max(dim=1).values   # (B, 3)
    bbox_diag = (t_max - t_min).norm(dim=1)   # (B,)

    # --- nearest-neighbour distances in both directions -----------------
    # knn_points returns (dists_squared, idx, nn_points)
    # pred -> target: for each predicted point, distance to nearest target
    d_pred_to_target = knn_points(pred_points, target_points, K=1).dists.squeeze(-1).sqrt()  # (B, N)
    # target -> pred: for each target point, distance to nearest predicted
    d_target_to_pred = knn_points(target_points, pred_points, K=1).dists.squeeze(-1).sqrt()  # (B, N)

    # --- compute precision / recall / F at each threshold ---------------
    n_thresholds = len(tau_fractions)
    precision = torch.zeros(batch_size, n_thresholds, device=pred_points.device)
    recall = torch.zeros(batch_size, n_thresholds, device=pred_points.device)
    f_scores = torch.zeros(batch_size, n_thresholds, device=pred_points.device)

    for t_idx, tau_frac in enumerate(tau_fractions):
        tau = tau_frac * bbox_diag  # (B,) -- absolute threshold per specimen

        # precision: fraction of predicted points within tau of a target point
        precision[:, t_idx] = (d_pred_to_target < tau.unsqueeze(1)).float().mean(dim=1)

        # recall: fraction of target points within tau of a predicted point
        recall[:, t_idx] = (d_target_to_pred < tau.unsqueeze(1)).float().mean(dim=1)

        # F-score: harmonic mean
        p = precision[:, t_idx]
        r = recall[:, t_idx]
        f_scores[:, t_idx] = 2 * p * r / (p + r + 1e-8)

    return {
        "thresholds_used": tau_fractions,
        "precision": precision.detach().cpu(),
        "recall": recall.detach().cpu(),
        "f_score": f_scores.detach().cpu(),
        "bbox_diag": bbox_diag.detach().cpu(),
    }
