"""GBCPD-inspired geodesic motion-coherence term, added to the existing optimizer.

WHY THIS EXISTS, and how it differs from everything closed so far
Everything tested to this point (SDF/geodesic/hull partitioning, HKS, raw and refined DINO,
kinematic-chain arc-length ordering) attacked the within-part problem by giving each vertex a
better FINGERPRINT for independent per-point nearest-neighbour search. None of them touch the
SEARCH mechanism itself. Geodesic-Based (Bayesian) Coherent Point Drift (Kondo et al., TPAMI
2022) is a structurally different attack: standard CPD "unnaturally deforms the different parts
of a shape... when they are neighbouring each other" because its motion-coherence prior is
defined by EUCLIDEAN proximity -- a folded leg touching the thorax gets pulled to move like the
thorax, because Euclidean distance can't tell they're different parts. GBCPD redefines that
coherence using GEODESIC (on-surface) distance instead.

`probreg` (pip) has plain Bayesian CPD but its coherence kernel is Euclidean, confirmed by
`grep -r geodesic` over its source returning nothing -- using it as-is would test a different,
weaker hypothesis (Euclidean BCPD, already implicitly related to what's closed) not GBCPD's
actual mechanism. Reimplementing GBCPD's full Bayesian-EM formulation faithfully is a
substantially larger undertaking than fits this investigation's remaining scope. Instead, per
the two options offered when this lever was proposed, this implements the SECOND: geodesic
motion coherence as an ADDED REGULARISATION TERM inside the existing gradient-descent optimizer
(`fitter_3d/trainer_moonshot.py`), not a from-scratch alternative registration architecture.
This tests the actual hypothesis (does geodesic, not Euclidean, motion coherence help) without
the much larger engineering risk of a full CPD-family reimplementation.

MECHANISM
For a sampled set of anchor vertices (reusing the SAME geodesic Dijkstra computation already
cached for the refinement-layer work, `diagnostics/khaoula_v2/out/template_geodesic.npz` -- not
recomputed), a Gaussian kernel in GEODESIC distance gives each anchor a soft neighbourhood.
Within that neighbourhood, this penalises the VARIANCE of per-vertex total motion (current fit
position minus rest pose) -- i.e. geodesically-close vertices are pushed to move alike,
regardless of their CURRENT Euclidean proximity (which is exactly what a folded, touching limb
would otherwise exploit). Computed via the weighted second-moment identity
(Var = E[x^2] - E[x]^2) so cost is O(n_anchors x V) via matmul, not O(n_anchors x V x V).

NOT a claim of matching GBCPD's own reported results or its exact Bayesian-EM formulation --
this is the geodesic-motion-coherence PRINCIPLE, integrated into the pipeline already validated
in this investigation (T0's harness), scored the identical way.
"""

import os

import numpy as np
import torch

from fitter_3d.trainer_moonshot import MoonshotStage


def geo_weight_matrix(geodesic_npz_path, sigma_frac=0.05, device="cuda"):
    """(n_anchors, V) Gaussian-kernel weights from the cached anchor-to-all geodesic distances.

    sigma = sigma_frac * mesh geodesic diameter. 0.05 is a first, reasoned pass (roughly one
    leg-segment's scale on this template), NOT a swept/tuned value -- flagged explicitly, same
    discipline as scale_cap's initial 0.052 before its own (never-run) sweep.
    """
    d = np.load(geodesic_npz_path)
    anchor_idx, dist = d["anchor_idx"], d["dist"]  # (A,), (A, V)
    diam = float(dist.max())
    sigma = sigma_frac * diam
    W = np.exp(-(dist**2) / (2 * sigma**2))
    return torch.tensor(W, dtype=torch.float32, device=device), anchor_idx, sigma, diam


def geo_coherence_loss(displacement, W):
    """displacement: (B, V, 3). W: (A, V) fixed geodesic Gaussian weights (no grad).

    Weighted variance of displacement within each anchor's geodesic neighbourhood, via the
    second-moment identity (avoids an O(A x V x 3) intermediate): for anchor a,
        Var_a = [sum_v W(a,v) ||d_v||^2 / sum_v W(a,v)] - ||mu_a||^2,  mu_a = sum_v W(a,v) d_v / sum_v W(a,v)
    averaged over anchors and batch.
    """
    Wsum = W.sum(1).clamp_min(1e-6)  # (A,)
    mu = torch.einsum("av,bvc->bac", W, displacement) / Wsum[None, :, None]  # (B, A, 3)
    d2 = (displacement**2).sum(-1)  # (B, V)
    e_d2 = torch.einsum("av,bv->ba", W, d2) / Wsum[None, :]  # (B, A)
    var = (e_d2 - (mu**2).sum(-1)).clamp_min(0.0)  # (B, A)
    return var.mean()


class GeoCoherentMoonshotStage(MoonshotStage):
    """MoonshotStage + w_geocoh: geodesic motion-coherence, unchanged otherwise.

    `w_geocoh` follows the SAME `.get(key, 0.0)` opt-in pattern every other addition in
    trainer_moonshot.py uses (w_scale, w_trans, ...) -- default 0.0, i.e. inert unless a config
    sets it, so this subclass is a strict superset of the validated base class's behaviour.
    """

    def __init__(self, *args, geodesic_npz_path=None, geocoh_sigma_frac=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self._geo_W = None
        if self.loss_weights.get("w_geocoh", 0.0) > 0:
            assert geodesic_npz_path and os.path.isfile(geodesic_npz_path), (
                "w_geocoh > 0 requires geodesic_npz_path pointing at a cached "
                "template_geodesic.npz (diagnostics/khaoula_v2/train_refine_autoencoder.py)."
            )
            W, anchor_idx, sigma, diam = geo_weight_matrix(
                geodesic_npz_path, sigma_frac=geocoh_sigma_frac, device=self.device
            )
            self._geo_W = W
            print(f"[geocoh] loaded {W.shape[0]} anchors, sigma={sigma:.4f} "
                  f"({geocoh_sigma_frac}x geodesic diameter {diam:.4f})", flush=True)

    def forward(self, src_mesh, it=0):
        loss, comp = super().forward(src_mesh, it)
        lw = self.loss_weights
        if lw.get("w_geocoh", 0.0) > 0 and self._geo_W is not None:
            cur = src_mesh.verts_padded()  # (B, V, 3), current fitted position
            displacement = cur - self.rest_verts  # rest_verts already captured by the base class
            l_gc = geo_coherence_loss(displacement, self._geo_W)
            comp["geocoh"] = l_gc
            loss = loss + lw["w_geocoh"] * l_gc
        return loss, comp
