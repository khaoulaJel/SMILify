"""CSE-style continuous-embedding correspondence head for SMIL point clouds (Variant 2).

WHY THIS EXISTS
---------------
Variant 1 (`smil_correspondence_net.SMILCorrespondenceNet`) predicts a discrete segment class plus a
1-D within-segment position. That representation is closed for a structural reason established in
this investigation: a leg segment is a 2-D tube, and a 1-D scalar cannot uniquely address a 2-D
surface -- confirmed three independent ways (degenerate `chain_pos`; real circumferential 3D spread
of 11-18% of segment length inside a narrow position band; near-zero class-controlled residual R^2
on the trained features). No authored UV coordinates exist anywhere in the model, so there is no
cheap coordinate to substitute.

This module replaces ONLY the decode head and its loss. The backbone (sa1/sa2/sa3 + fp3/fp2/fp1),
the corrected sampler, the training pipeline and the evaluation harness all carry over unchanged.
`smil_correspondence_net.py` is deliberately left byte-identical: it feeds the live D1 chain.

LITERATURE THIS IMPLEMENTS
--------------------------
* CSE -- Neverova et al., "Continuous Surface Embeddings", NeurIPS 2020 (arXiv:2011.12438).
  Predict a per-point embedding; hold a learned embedding per template vertex; correspond by
  nearest neighbour. Removes DensePose's discretisation seams -- our exact failure mode.
* CoE -- Zeng, Gao & Cremers, 3DV 2025 (arXiv:2412.05557). The point-cloud-native realisation of
  the same idea (CSE itself is image-based), robust to noise and partiality: our regime.
* SurfEmb -- Haugaard & Buch, CVPR 2022 (arXiv:2111.13489). Learns the correspondence distribution
  with a CONTRASTIVE query/key objective and no symmetry annotation. This is why the loss below is
  InfoNCE over vertex identity rather than an L2/triplet regression onto a fixed target.

EMPIRICAL JUSTIFICATION FOR THE CONTRASTIVE CHOICE (not a stylistic preference)
------------------------------------------------------------------------------
`diagnostics/anatomical_pose_init/RESULTS_cse_feasibility_20260826.md`, measured on the frozen B2
backbone over 12 held-out specimens: cross-specimen retrieval closes 0.27-0.55 of the measured
random->ceiling gap, but decomposed by direction the AXIAL component closes 0.14-0.57 while the
CIRCUMFERENTIAL component closes only 0.09-0.20 -- near chance on every segment, negative on `ta`.
The residual ambiguity is specifically circumferential, i.e. the near-rotationally-symmetric
direction. That is precisely the ambiguity SurfEmb's contrastive formulation exists to represent,
and precisely what a deterministic regression head cannot.

WHAT THIS DOES NOT CLAIM
------------------------
The feasibility numbers justify this as a well-founded bet, NOT a predicted success: 0.27-0.55 is
real signal well above chance and nowhere near the ceiling. `ta` showed no usable cross-specimen
signal and `pt` was never testable (8 template vertices). All evidence so far is synthetic, single
corpus, shared topology; nothing here speaks to bench50 real scans.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pointnet2_utils import (  # noqa: E402
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
)


class SMILCSENet(nn.Module):
    """Per-point continuous embedding + a learned per-template-vertex embedding table.

    Backbone shapes are IDENTICAL to `SMILCorrespondenceNet` (320/640/1024, fp3/fp2/fp1 -> 128),
    so a trained Variant-1 checkpoint can warm-start this network exactly -- see
    `load_backbone_from_correspondence_ckpt`. Only the head differs.

    forward(x: (B,N,3)) -> query embeddings (B,N,D), L2-normalised.
    `vertex_embeddings()` -> (V,D), L2-normalised keys, one per template vertex.

    Both sides are unit-norm so inner product == cosine similarity, and nearest neighbour by cosine
    is equivalent to nearest neighbour by Euclidean distance on the sphere -- meaning the retrieval
    performed at inference is exactly the quantity the InfoNCE loss optimises, with no
    train/inference metric mismatch.
    """

    def __init__(self, n_vertices, embed_dim=16, temperature=0.07, learn_temperature=True):
        super().__init__()
        self.n_vertices = n_vertices
        self.embed_dim = embed_dim

        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 0,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fp3 = PointNetFeaturePropagation(1024 + 640, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # query head: per-point 128-d backbone feature -> D-d embedding
        self.embed_head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, embed_dim, 1),
        )
        # key model: one free embedding per template vertex (CSE's formulation). A table is used
        # rather than SurfEmb's coordinate-MLP because our topology is FIXED across every specimen
        # in the corpus, so vertex identity is a stable index -- there is nothing to generalise
        # across, and a coordinate-MLP would smooth over exactly the fine circumferential
        # distinctions this head exists to preserve.
        self.vertex_embed = nn.Embedding(n_vertices, embed_dim)
        nn.init.normal_(self.vertex_embed.weight, std=0.02)

        # log-temperature, learned (CLIP-style), clamped in `logit_scale` so the softmax cannot
        # saturate early in training.
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)),
                                     requires_grad=learn_temperature)

    def logit_scale(self):
        cap = torch.log(torch.tensor(100.0, device=self.log_temp.device))
        return self.log_temp.clamp(max=cap).exp()

    def backbone(self, x):
        """(B,N,3) -> per-point features (B,128,N). Mirrors SMILCorrespondenceNet.forward's path."""
        xyz = x.transpose(2, 1)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        return self.fp1(xyz, l1_xyz, None, l1_points)

    def forward(self, x):
        q = self.embed_head(self.backbone(x)).transpose(2, 1)   # (B,N,D)
        return F.normalize(q, dim=-1)

    def vertex_embeddings(self, idx=None):
        w = self.vertex_embed.weight if idx is None else self.vertex_embed(idx)
        return F.normalize(w, dim=-1)


def info_nce_loss(q, true_vert, model, n_negatives=4096, generator=None):
    """SurfEmb-style contrastive loss over TRUE VERTEX IDENTITY.

    q          : (B,N,D) L2-normalised query embeddings
    true_vert  : (B,N) long, each point's true template vertex index
    n_negatives: how many extra template vertices to draw as negative keys

    The key set is the union of the batch's true vertices (so every positive is guaranteed present)
    and a uniform random sample of other template vertices -- SurfEmb's "uniformly sampled object
    points as negative keys". A full softmax over all V vertices is avoided because B*N*V logits do
    not fit in memory at realistic batch sizes; sampled softmax over a large negative set is the
    standard substitute and preserves the ranking the metric cares about.

    Returns (loss, accuracy over the sampled key set). That accuracy is an OPTIMISTIC proxy -- it
    scores against ~n_negatives candidates, not all V -- so it is for monitoring convergence only.
    Report retrieval from the evaluation harness, never this number.
    """
    B, N, D = q.shape
    flat_q = q.reshape(B * N, D)
    flat_v = true_vert.reshape(B * N)

    device = q.device
    n_neg = min(n_negatives, model.n_vertices)
    neg_ids = torch.randint(0, model.n_vertices, (n_neg,), device=device, generator=generator)
    key_ids = torch.unique(torch.cat([torch.unique(flat_v), neg_ids]))   # sorted, (K,)

    keys = model.vertex_embeddings(key_ids)                              # (K,D)
    logits = model.logit_scale() * (flat_q @ keys.t())                   # (B*N, K)

    target = torch.searchsorted(key_ids, flat_v)   # valid: key_ids is sorted and contains flat_v
    loss = F.cross_entropy(logits, target)
    acc = (logits.argmax(1) == target).float().mean()
    return loss, acc


def build_geodesic_soft_targets(v_template, faces, n_neighbors=24, sigma=None, group=None):
    """F6: per-template-vertex soft target support, on the mesh's own EDGE GRAPH.

    Returns (idx (V,M) int64, w (V,M) float32) -- for each vertex, its M nearest vertices by graph
    distance (itself first) and a Gaussian weight exp(-d^2 / 2 sigma^2), row-normalised.

    WHY GRAPH DISTANCE AND NOT EUCLIDEAN: adjacent coxae are physically close but not connected by
    edges. Euclidean neighbourhoods would leak target mass across legs -- which is exactly the
    ambiguity E0_identity measured (a perfect mesh still misassigns ~9% of coxal points). Graph
    distance cannot jump that gap.

    sigma defaults to the median distance to the 8th graph-neighbour, so the smoothing scale is
    read off the mesh rather than chosen.
    """
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(
        _o.path.abspath(__file__)))), "diagnostics", "correspondence_accuracy"))
    import numpy as _np
    from scipy.sparse.csgraph import dijkstra as _dij
    from geodesic import build_edge_graph as _beg

    G = _beg(_np.asarray(v_template, dtype=_np.float64), _np.asarray(faces))
    V = G.shape[0]
    D = _dij(G, directed=False) if sigma is None else _dij(G, directed=False, limit=4.0 * sigma)
    order = _np.argsort(D, axis=1)[:, :n_neighbors]
    d = _np.take_along_axis(D, order, axis=1)
    if sigma is None:
        # 3rd graph-neighbour, not 8th: at the 8th the target goes nearly uniform over the
        # neighbourhood (measured mean self-weight 0.080 vs 1/24 = 0.042), which destroys the
        # identity signal the loss exists to teach.
        finite = d[:, min(3, n_neighbors - 1)]
        sigma = float(_np.median(finite[_np.isfinite(finite)]))
    w = _np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    w[~_np.isfinite(d)] = 0.0
    leaked = 0.0
    if group is not None:
        # HARD CONSTRAINT: target mass may never cross a group boundary (leg -> different leg).
        # Graph distance can still bridge adjacent coxae through the thorax -- measured at up to
        # 46% of one vertex's mass before this mask -- and that is precisely the confusion
        # E0_identity showed is irreducible in the metric. Teaching it would be teaching the error.
        g = _np.asarray(group, dtype=object)
        cross = (g[order] != g[:, None]) & (g[:, None] != "") & (g[order] != "")
        leaked = float((w * cross).sum() / max(w.sum(), 1e-12))
        w = w * (~cross)
    w /= _np.maximum(w.sum(1, keepdims=True), 1e-12)
    print(f"[F6] geodesic soft targets: sigma={sigma:.6f} (median 3rd-neighbour graph distance), "
          f"M={n_neighbors}, mean self-weight={w[:, 0].mean():.3f}, "
          f"cross-group mass removed={leaked:.5f}", flush=True)
    return torch.as_tensor(order, dtype=torch.long), torch.as_tensor(w, dtype=torch.float32)


def info_nce_soft_loss(q, true_vert, model, soft_idx, soft_w, n_negatives=4096, generator=None):
    """F6: InfoNCE with GEODESIC-SOFTENED targets instead of a one-hot vertex label.

    The stock loss (`info_nce_loss`) is F.cross_entropy against a single true vertex, so on a
    10,235-vertex mesh the vertex 0.1mm away is punished exactly as hard as one on the opposite
    leg -- while the evaluation harness and E0's measured 0.0908 metric floor both say near-misses
    are cheap. This spreads target mass over the true vertex's graph neighbourhood instead.

    The key set is extended with the neighbours of the batch's true vertices, so the soft mass has
    somewhere to land; without that the target would silently collapse back to one-hot.
    """
    B, N, D = q.shape
    flat_q = q.reshape(B * N, D)
    flat_v = true_vert.reshape(B * N)
    device = q.device

    nbr = soft_idx.to(device)[flat_v]                                    # (BN,M)
    n_neg = min(n_negatives, model.n_vertices)
    neg_ids = torch.randint(0, model.n_vertices, (n_neg,), device=device, generator=generator)
    key_ids = torch.unique(torch.cat([torch.unique(nbr), neg_ids]))      # neighbours guaranteed in
    keys = model.vertex_embeddings(key_ids)
    logits = model.logit_scale() * (flat_q @ keys.t())                   # (BN,K)

    pos = torch.searchsorted(key_ids, nbr)                               # (BN,M), valid by construction
    tgt = torch.zeros_like(logits)
    tgt.scatter_(1, pos, soft_w.to(device)[flat_v])
    tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-12)

    loss = -(tgt * F.log_softmax(logits, dim=1)).sum(1).mean()
    hard = torch.searchsorted(key_ids, flat_v)
    acc = (logits.argmax(1) == hard).float().mean()
    return loss, acc


def info_nce_loss_hard(q, true_vert, model, hard_pool, hard_frac=0.5, n_negatives=4096,
                       generator=None):
    """InfoNCE with CIRCUMFERENTIAL HARD NEGATIVES.

    WHY: the LRF check (lrf_wellposedness_20260827.py, toy-verified metric) showed the
    circumferential disambiguating signal IS present in raw geometry -- real ant leg segments are
    tapered/flattened, scoring circum_gap 0.38-0.44 against a perfect cylinder's 0.031 floor. Yet
    circumferential retrieval sits near chance. So the information is present and UNLEARNED, not
    unrepresentable -- which makes this a training problem, not an architecture problem.

    Uniformly-sampled negatives are almost all trivially easy: a random template vertex is usually
    on a different segment entirely, so the loss is minimised long before circumferential detail
    matters. `hard_pool` supplies, per vertex, the vertices that are in the SAME segment at a
    SIMILAR axial position -- i.e. differing mainly circumferentially. These are exactly the
    distinctions the network currently fails to make, so forcing them into the denominator is the
    targeted fix.

    hard_pool : (V, P) long tensor of candidate hard negatives per vertex (padded with -1)
    hard_frac : fraction of the negative budget drawn from the hard pool
    """
    B, N, D = q.shape
    flat_q = q.reshape(B * N, D)
    flat_v = true_vert.reshape(B * N)
    device = q.device

    n_total = min(n_negatives, model.n_vertices)
    n_hard = int(n_total * hard_frac)
    n_easy = n_total - n_hard

    easy = torch.randint(0, model.n_vertices, (n_easy,), device=device, generator=generator)

    # draw hard negatives from the pools of the vertices actually present in this batch
    present = torch.unique(flat_v)
    pools = hard_pool[present]                                   # (S,P)
    valid = pools >= 0
    flat_pool = pools[valid]
    if flat_pool.numel() > 0 and n_hard > 0:
        sel = torch.randint(0, flat_pool.numel(), (n_hard,), device=device, generator=generator)
        hard = flat_pool[sel]
    else:
        hard = torch.empty(0, dtype=torch.long, device=device)

    key_ids = torch.unique(torch.cat([present, easy, hard]))
    keys = model.vertex_embeddings(key_ids)
    logits = model.logit_scale() * (flat_q @ keys.t())
    target = torch.searchsorted(key_ids, flat_v)
    loss = F.cross_entropy(logits, target)
    acc = (logits.argmax(1) == target).float().mean()
    return loss, acc


def build_circumferential_hard_pool(v_template, seg_class_id, seg_axis, axial_band=0.15,
                                    max_pool=64):
    """For each leg-segment vertex, the same-segment vertices at a SIMILAR axial position.

    Those differ from it mainly CIRCUMFERENTIALLY, which is precisely the distinction the trained
    head currently fails to make. Body vertices get empty pools (padded -1) -- body correspondence
    was never the failing case.

    axial_band: fraction of the segment's own axial extent within which a vertex counts as "similar
                axial position". 0.15 keeps the pool genuinely circumferential rather than just
                same-segment.
    """
    import numpy as np
    V = len(v_template)
    pool = np.full((V, max_pool), -1, dtype=np.int64)
    for cid, (axis, length) in seg_axis.items():
        idx = np.where(seg_class_id == cid)[0]
        if len(idx) < 2:
            continue
        t = (v_template[idx] - v_template[idx].mean(0)) @ axis      # axial coordinate
        for a, i in enumerate(idx):
            near = idx[np.abs(t - t[a]) <= axial_band * max(length, 1e-9)]
            near = near[near != i]
            if len(near) == 0:
                continue
            if len(near) > max_pool:
                near = np.random.default_rng(int(i)).choice(near, max_pool, replace=False)
            pool[i, :len(near)] = near
    return torch.as_tensor(pool)


def load_backbone_from_correspondence_ckpt(model, ckpt_path, map_location="cpu"):
    """Warm-start the backbone from a trained Variant-1 (`SMILCorrespondenceNet`) checkpoint.

    Copies ONLY sa*/fp* weights -- the two heads differ by construction and stay at random init.
    Returns (n_copied, n_skipped) so callers can assert the transfer actually happened rather than
    silently training from scratch off a typo'd path.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    src = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    own = model.state_dict()
    copied, skipped = 0, 0
    for k, v in src.items():
        if not (k.startswith("sa") or k.startswith("fp")):
            continue
        if k in own and own[k].shape == v.shape:
            own[k] = v
            copied += 1
        else:
            skipped += 1
    model.load_state_dict(own)
    return copied, skipped
