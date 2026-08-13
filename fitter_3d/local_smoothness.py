"""
local_smoothness.py

Diagnostic-experiment support for the neighbor-drag hypothesis: does locally
down-weighting mesh smoothness (edge/laplacian) around currently-penetrating
vertices let the penetration loss actually reduce collision COUNT, rather
than only depth, by giving those vertices more freedom to move without
dragging their neighbors along?

This module is deliberately separate from penetration_loss.py (the
production loss) -- everything here is opt-in, used only by Stage's
optional `local_downweight` config (see trainer.py), and every existing
yaml config is unaffected because it never sets that option.

Two things are tested against each other, matched on COUNT and per-vertex
weight, differing only in spatial arrangement:
  - "local" mode: down-weight vertices/edges within k mesh-hops of a
    currently-penetrating vertex (khop_expand_mask) -- spatially clustered.
  - "global" mode: down-weight the SAME NUMBER of vertices each iteration,
    at the SAME "factor" weight, but drawn as a FRESH random subset of the
    whole mesh every iteration (random_touched_mask) -- spatially
    scattered. This is the control: if a positive result shows up in
    "global" too, it isn't localness doing the work, it's just that some
    equivalent amount of down-weighting happened somewhere.

    Deliberate design choice, not an accident: the random subset is
    RESAMPLED every iteration rather than fixed once. This isolates
    "clustered vs. scattered" cleanly, at the cost of giving "global" zero
    temporal persistence at any one location -- "local"'s target region
    tends to recur (the same collision persisting across iterations),
    while "global" never sustains attention on the same vertices twice. A
    positive result for "local" over "global" could therefore still be
    read either as "clustering matters" or "sustained targeting matters",
    not disentangled by this control alone.
"""

import numpy as np
import torch


def build_vertex_adjacency(faces: np.ndarray, n_verts: int, device: str) -> torch.Tensor:
    """
    Symmetric, unweighted (V, V) sparse adjacency from template faces.
    Built once -- shared topology, reused for every specimen and iteration.
    """
    edges = set()
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        edges.add((a, b)); edges.add((b, c)); edges.add((c, a))
        edges.add((b, a)); edges.add((c, b)); edges.add((a, c))
    edges = list(edges)
    idx = torch.tensor(edges, dtype=torch.long, device=device).t()  # (2, E)
    vals = torch.ones(idx.shape[1], device=device)
    adj = torch.sparse_coo_tensor(idx, vals, size=(n_verts, n_verts)).coalesce()
    return adj


def khop_expand_mask(seed_mask: torch.Tensor, adj: torch.Tensor, k: int) -> torch.Tensor:
    """
    Expands a boolean seed mask outward by k mesh-hops via k sparse
    adjacency multiplications (cheap: O(k * nnz(adj)), no shortest-path
    computation -- appropriate for calling every training iteration).

    Args:
        seed_mask: (V, B) float32 tensor, 1.0 where a vertex is currently
            penetrating, 0.0 elsewhere (transposed layout so a single
            sparse-dense matmul covers the whole specimen batch at once).
        adj: (V, V) sparse adjacency from build_vertex_adjacency.
        k: number of hops to expand.

    Returns:
        (V, B) float32 tensor, 1.0 where a vertex is within k hops of a
        seed vertex (including the seed vertices themselves), else 0.0.
    """
    expanded = seed_mask.clone()
    frontier = seed_mask.clone()
    for _ in range(max(k, 0)):
        frontier = torch.sparse.mm(adj, frontier)
        frontier = (frontier > 0).float()
        expanded = torch.clamp(expanded + frontier, max=1.0)
    return expanded


def weighted_edge_loss(meshes, vertex_weight: torch.Tensor) -> torch.Tensor:
    """
    Same quantity as pytorch3d.loss.mesh_edge_loss (mean squared edge
    length across the batch), but with each edge weighted by the MINIMUM
    of its two endpoints' vertex_weight -- an edge touching even one
    down-weighted vertex is treated as down-weighted, since that's the
    edge doing the "dragging" of a down-weighted vertex's neighbor.

    Not guaranteed numerically identical to pytorch3d's own
    implementation when vertex_weight is all-ones -- this is a
    from-scratch weighted reimplementation for this diagnostic
    experiment, not a drop-in replacement for the production loss.

    Args:
        meshes: pytorch3d Meshes, batch of B specimens.
        vertex_weight: (B, n_verts) weight in [0, 1] per vertex per specimen
            (same n_verts for every specimen -- shared template topology).

    Returns:
        Scalar weighted mean squared edge length.
    """
    verts_packed = meshes.verts_packed()          # (sum_V, 3)
    edges_packed = meshes.edges_packed()          # (sum_E, 2), already packed indices
    edge_to_mesh = meshes.edges_packed_to_mesh_idx()  # (sum_E,) which specimen each edge belongs to
    verts_per_mesh = meshes.num_verts_per_mesh()

    # map packed vertex weight the same way verts are packed
    weight_packed = vertex_weight.reshape(-1)  # valid because every mesh has the same n_verts, packed in specimen order

    v0 = verts_packed[edges_packed[:, 0]]
    v1 = verts_packed[edges_packed[:, 1]]
    edge_len_sq = ((v0 - v1) ** 2).sum(dim=-1)  # (sum_E,)

    w0 = weight_packed[edges_packed[:, 0]]
    w1 = weight_packed[edges_packed[:, 1]]
    edge_weight = torch.minimum(w0, w1)  # (sum_E,)

    denom = edge_weight.sum().clamp(min=1e-8)
    return (edge_len_sq * edge_weight).sum() / denom


def weighted_laplacian_loss(meshes, vertex_weight: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Weighted uniform-Laplacian smoothing loss: for each vertex, its
    Laplacian is (mean of neighbor positions - own position); the loss is
    the weighted mean of the squared Laplacian norm, weighted per-vertex
    by vertex_weight.

    Not guaranteed numerically identical to pytorch3d's
    mesh_laplacian_smoothing(method="uniform") when vertex_weight is
    all-ones -- own reimplementation, same caveat as weighted_edge_loss.

    Args:
        meshes: pytorch3d Meshes, batch of B specimens.
        vertex_weight: (B, n_verts) weight in [0, 1] per vertex per specimen.
        adj: (n_verts, n_verts) sparse adjacency from build_vertex_adjacency
            (shared template topology).

    Returns:
        Scalar weighted mean squared Laplacian norm.
    """
    verts = meshes.verts_padded()  # (B, V, 3)
    B, V, _ = verts.shape

    degree = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)  # (V,)

    lap_list = []
    for b in range(B):
        neighbor_sum = torch.sparse.mm(adj, verts[b])  # (V, 3)
        lap = neighbor_sum / degree.unsqueeze(-1) - verts[b]  # (V, 3)
        lap_list.append(lap)
    lap = torch.stack(lap_list, dim=0)  # (B, V, 3)

    lap_sq = (lap ** 2).sum(dim=-1)  # (B, V)
    denom = vertex_weight.sum().clamp(min=1e-8)
    return (lap_sq * vertex_weight).sum() / denom


def random_touched_mask(n_touched: int, n_verts: int, device, generator=None) -> torch.Tensor:
    """
    (n_verts,) boolean mask with exactly n_touched True entries, chosen
    uniformly at random -- the spatially-SCATTERED counterpart to
    khop_expand_mask's spatially-CLUSTERED mask. Used to build the "global"
    control's per-iteration touched set at the same size "local" mode's
    actual k-hop coverage used that iteration, so the two arms are matched
    on count and per-vertex weight, differing only in WHERE the down-
    weighted vertices are located.

    Args:
        n_touched: exact number of vertices to mark True (clamped to
            [0, n_verts]).
        n_verts: total vertex count.
        device: torch device.
        generator: optional torch.Generator for reproducible draws.

    Returns:
        (n_verts,) bool tensor.
    """
    n_touched = min(max(int(n_touched), 0), n_verts)
    mask = torch.zeros(n_verts, dtype=torch.bool, device=device)
    if n_touched == 0:
        return mask
    idx = torch.randperm(n_verts, device=device, generator=generator)[:n_touched]
    mask[idx] = True
    return mask


def mesh_integrity_metrics(source_verts: np.ndarray, deformed_verts: np.ndarray, faces: np.ndarray) -> dict:
    """
    Two-tier sanity check that a geometry-relaxing change (like locally
    down-weighting edge/laplacian) isn't quietly damaging mesh quality
    even as penetration numbers improve.

    edge_logratio: for every edge, log(deformed_length / source_length).
        0 = no change. Reported as mean absolute value (typical distortion)
        and p95 (worst-case distortion) per specimen. source_length is
        measured on the SMAL template rest pose (v_template), the
        known-good reference every stage's edge/laplacian regularizer is
        implicitly trying to stay close to.
    folded_face_frac: fraction of faces whose normal has flipped by more
        than 90 degrees relative to the same face's normal in the source
        (rest-pose) mesh -- a direct signature of local self-folding, not
        just stretching.

    Args:
        source_verts: (V, 3) rest-pose (template) vertex positions.
        deformed_verts: (V, 3) fitted vertex positions for one specimen.
        faces: (F, 3) face indices (shared template topology).

    Returns:
        dict with edge_logratio_mean_abs, edge_logratio_p95, folded_face_frac.
    """
    edges = set()
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        edges.add((a, b)); edges.add((b, c)); edges.add((c, a))
    edges = np.array(list(edges))

    src_len = np.linalg.norm(source_verts[edges[:, 0]] - source_verts[edges[:, 1]], axis=1)
    dst_len = np.linalg.norm(deformed_verts[edges[:, 0]] - deformed_verts[edges[:, 1]], axis=1)
    eps = 1e-8
    logratio = np.log((dst_len + eps) / (src_len + eps))

    def face_normals(verts):
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.clip(norm, eps, None)

    n_src = face_normals(source_verts)
    n_dst = face_normals(deformed_verts)
    dot = (n_src * n_dst).sum(axis=1)
    folded_frac = float((dot < 0).mean())

    return {
        "edge_logratio_mean_abs": float(np.abs(logratio).mean()),
        "edge_logratio_p95": float(np.percentile(np.abs(logratio), 95)),
        "folded_face_frac": folded_frac,
    }
