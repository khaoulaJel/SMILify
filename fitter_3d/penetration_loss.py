"""
penetration_loss.py

THE IDEA
--------
A fitted mesh can look accurate from the outside (good Chamfer distance,
good silhouette overlap) while still being anatomically impossible on the
inside -- a leg passing straight through the gaster, an antenna passing
through the head. Surface-matching metrics have no way to see this, since
they only ever look at the nearest point on the target surface, never at
whether the model has folded through itself.

This module adds a differentiable penalty for exactly that failure mode.
For every pair of body parts that should never touch (e.g. "legs" and
"gaster"), we check every vertex of one part against the *triangulated
surface* of the other part -- not just its vertices, since two vertices
can sit in the gaps of each other's mesh while the actual triangle between
them is intersected. For each query vertex we:

  1. find the nearest candidate triangle on the other part (by distance
     to that triangle's centroid -- cheap, and a good enough proxy for
     "which triangle is relevant" even though it isn't the exact closest
     point),
  2. measure how far the vertex is from that triangle's plane, using the
     triangle's own flat face normal to get a signed distance (negative =
     inside the other part, i.e. penetrating),
  3. gate this by two per-specimen thresholds, both expressed as a
     fraction of that specimen's own bounding-box diagonal so the same
     config works across ants of different sizes:
       - proximity_tau_fraction: only vertices already close to the other
         part are considered at all (far-away vertices contribute zero,
         not a large meaningless "distance"),
       - max_depth_fraction: any single vertex's contribution is clamped,
         so one badly-initialized vertex early in optimisation can't
         produce a gradient large enough to fight the rest of the fit and
         drag the whole mesh out of shape.

The result is averaged into one scalar per specimen, which can be added
to the training objective with a weight, or just logged as a diagnostic
with weight zero.

A short linear ramp (`n_ramp_iters`) fades the penalty in gradually at
the start of a stage, for the same reason the depth clamp exists: an
untrained pose can be badly self-intersecting purely from initialisation,
and hitting it with the full penalty from iteration 0 would fight the
optimiser rather than help it.


"""

import numpy as np
import torch


def _ramp_factor(iteration: int, n_ramp_iters: int) -> float:
    """Linear ramp from 0 to 1 over the first n_ramp_iters iterations of
    a stage, then constant at 1. n_ramp_iters=0 disables ramping
    (full strength from the first iteration)."""
    if n_ramp_iters <= 0:
        return 1.0
    return min(1.0, iteration / n_ramp_iters)


def _build_part_faces(faces_np: np.ndarray, part_vertex_indices: dict) -> dict:
    """
    Precompute, once, the subset of template faces fully contained within
    each anatomical part -- i.e. that part's own surface as triangles,
    not just its vertex set. Template topology is fixed across specimens
    and iterations, so this only needs to run once per process, not once
    per call.

    Args:
        faces_np: (F, 3) template face indices.
        part_vertex_indices: dict part_name -> array of vertex indices,
            from part_groups.get_part_vertex_indices().

    Returns:
        dict part_name -> (Nf_part, 3) array of faces (by original,
        un-remapped global vertex index) whose three vertices all belong
        to that part.
    """
    part_faces = {}
    for part_name, v_idx in part_vertex_indices.items():
        v_set = set(v_idx.tolist())
        mask = np.array([all(v in v_set for v in f) for f in faces_np])
        part_faces[part_name] = faces_np[mask]
    return part_faces


def _directional_penalty(
    verts_padded: torch.Tensor,
    idx_query: torch.Tensor,
    faces_surface: torch.Tensor,
    tau: torch.Tensor,
    max_depth: torch.Tensor,
):
    """
    Mean clamped penetration depth of one part's vertices (idx_query)
    against another part's triangulated surface (faces_surface), in one
    direction only -- the caller runs this twice per pair, swapping which
    part is "query" and which is "surface", since penetration is not
    symmetric (A's vertices can be inside B without any of B's vertices
    being inside A).

    Matching strategy: each query vertex is matched to whichever surface
    triangle has the nearest CENTROID (cheap: one distance computation,
    no per-triangle projection), and that same centroid is then used
    directly as the surface position for the distance/sign test, together
    with that triangle's own flat, unaveraged face normal. This is an
    approximation -- the true closest point on a triangle can differ from
    its centroid, particularly near an edge or corner of a long thin
    triangle -- but it is fast and, at the gentle loss weights this
    penalty is actually trained with, has proven to give a good net
    improvement in penetration without disturbing the rest of the fit.

    Returns:
        depth_mean: (B,) mean gated depth over this part's query vertices.
        diagnostics: dict of raw (unramped) per-specimen accumulators,
            described in penetration_loss_batched's docstring.
    """
    B = verts_padded.shape[0]
    device = verts_padded.device
    Nq = idx_query.shape[0]
    Nf = faces_surface.shape[0]

    if Nq == 0 or Nf == 0:
        zeros = torch.zeros(B, device=device)
        empty_mask = torch.zeros(B, Nq, dtype=torch.bool, device=device)
        diagnostics = {
            "num_penetrating": zeros.clone(),
            "sum_depth": zeros.clone(),
            "max_depth": zeros.clone(),
            "n_query": 0,
            "penetrating_mask": empty_mask,
        }
        return zeros, diagnostics

    query_pts = verts_padded[:, idx_query, :]  # (B, Nq, 3)

    # This iteration's positions for the surface part's own triangles.
    tri_verts = verts_padded[:, faces_surface, :]  # (B, Nf, 3, 3)
    va, vb, vc = tri_verts[:, :, 0, :], tri_verts[:, :, 1, :], tri_verts[:, :, 2, :]
    face_normals = torch.nn.functional.normalize(
        torch.cross(vb - va, vc - va, dim=-1), dim=-1, eps=1e-8
    )  # (B, Nf, 3)
    face_centroids = (va + vb + vc) / 3.0  # (B, Nf, 3)

    # Nearest candidate triangle by centroid distance.
    dists_to_centroids = torch.cdist(query_pts, face_centroids)  # (B, Nq, Nf)
    dist, nearest_idx = dists_to_centroids.min(dim=2)  # (B, Nq)

    nearest_idx_expanded = nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
    matched_centroid = torch.gather(face_centroids, 1, nearest_idx_expanded)
    matched_normal = torch.gather(face_normals, 1, nearest_idx_expanded)

    # Signed distance to the matched triangle's plane: negative means the
    # query vertex sits on the inward side of the surface, i.e. penetrating.
    offset = query_pts - matched_centroid
    sign = (offset * matched_normal).sum(dim=-1)  # (B, Nq)

    depth = torch.where(sign < 0, dist, torch.zeros_like(dist))
    depth = torch.clamp(depth, max=max_depth.unsqueeze(1))  # per-specimen clamp

    # Only vertices already close to the other part count at all -- this
    # keeps distant, clearly-non-colliding vertices from ever contributing
    # a nonzero (if tiny) value, and keeps the gate itself a meaningful
    # proximity check rather than a global distance term in disguise.
    is_close = dist < tau.unsqueeze(1)
    gated_depth = depth * is_close.float()  # (B, Nq); nonzero exactly where penetrating

    penetrating_mask = gated_depth > 0  # (B, Nq)
    diagnostics = {
        "num_penetrating": penetrating_mask.float().sum(dim=1),
        "sum_depth": gated_depth.sum(dim=1),  # undiluted total, for mean-among-penetrating
        "max_depth": gated_depth.max(dim=1).values,
        "n_query": Nq,
        "penetrating_mask": penetrating_mask,
    }
    return gated_depth.mean(dim=1), diagnostics


def _pair_direction_summary(diag: dict) -> dict:
    """Reshape one direction's raw accumulators into the same vocabulary
    the aggregate diagnostics use, so a per-pair row and the aggregate row
    are read the same way. The denominator is clamped to >= 1 so a pair
    with zero collisions reports 0.0, not NaN."""
    n_pen = diag["num_penetrating"]
    return {
        "num_penetrating": n_pen,
        "max_depth": diag["max_depth"],
        "mean_depth_among_penetrating": diag["sum_depth"] / torch.clamp(n_pen, min=1),
        "n_query": diag["n_query"],
    }


def penetration_loss_batched(
    verts_padded: torch.Tensor,
    part_vertex_indices: dict,
    part_faces: dict,
    non_adjacent_pairs: list,
    proximity_tau_fraction: float = 0.03,
    max_depth_fraction: float = 0.08,
    iteration: int = 0,
    n_ramp_iters: int = 200,
    return_diagnostics: bool = False,
    return_per_pair: bool = False,
):
    """
    Differentiable inter-part penetration penalty, batched across
    specimens. See the module docstring for the idea; this docstring
    covers the call contract.

    Args:
        verts_padded: (B, V, 3) current fitted vertices, WITH gradient.
        part_vertex_indices: dict part_name -> vertex indices, from
            part_groups.get_part_vertex_indices() -- computed once
            outside the training loop.
        part_faces: dict part_name -> face index arrays, from
            _build_part_faces() -- computed once outside the training
            loop, alongside part_vertex_indices.
        non_adjacent_pairs: list of (part_a, part_b) tuples, from
            part_groups.get_non_adjacent_pairs() -- anatomically adjacent
            parts (e.g. a leg and the thorax it attaches to) are expected
            to touch and are never checked.
        proximity_tau_fraction: proximity gate, as a fraction of each
            specimen's own bounding-box diagonal.
        max_depth_fraction: penetration depth is clamped to this fraction
            of the specimen's bounding-box diagonal before contributing
            to the loss, so one badly-initialized vertex cannot produce a
            gradient that dominates and distorts otherwise-correct
            geometry.
        iteration: current iteration within the stage (0-indexed), used
            only for the ramp -- pass the same value already threaded
            through Stage.step()/Stage.forward().
        n_ramp_iters: number of iterations over which the penalty ramps
            linearly from 0 to full strength. 0 disables ramping.
        return_diagnostics: if True, also return a dict of unramped,
            unweighted collision diagnostics (below). Default False
            preserves the plain-tensor return used by existing callers.
        return_per_pair: if True (requires return_diagnostics), the
            diagnostics dict additionally carries a "per_pair" entry
            breaking the same quantities down by individual
            (part_a, part_b) pair and direction. No extra GPU work -- these
            values are already computed per pair below and merely summed
            away otherwise -- so this is opt-in only because of the extra
            tensors it adds to the returned dict.

    Returns:
        return_diagnostics=False (default):
            (B,) tensor -- per-specimen mean clamped, ramped penetration
            depth across all checked pairs and both directions.

        return_diagnostics=True:
            ((B,) tensor, dict) -- the same tensor, plus a dict of
            unramped quantities describing the collision itself:
              "mean_depth_unramped": the loss's own mean-depth quantity
                  with the ramp factored out -- use this, not the primary
                  return value, when comparing runs at different weights
                  or ramp schedules.
              "num_penetrating": count of (vertex, direction, pair)
                  instances currently penetrating.
              "fraction_penetrating": num_penetrating divided by the total
                  number of checks performed.
              "max_depth": worst single-vertex clamped, gated depth.
              "mean_depth_among_penetrating": mean depth over penetrating
                  instances only -- NOT diluted by the (usually large)
                  majority of non-penetrating vertices, unlike the primary
                  loss value. Use this to judge collision severity.
              "penetrating_vertex_mask": (B, V) bool, True at a vertex
                  currently penetrating something -- for visualisation,
                  not used in the loss itself.
              "bbox_diag", "tau", "max_depth_gate": the per-specimen scale
                  and derived gate values every pair's thresholds are
                  computed from, logged so a shared-normalisation effect
                  can be checked directly.
              "per_pair" (only if return_per_pair=True): dict
                  "{part_a}__{part_b}" -> {"A_into_B": {...},
                  "B_into_A": {...}}. Pairs skipped because a part is
                  empty in the template (e.g. "waist", which carries zero
                  skinning weight in SMIL_OmniAnt.pkl) are absent, exactly
                  as they are absent from the aggregate sums.
    """
    device = verts_padded.device
    B, V, _ = verts_padded.shape

    bbox_min = verts_padded.min(dim=1).values
    bbox_max = verts_padded.max(dim=1).values
    bbox_diag = (bbox_max - bbox_min).norm(dim=1)  # (B,) -- per-specimen scale
    tau = proximity_tau_fraction * bbox_diag
    max_depth = max_depth_fraction * bbox_diag
    ramp = _ramp_factor(iteration, n_ramp_iters)

    per_specimen_penalty = torch.zeros(B, device=device)
    n_pairs_used = 0

    total_num_penetrating = torch.zeros(B, device=device)
    total_sum_depth = torch.zeros(B, device=device)
    total_max_depth = torch.zeros(B, device=device)
    total_n_query = 0
    penetrating_vertex_mask = (
        torch.zeros(B, V, dtype=torch.bool, device=device) if return_diagnostics else None
    )
    per_pair_out = {} if (return_diagnostics and return_per_pair) else None

    for part_a, part_b in non_adjacent_pairs:
        idx_a = part_vertex_indices.get(part_a)
        idx_b = part_vertex_indices.get(part_b)
        faces_a = part_faces.get(part_a)
        faces_b = part_faces.get(part_b)
        if idx_a is None or idx_b is None or faces_a is None or faces_b is None:
            continue
        if len(idx_a) == 0 or len(idx_b) == 0 or len(faces_a) == 0 or len(faces_b) == 0:
            continue  # e.g. "waist" -- present in part_vertex_indices but carries no faces/vertices in this template

        idx_a_t = torch.as_tensor(idx_a, device=device, dtype=torch.long)
        idx_b_t = torch.as_tensor(idx_b, device=device, dtype=torch.long)
        faces_a_t = torch.as_tensor(faces_a, device=device, dtype=torch.long)
        faces_b_t = torch.as_tensor(faces_b, device=device, dtype=torch.long)

        # Both directions: A's vertices into B's surface, and B's vertices
        # into A's surface -- penetration is not symmetric.
        penalty_a_into_b, diag_a_into_b = _directional_penalty(
            verts_padded, idx_a_t, faces_b_t, tau, max_depth
        )
        penalty_b_into_a, diag_b_into_a = _directional_penalty(
            verts_padded, idx_b_t, faces_a_t, tau, max_depth
        )

        per_specimen_penalty = per_specimen_penalty + penalty_a_into_b + penalty_b_into_a
        n_pairs_used += 1

        if return_diagnostics:
            total_num_penetrating += diag_a_into_b["num_penetrating"] + diag_b_into_a["num_penetrating"]
            total_sum_depth += diag_a_into_b["sum_depth"] + diag_b_into_a["sum_depth"]
            total_max_depth = torch.maximum(total_max_depth, diag_a_into_b["max_depth"])
            total_max_depth = torch.maximum(total_max_depth, diag_b_into_a["max_depth"])
            total_n_query += diag_a_into_b["n_query"] + diag_b_into_a["n_query"]

            if diag_a_into_b["penetrating_mask"].shape[1] > 0:
                penetrating_vertex_mask[:, idx_a_t] |= diag_a_into_b["penetrating_mask"]
            if diag_b_into_a["penetrating_mask"].shape[1] > 0:
                penetrating_vertex_mask[:, idx_b_t] |= diag_b_into_a["penetrating_mask"]

            if per_pair_out is not None:
                per_pair_out[f"{part_a}__{part_b}"] = {
                    "A_into_B": _pair_direction_summary(diag_a_into_b),
                    "B_into_A": _pair_direction_summary(diag_b_into_a),
                }

    if n_pairs_used > 0:
        per_specimen_penalty = per_specimen_penalty / (2 * n_pairs_used)  # 2x: both directions per pair

    # The ramp is a training-schedule artifact, not a property of the
    # geometry -- keep the pre-ramp value so every diagnostic below stays
    # comparable across runs with different ramp schedules or weights.
    unramped_penalty = per_specimen_penalty.detach()
    per_specimen_penalty = per_specimen_penalty * ramp

    if not return_diagnostics:
        return per_specimen_penalty

    diagnostics_out = {
        "mean_depth_unramped": unramped_penalty,
        "num_penetrating": total_num_penetrating,
        "fraction_penetrating": total_num_penetrating / max(total_n_query, 1),
        "max_depth": total_max_depth,
        "mean_depth_among_penetrating": total_sum_depth / torch.clamp(total_num_penetrating, min=1),
        "penetrating_vertex_mask": penetrating_vertex_mask,
        "bbox_diag": bbox_diag.detach(),
        "tau": tau.detach(),
        "max_depth_gate": max_depth.detach(),
    }
    if per_pair_out is not None:
        diagnostics_out["per_pair"] = per_pair_out
    return per_specimen_penalty, diagnostics_out