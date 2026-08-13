"""
gwn_penetration_loss.py

Differentiable generalized-winding-number (GWN) penetration penalty -- a NEW,
additive alternative to penetration_loss.py's nearest-triangle-centroid
proximity test, motivated by diagnostics/gwn_diagnostic_full.py's confirmed
finding: the proximity test's inside/outside verdict disagrees with the true
GWN signal specifically (and significantly) more often in whichever
direction of a pair fails to improve under training.

This module does NOT replace penetration_loss.py -- both are available, and
Stage's optional w_penetration_gwn (see trainer.py) is purely additive,
default 0, zero effect on any existing config.

Design, matching the already-validated CPU diagnostic (gwn_diagnostic.py)
as closely as possible so results are comparable, but reimplemented in
torch for differentiability:
  - Jacobson et al. 2013 solid-angle formula, batched over specimens.
  - Each anatomical part is CAPPED (fan triangulation from each boundary
    loop's centroid) to make it a closed surface before GWN is meaningful,
    same as the diagnostic. Cap FACE TOPOLOGY (which vertices, in which
    order) is precomputed ONCE from the rest-pose template and fixed for
    all training -- only the cap CENTROID POSITIONS are recomputed every
    iteration (a differentiable mean of the current loop vertex positions).
    Re-deriving orientation from live geometry every iteration was
    deliberately avoided: it would be both unnecessary (topology is fixed)
    and a source of gradient discontinuities if a cap ever flipped.
  - No manifoldness gate needed here (unlike the diagnostic, which checked
    it as a precondition on already-fixed data): confirmed structurally in
    Step 6 that SMIL's fixed topology can't produce non-manifold geometry
    under any deformation, so it's a closed question for this model, not
    something to recheck every training run.
  - Penalty per query vertex: relu(w - 0.5) -- zero outside/on the surface,
    positive and increasing with how deep inside the winding number says
    the vertex is. The GWN analog of "depth" in the proximity loss, but
    from a globally-correct signal rather than a local nearest-point
    approximation. No proximity gate (tau) is needed: GWN is exact
    everywhere for a genuinely closed surface, not just near it, unlike
    the nearest-point test it replaces.
"""

import math
from collections import defaultdict

import numpy as np
import torch


def torch_winding_number(q: torch.Tensor, V: torch.Tensor, F: torch.Tensor, chunk: int = 1000) -> torch.Tensor:
    """
    Differentiable generalized winding number (Jacobson et al. 2013).

    Args:
        q: (B, Nq, 3) query points.
        V: (B, Nv, 3) closed mesh vertices, per specimen (gradient flows here).
        F: (Nf, 3) face indices, shared topology across the batch.
        chunk: faces processed per step (memory/speed tradeoff).

    Returns:
        (B, Nq) winding number. |w| > 0.5 <=> inside, for a closed,
        consistently-oriented mesh.
    """
    B, Nq, _ = q.shape
    Nf = F.shape[0]
    out = torch.zeros(B, Nq, device=q.device, dtype=q.dtype)
    tri = V[:, F]  # (B, Nf, 3, 3)
    for s in range(0, Nf, chunk):
        t = tri[:, s : s + chunk]  # (B, c, 3, 3)
        a = t[:, None, :, 0, :] - q[:, :, None, :]  # (B, Nq, c, 3)
        b = t[:, None, :, 1, :] - q[:, :, None, :]
        c = t[:, None, :, 2, :] - q[:, :, None, :]
        la = a.norm(dim=-1)
        lb = b.norm(dim=-1)
        lc = c.norm(dim=-1)
        num = (a * torch.cross(b, c, dim=-1)).sum(-1)
        den = (
            la * lb * lc
            + (a * b).sum(-1) * lc
            + (a * c).sum(-1) * lb
            + (b * c).sum(-1) * la
        )
        out = out + (2.0 * torch.atan2(num, den)).sum(dim=2)
    return out / (4.0 * math.pi)


def _find_boundary_loops(faces_subset: np.ndarray):
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces_subset):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (a, b) if a < b else (b, a)
            edge_faces[key].append(fi)
    boundary_edges = [k for k, v in edge_faces.items() if len(v) == 1]

    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    loops = []
    for a, b in boundary_edges:
        e0 = frozenset((a, b))
        if e0 in visited:
            continue
        loop = [a, b]
        visited.add(e0)
        prev, curr = a, b
        while True:
            nxt = None
            for n in adj[curr]:
                if n == prev:
                    continue
                ee = frozenset((curr, n))
                if ee not in visited:
                    nxt = n
                    break
            if nxt is None:
                break
            visited.add(frozenset((curr, nxt)))
            loop.append(nxt)
            prev, curr = curr, nxt
            if curr == a:
                break
        if loop[0] == loop[-1]:
            loop = loop[:-1]
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def precompute_capped_topology(rest_pose_verts: np.ndarray, part_faces: dict) -> dict:
    """
    Once, from the REST-POSE template (fixed topology -> valid for every
    specimen and every iteration of training): for each part, its boundary
    loops and a FIXED, correctly-oriented cap face list (referencing loop
    vertex indices plus a placeholder index for each loop's centroid,
    resolved to real tensor indices at forward time in build_capped_mesh).

    Returns:
        dict part_name -> {
            "orig_faces": (Nf, 3) int64 array, this part's own faces,
            "loops": list of vertex-index lists (one per boundary loop),
            "cap_faces_local": list of (loop_idx, a, b) triples per loop,
                already oriented outward -- "a, b" are positions WITHIN
                the loop (indices into loops[loop_idx]), not raw vertex ids,
                so build_capped_mesh only has to look up real vertex ids.
        }
    """
    result = {}
    for part_name, faces_subset in part_faces.items():
        if len(faces_subset) == 0:
            result[part_name] = {"orig_faces": faces_subset, "loops": [], "cap_faces_local": []}
            continue

        part_vert_ids = np.unique(faces_subset)
        part_centroid = rest_pose_verts[part_vert_ids].mean(axis=0)
        loops = _find_boundary_loops(faces_subset)

        cap_faces_local = []
        for li, loop in enumerate(loops):
            pts = rest_pose_verts[loop]
            loop_centroid = pts.mean(axis=0)
            outward_ref = loop_centroid - part_centroid
            for i in range(len(loop)):
                a_pos, b_pos = i, (i + 1) % len(loop)
                a_id, b_id = loop[a_pos], loop[b_pos]
                tri_normal = np.cross(rest_pose_verts[b_id] - rest_pose_verts[a_id], loop_centroid - rest_pose_verts[a_id])
                if np.dot(tri_normal, outward_ref) < 0:
                    cap_faces_local.append((li, b_pos, a_pos))
                else:
                    cap_faces_local.append((li, a_pos, b_pos))

        result[part_name] = {"orig_faces": faces_subset, "loops": loops, "cap_faces_local": cap_faces_local}
    return result


def build_capped_mesh(verts_padded: torch.Tensor, part_name: str, capped_topology: dict):
    """
    Per-iteration: appends one differentiable centroid vertex per boundary
    loop (mean of that loop's current vertex positions) and assembles the
    fixed, precomputed cap faces referencing them -- the closed "true
    surface" for this part, this iteration.

    Args:
        verts_padded: (B, V, 3) current fitted vertices (gradient source).
        part_name: which part to close.
        capped_topology: from precompute_capped_topology().

    Returns:
        capped_V: (B, V + n_loops, 3)
        capped_F: (Nf_total, 3) int64, shared across the batch.
    """
    info = capped_topology[part_name]
    B, V, _ = verts_padded.shape
    device = verts_padded.device

    all_faces = [tuple(f) for f in info["orig_faces"]]
    if not info["loops"]:
        capped_F = torch.tensor(all_faces, dtype=torch.long, device=device)
        return verts_padded, capped_F

    centroids = []
    next_idx = V
    loop_start_idx = {}
    for li, loop in enumerate(info["loops"]):
        loop_t = torch.as_tensor(loop, dtype=torch.long, device=device)
        centroid = verts_padded[:, loop_t, :].mean(dim=1, keepdim=True)  # (B, 1, 3)
        centroids.append(centroid)
        loop_start_idx[li] = next_idx
        next_idx += 1

    for li, a_pos, b_pos in info["cap_faces_local"]:
        loop = info["loops"][li]
        a_id, b_id = loop[a_pos], loop[b_pos]
        cidx = loop_start_idx[li]
        all_faces.append((a_id, b_id, cidx))

    capped_V = torch.cat([verts_padded] + centroids, dim=1)  # (B, V + n_loops, 3)
    capped_F = torch.tensor(all_faces, dtype=torch.long, device=device)
    return capped_V, capped_F


def _ramp_factor(iteration: int, n_ramp_iters: int) -> float:
    """Same convention as penetration_loss.py's _ramp_factor: linear 0->1 over
    the first n_ramp_iters iterations of a stage, then constant at 1.
    n_ramp_iters<=0 disables ramping (full strength from iteration 0)."""
    if n_ramp_iters <= 0:
        return 1.0
    return min(1.0, iteration / n_ramp_iters)


def gwn_directional_penalty(verts_padded: torch.Tensor, idx_query: torch.Tensor, capped_V: torch.Tensor,
                              capped_F: torch.Tensor):
    """
    Mean relu(w - 0.5) over one part's query vertices against another
    part's capped (closed) surface -- the GWN analog of
    penetration_loss.py's _directional_penalty. Caller runs this twice per
    pair, swapping which part is query vs. surface (penetration isn't
    symmetric).

    Returns:
        depth_mean: (B,) mean penalty.
        diagnostics: {"inside_mask": (B, Nq) bool, "w": (B, Nq) float}.
    """
    query_pts = verts_padded[:, idx_query, :]  # (B, Nq, 3)
    w = torch_winding_number(query_pts, capped_V, capped_F)
    depth = torch.clamp(w - 0.5, min=0.0)
    inside_mask = w > 0.5
    return depth.mean(dim=1), {"inside_mask": inside_mask, "w": w.detach()}


def gwn_penetration_loss_batched(verts_padded: torch.Tensor, part_vertex_indices: dict, capped_topology: dict,
                                   pairs: list, iteration: int = 0, n_ramp_iters: int = 0,
                                   return_diagnostics: bool = False):
    """
    GWN-based penetration penalty, batched across specimens, scoped to an
    explicit list of (part_a, part_b) pairs (deliberately NOT defaulting to
    every non-adjacent pair -- this is a targeted first test on the one
    pair (gaster, legs) Step 6 validated, not a full replacement rollout).

    Args:
        iteration, n_ramp_iters: same convention and purpose as
            penetration_loss.py's penetration_loss_batched -- a linear 0->1
            ramp over the first n_ramp_iters iterations of a stage, so an
            untrained/undertrained pose can't get hit with full-strength
            penalty from iteration 0. n_ramp_iters=0 (default) disables
            ramping. Diagnostics (num_inside/fraction_inside) are always
            UNRAMPED -- they describe the actual geometry, not the
            training-schedule-scaled loss value.

    Returns:
        return_diagnostics=False: (B,) tensor, per-specimen mean penalty
            (ramped).
        return_diagnostics=True: ((B,) tensor, dict) with "num_inside" and
            "fraction_inside" (B,) -- the GWN analog of penetration_loss.py's
            num_penetrating/fraction_penetrating, for direct comparison.
    """
    device = verts_padded.device
    B = verts_padded.shape[0]

    per_specimen_penalty = torch.zeros(B, device=device)
    n_pairs_used = 0
    total_inside = torch.zeros(B, device=device)
    total_n_query = 0

    capped_cache = {}

    def get_capped(part_name):
        if part_name not in capped_cache:
            capped_cache[part_name] = build_capped_mesh(verts_padded, part_name, capped_topology)
        return capped_cache[part_name]

    for part_a, part_b in pairs:
        idx_a = part_vertex_indices.get(part_a)
        idx_b = part_vertex_indices.get(part_b)
        if idx_a is None or idx_b is None or len(idx_a) == 0 or len(idx_b) == 0:
            continue
        if len(capped_topology[part_a]["orig_faces"]) == 0 or len(capped_topology[part_b]["orig_faces"]) == 0:
            continue

        idx_a_t = torch.as_tensor(idx_a, dtype=torch.long, device=device)
        idx_b_t = torch.as_tensor(idx_b, dtype=torch.long, device=device)

        capped_V_b, capped_F_b = get_capped(part_b)
        capped_V_a, capped_F_a = get_capped(part_a)

        penalty_a_into_b, diag_ab = gwn_directional_penalty(verts_padded, idx_a_t, capped_V_b, capped_F_b)
        penalty_b_into_a, diag_ba = gwn_directional_penalty(verts_padded, idx_b_t, capped_V_a, capped_F_a)

        per_specimen_penalty = per_specimen_penalty + penalty_a_into_b + penalty_b_into_a
        n_pairs_used += 1

        if return_diagnostics:
            total_inside = total_inside + diag_ab["inside_mask"].float().sum(1) + diag_ba["inside_mask"].float().sum(1)
            total_n_query += len(idx_a) + len(idx_b)

    if n_pairs_used > 0:
        per_specimen_penalty = per_specimen_penalty / (2 * n_pairs_used)

    ramp = _ramp_factor(iteration, n_ramp_iters)
    per_specimen_penalty = per_specimen_penalty * ramp

    if not return_diagnostics:
        return per_specimen_penalty

    diagnostics = {
        "num_inside": total_inside,
        "fraction_inside": total_inside / max(total_n_query, 1),
    }
    return per_specimen_penalty, diagnostics
