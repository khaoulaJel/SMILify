"""
bvh_penetration_loss.py

Differentiable BVH-based penetration penalty -- the field-standard approach (SMPLify-X,
Pavlakos et al. 2019): exact triangle-triangle collision detection via a BVH tree
(`mesh_intersection.bvh_search_tree.BVH`), then a conical signed-distance-field push
resolving each detected collision (`mesh_intersection.loss.DistanceFieldPenetrationLoss`),
both reused directly from the compiled `bvh_cuda` extension rather than reimplemented here.

This was the "original spec's TASK 4", named untried in every prior deliverable in this
directory because the reference implementation was believed non-functional on this project's
environment. That belief was correct for one fork (`EthanFifle/torch-mesh-intersection`, kernel
compiles but never detects any collision, confirmed via a coincident-triangle test) and wrong
for another (`wonjongg/torch-mesh-isect`, the fork behind SIGGRAPH 2025's "Instant
Self-Intersection Repair" paper, confirmed correctly detecting real non-degenerate
intersections including multiple simultaneous ones -- see
`diagnostics/khaoula_review/DELIVERABLE_penetration_fix_TASK1-5.md`'s 2026-08-24 update). This
module is the first actual use of that confirmed-working primitive as a training signal, not
another compat probe.

Design, matching this project's existing part-pair scoping so results are comparable to every
prior arm:
  - The BVH searches ALL 20454 template faces at once (it has no concept of "parts"), so
    collision pairs are filtered AFTER detection down to whichever (part_a, part_b) pairs
    `train_pairs` allows -- same purpose as `penetration_train_pairs` in
    `penetration_loss.py`/`trainer.py`, reimplemented here rather than reused because BVH's
    collision_idxs are FACE pairs, not per-vertex query/surface splits, a different shape of
    problem.
  - A face belongs to a part only if all 3 of its vertices do (matching `_build_part_faces`'s
    convention) -- 518/20454 template faces (2.5%) are seam faces spanning >1 part and are
    never eligible for any trained pair, same coverage tradeoff `part_faces` already accepts
    elsewhere in this codebase.
  - Same-part collisions (a part folding into itself) are always filtered out, matching every
    other penetration signal in this project, which only ever checks NON-ADJACENT PART pairs.
  - The BVH search itself (`BVH.forward`) is `@torch.no_grad()` upstream (see
    `bvh_search_tree.py`) -- gradients flow only through `DistanceFieldPenetrationLoss`, which
    recomputes the conical distance field from the SAME live (non-detached) triangle positions
    at the indices the (detached) search returned. This is the library's own established
    design, not something introduced here.
"""

import numpy as np
import torch

from mesh_intersection.bvh_search_tree import BVH
from mesh_intersection.loss import DistanceFieldPenetrationLoss


def build_face_part_ids(faces: np.ndarray, part_vertex_indices: dict) -> np.ndarray:
    """
    Once, from the template: (F,) int64 array, part index per face (order matching
    part_vertex_indices' iteration order) or -1 for a seam face whose 3 vertices don't all
    belong to the same part.
    """
    n_verts = faces.max() + 1
    vert_to_part = -np.ones(n_verts, dtype=np.int64)
    part_names = list(part_vertex_indices.keys())
    for part_idx, name in enumerate(part_names):
        idx = part_vertex_indices[name]
        if len(idx) > 0:
            vert_to_part[idx] = part_idx

    face_vert_parts = vert_to_part[faces]  # (F, 3)
    same_part = (face_vert_parts[:, 0] == face_vert_parts[:, 1]) & (
        face_vert_parts[:, 1] == face_vert_parts[:, 2]
    )
    face_part_id = np.where(same_part, face_vert_parts[:, 0], -1)
    part_name_to_idx = {name: idx for idx, name in enumerate(part_names)}
    return face_part_id, part_name_to_idx


def _ramp_factor(iteration: int, n_ramp_iters: int) -> float:
    """Same convention as penetration_loss.py's _ramp_factor."""
    if n_ramp_iters <= 0:
        return 1.0
    return min(1.0, iteration / n_ramp_iters)


def bvh_penetration_loss_batched(
    verts_padded: torch.Tensor,
    faces: torch.Tensor,
    face_part_id: torch.Tensor,
    part_name_to_idx: dict,
    train_pairs: list,
    dfp_loss_module: DistanceFieldPenetrationLoss,
    bvh_module: BVH,
    iteration: int = 0,
    n_ramp_iters: int = 0,
    return_diagnostics: bool = False,
):
    """
    Args:
        verts_padded: (B, V, 3) current fitted vertices, WITH gradient.
        faces: (F, 3) long, shared template topology.
        face_part_id: (F,) long, from build_face_part_ids -- -1 for seam faces.
        part_name_to_idx: part name -> index into face_part_id's part axis, from
            build_face_part_ids's second return value.
        train_pairs: list of (part_a, part_b) name tuples -- ONLY collisions between
            these part pairs contribute to the loss (both orders allowed); everything
            else (including same-part and any pair not listed) is filtered out before
            DistanceFieldPenetrationLoss ever sees it. Deliberately explicit, not
            "every non-adjacent pair", to allow the same pair-scoped comparison TASK4/5/7
            already use.
        dfp_loss_module, bvh_module: pre-constructed once outside the training loop
            (BVH holds no learnable state but reconstructing it and the loss module
            every iteration is unnecessary overhead).

    Returns:
        return_diagnostics=False: (B,) tensor, per-specimen ramped penalty.
        return_diagnostics=True: ((B,) tensor, dict) with "num_colliding_face_pairs" (B,) --
            NOTE: a face-pair count, not a per-vertex count -- not directly comparable in
            units to penetration_loss.py's num_penetrating/soft_num_penetrating, logged
            alongside them for its own trend, not as a drop-in replacement.
    """
    device = verts_padded.device
    B = verts_padded.shape[0]

    triangles = verts_padded[:, faces]  # (B, F, 3, 3)

    with torch.no_grad():
        collision_idxs = bvh_module(triangles)  # (B, max_collisions*F, 2), -1-padded

    recv = collision_idxs[:, :, 0]
    intr = collision_idxs[:, :, 1]
    valid = recv.ge(0) & intr.ge(0)

    recv_part = torch.where(valid, face_part_id[recv.clamp(min=0)], torch.full_like(recv, -1))
    intr_part = torch.where(valid, face_part_id[intr.clamp(min=0)], torch.full_like(intr, -1))

    allowed = torch.zeros_like(valid)
    for part_a, part_b in train_pairs:
        ia, ib = part_name_to_idx[part_a], part_name_to_idx[part_b]
        allowed = allowed | ((recv_part == ia) & (intr_part == ib))
        allowed = allowed | ((recv_part == ib) & (intr_part == ia))

    keep = valid & allowed
    filtered_collision_idxs = torch.where(
        keep.unsqueeze(-1), collision_idxs, torch.full_like(collision_idxs, -1)
    )

    per_specimen_loss = dfp_loss_module(triangles, filtered_collision_idxs)  # (B,)

    ramp = _ramp_factor(iteration, n_ramp_iters)
    ramped = per_specimen_loss * ramp

    if not return_diagnostics:
        return ramped

    diagnostics = {"num_colliding_face_pairs": keep.float().sum(dim=1)}
    return ramped, diagnostics
