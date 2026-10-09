"""Fresh, deliberately simple leg-pose heuristic for Arm B (cheap_anatomical), written after
`fitter_3d.geom_leg_init`'s per-segment chain-IK was measured (2026-08-20, `probe_B0_geom_init_
validation.py` run for the first time end-to-end) to make EVERY leg joint substantially WORSE
than zero-init (leg_distal position error 0.436 vs 0.163 zero-init, 3x worse; leg_prox 0.124 vs
0.056, also worse) -- not a wiring bug, reproduced across all 36 leg joints on `synth_clean`. Per
explicit user decision, geom_leg_init's per-segment band-estimation + chain IK
(`estimate_leg_curve`/`solve_chain_rotations`) is NOT reused here. This module still reuses the
two PURE-ARITHMETIC pieces of `geom_leg_init` that are not implicated in that failure (no
per-segment estimation happens in either):
  - `analytic_coxa_anchors`: 6 coxa positions from a rigid global fit + template rest geometry.
  - `assign_points_to_legs`: nearest-of-6-analytic-anchor point clustering.
  - `_graph_distance_from_anchor`: a generic k-NN graph shortest-path utility (Isomap-style),
    used here only to find each leg's farthest assigned point, not to bin points into bands.

METHOD: treat each leg as a single RIGID ROD pivoting only at the coxa -- one whole-leg
direction estimate, one rotation, applied once. No per-segment decomposition, no sequential
re-estimation, nothing left for a multi-stage compounding error to build up in:
  1. Find the assigned point farthest (by graph/geodesic distance, robust to a bent leg unlike
     Euclidean distance -- same reasoning `geom_leg_init`'s docstring already established) from
     that leg's analytic coxa anchor. This approximates "the distal extremity."
  2. Rotate the REST-pose coxa->pretarsus vector to point at that farthest point (single
     `_align_rotation` call, the same closed-form minimal-rotation formula `geom_leg_init` uses).
  3. Assign that ONE rotation, converted to the coxa joint's LOCAL frame, to the coxa's
     `joint_rot` row. Every other joint in the chain (trochanter, femur, tibia, tarsus,
     pretarsus) keeps ZERO local rotation, i.e. the whole rest-pose sub-chain swings rigidly
     with the coxa -- deliberately not attempting per-segment articulation (matches the
     protocol's explicit allowance: "leave distal joints... at or near zero if the heuristic
     cannot reliably estimate them").

Falls back to zero rotation (leg stays at rest) if fewer than `MIN_PTS` points were assigned to
that leg. Deterministic, no ground truth, no learned model, no fitting run required.
"""

import torch

from fitter_3d.geom_leg_init import (  # noqa: F401  (re-exported for callers)
    leg_chains,
    rest_joint_positions,
    analytic_coxa_anchors,
    assign_points_to_legs,
    _align_rotation,
    _graph_distance_from_anchor,
)

MIN_PTS = 4


def estimate_leg_rotations(global_rot_aa, trans, rest_J, jnames, target_pts):
    """(N_POSE,3) axis-angle joint_rot, legs from the single-rotation rigid-rod heuristic,
    everything else zero. Same call signature as `geom_leg_init.init_joint_rot_for_specimen`.
    """
    n_pose = len(jnames) - 1
    out = torch.zeros(n_pose, 3, dtype=trans.dtype, device=trans.device)
    chains = leg_chains(jnames)
    anchors, R_root = analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
    assigned = assign_points_to_legs(target_pts, anchors)

    for key, chain_idx in chains.items():
        pts = assigned[key]
        coxa_pos = anchors[key]
        if pts.shape[0] < MIN_PTS:
            continue  # leave this leg at rest
        d = _graph_distance_from_anchor(coxa_pos, pts)
        finite = torch.isfinite(d)
        if not bool(finite.any()):
            continue
        far_idx = int(torch.argmax(torch.where(finite, d, torch.full_like(d, -1.0))))
        far_pt = pts[far_idx]

        rest_rod = rest_J[chain_idx[5]] - rest_J[chain_idx[0]]  # rest coxa -> pretarsus
        desired_rod = far_pt - coxa_pos
        R_align = _align_rotation(rest_rod[None, :], desired_rod[None, :])[0]
        R_local_coxa = R_root.transpose(-1, -2) @ R_align

        from pytorch3d.transforms import matrix_to_axis_angle

        row = chain_idx[0] - 1  # joint_rot row = global joint idx - 1 (root has no row)
        out[row] = matrix_to_axis_angle(R_local_coxa.unsqueeze(0))[0]

    return out
