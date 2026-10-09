"""Ground-truth anatomical labels for every TEMPLATE vertex/face, at the granularities the
correspondence-accuracy metric needs: leg identity, leg segment, antenna side, antenna segment.

Built from `M["jnames"]`/`M["dominant"]` (per-vertex dominant skinning-weight joint), the same
source `fitter_3d.trainer_hierarchical.vertex_groups` and the D2a/D2b probes used -- not a new
labeling scheme, just made complete: those probes only ever labeled `l_*` joints (legs) and
folded everything else (head, mandible, antenna) into a single unlabeled bucket. Antenna is
split out here because the task calls for tracking it separately from legs, and because its
joint structure (`an_1/an_2/an_3`, proximal->distal per side, see JOINT_NAMES in
`fitter_3d/part_groups.py`) is anatomically the same kind of thin-chain structure as a leg:

    an_1_r  71 dominant verts     (scape-equivalent, proximal)
    an_2_r 106 dominant verts     (pedicel/funiculus-equivalent, mid)
    an_3_r   4 dominant verts     (flagellum tip, distal -- as thin as pretarsus's 8)

confirmed by direct query against `OmniAnt_25PCs_joint_limited.pkl` (2026-08-19).
"""

import numpy as np

LEG_SEGMENTS = ("co", "tr", "fe", "ti", "ta", "pt")
ANTENNA_SEGMENTS = ("an_1", "an_2", "an_3")
LEGS = tuple(f"l{k}_{s}" for k in (1, 2, 3) for s in ("r", "l"))
SIDES = ("r", "l")


def vertex_labels(jnames, dominant):
    """Per-TEMPLATE-vertex labels, computed once from the dominant skinning-weight joint.

    Returns a dict of parallel (n_verts,) arrays:
      region        -- 'leg' | 'antenna' | 'other'
      leg_id        -- 'l{1,2,3}_{r,l}' where region=='leg', else None
      leg_seg       -- one of LEG_SEGMENTS where region=='leg', else None
      ant_side      -- 'r' | 'l' where region=='antenna', else None
      ant_seg       -- one of ANTENNA_SEGMENTS where region=='antenna', else None
    """
    n = len(dominant)
    region = np.full(n, "other", dtype=object)
    leg_id = np.full(n, None, dtype=object)
    leg_seg = np.full(n, None, dtype=object)
    ant_side = np.full(n, None, dtype=object)
    ant_seg = np.full(n, None, dtype=object)

    for j, nm in enumerate(jnames):
        m = dominant == j
        if not m.any():
            continue
        if nm.startswith("l_"):
            # 'l_2_fe_r' -> leg 2, segment 'fe', side 'r'
            bits = nm.split("_")
            k, seg, side = bits[1], bits[2], bits[-1]
            region[m] = "leg"
            leg_id[m] = f"l{k}_{side}"
            leg_seg[m] = seg
        elif nm.startswith("an_"):
            # 'an_2_r' -> segment 'an_2', side 'r'
            bits = nm.split("_")
            seg, side = f"an_{bits[1]}", bits[-1]
            region[m] = "antenna"
            ant_side[m] = side
            ant_seg[m] = seg

    return dict(region=region, leg_id=leg_id, leg_seg=leg_seg, ant_side=ant_side, ant_seg=ant_seg)


def face_labels(faces, vlabels):
    """Per-TEMPLATE-face labels, by majority vote of its 3 vertices' vertex_labels.

    Same reduction `face_group`/`face_group_by_segment` in
    `diagnostics/registration_failure/probe_d2a_sample_starvation.py` use (majority of 3, ties
    broken by first vertex -- faces are tiny enough this never matters in practice), applied to
    every field in `vlabels` at once so leg and antenna faces are labeled in one pass.
    """
    out = {}
    for key, varr in vlabels.items():
        fvals = varr[faces]  # (F, 3)
        col = np.empty(faces.shape[0], dtype=object)
        for i in range(faces.shape[0]):
            vals, counts = np.unique(fvals[i].astype(str), return_counts=True)
            winner = vals[np.argmax(counts)]
            col[i] = None if winner == "None" else winner
        out[key] = col
    return out
