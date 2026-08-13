"""
part_groups.py

Joint -> anatomical part mapping (derived from SMIL's skinning weights),
plus a notion of which parts are anatomically ADJACENT -- i.e. genuinely
expected to touch -- versus NON-ADJACENT -- i.e. should never occupy the
same space in a correct fit.

This exists to support fitter_3d/penetration_loss.py: the penetration
penalty must only be checked between parts that are NOT expected to
touch. A leg's coxa is expected to sit right at the thorax (that's the
joint); flagging that as "penetration" would drown the real signal.
Without this distinction every normal joint contact would look like an
error.

Adjacency is derived programmatically from `kintree_table`, the same
skeleton hierarchy already used by SMAL3DFitter.get_joint_scales() in
trainer.py -- not hand-typed, so it can't silently drift from the model.
"""

import pickle
import numpy as np
import config

JOINT_NAMES = [
    "b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5",
    "l_1_co_r", "l_1_tr_r", "l_1_fe_r", "l_1_ti_r", "l_1_ta_r", "l_1_pt_r",
    "l_2_co_r", "l_2_tr_r", "l_2_fe_r", "l_2_ti_r", "l_2_ta_r", "l_2_pt_r",
    "l_3_co_r", "l_3_tr_r", "l_3_fe_r", "l_3_ti_r", "l_3_ta_r", "l_3_pt_r",
    "w_1_r", "w_2_r",
    "l_1_co_l", "l_1_tr_l", "l_1_fe_l", "l_1_ti_l", "l_1_ta_l", "l_1_pt_l",
    "l_2_co_l", "l_2_tr_l", "l_2_fe_l", "l_2_ti_l", "l_2_ta_l", "l_2_pt_l",
    "l_3_co_l", "l_3_tr_l", "l_3_fe_l", "l_3_ti_l", "l_3_ta_l", "l_3_pt_l",
    "w_1_l", "w_2_l",
    "b_h",
    "ma_r", "an_1_r", "an_2_r", "an_3_r",
    "ma_l", "an_1_l", "an_2_l", "an_3_l",
]
assert len(JOINT_NAMES) == 55

# Coarse anatomical grouping -- used by penetration_loss.py (inter-part
# checks) and the eval-metrics panel.
PART_GROUPS_COARSE = {
    "head":     [46],
    "mandible": [47, 51],
    "antenna":  [48, 49, 50, 52, 53, 54],
    "thorax":   [0],
    "waist":    [24, 25, 44, 45],
    "gaster":   [1, 2, 3, 4, 5],
    "legs":     [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
}

# Fine anatomical grouping -- each leg kept separate (PART_GROUPS_COARSE's
# "legs" merges all six, which is wrong for the pose-plausibility panel:
# comparing a joint's rotation against a group spanning six different legs
# would blend six different articulations into one baseline).
PART_GROUPS_FINE = {
    "thorax":    [0],
    "gaster":    [1, 2, 3, 4, 5],
    "leg1_r":    [6, 7, 8, 9, 10, 11],
    "leg2_r":    [12, 13, 14, 15, 16, 17],
    "leg3_r":    [18, 19, 20, 21, 22, 23],
    "waist":     [24, 25, 44, 45],
    "leg1_l":    [26, 27, 28, 29, 30, 31],
    "leg2_l":    [32, 33, 34, 35, 36, 37],
    "leg3_l":    [38, 39, 40, 41, 42, 43],
    "head":      [46],
    "mandible":  [47, 51],
    "antenna_r": [48, 49, 50],
    "antenna_l": [52, 53, 54],
}


def _load_smal_data():
    with open(config.SMAL_FILE, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def load_skinning_weights():
    """Loads the (n_verts, 55) skinning weight matrix from config.SMAL_FILE."""
    return _load_smal_data()["weights"]


def load_kintree_table():
    """
    Loads the (2, 55) kinematic tree table from config.SMAL_FILE.
    Row 0: parent joint index for each of the 55 joints.
    Row 1: joint index itself (0..54).
    Same array used by SMAL3DFitter.get_joint_scales() in trainer.py.
    """
    return np.asarray(_load_smal_data()["kintree_table"])


def get_vertex_joint_ids(weights=None):
    """(n_verts,) array: index (0-54) of the joint with max skinning weight per vertex."""
    if weights is None:
        weights = load_skinning_weights()
    return np.argmax(weights, axis=1)


def get_zero_weight_joint_rot_indices(weights=None):
    """
    (np.array) joint_rot indices (0-53, i.e. joint_id - 1 -- joint_rot
    excludes the root) for joints with ZERO skinning weight over every
    vertex -- pivot-only joints with no owned mesh region.

    Why this matters: no vertex-based loss (chamfer/edge/normal/
    laplacian/penetration) has any gradient path to a joint's rotation
    if no vertex is skinned to it. Its fitted value therefore never
    moves from initialization, regardless of the true pose. Any
    per-joint ground-truth comparison or outlier metric (see
    eval_metrics_pose_plausibility.py) is measuring something
    meaningless for these joints -- this is a fitter/rigging limitation,
    not a metric bug.
    """
    vertex_joint_ids = get_vertex_joint_ids(weights)
    counts = np.bincount(vertex_joint_ids, minlength=55)
    zero_joint_ids = np.where(counts == 0)[0]
    zero_joint_ids = zero_joint_ids[zero_joint_ids > 0]  # root (id 0) always owns vertices; exclude defensively
    return zero_joint_ids - 1  # joint_id -> joint_rot index


def get_part_vertex_indices(group_dict=PART_GROUPS_COARSE, weights=None):
    """dict: part_name -> np.array of vertex indices belonging to that part."""
    vertex_joint_ids = get_vertex_joint_ids(weights)
    joint_to_part = {}
    for part_name, joint_ids in group_dict.items():
        for j in joint_ids:
            joint_to_part[j] = part_name

    part_indices = {part_name: [] for part_name in group_dict.keys()}
    for v_idx, j_id in enumerate(vertex_joint_ids):
        part_name = joint_to_part.get(int(j_id))
        if part_name is not None:
            part_indices[part_name].append(v_idx)

    return {k: np.array(v, dtype=np.int64) for k, v in part_indices.items()}


def _joint_parent_map(kintree_table):
    """dict: joint_idx -> parent_joint_idx (or None for the root)."""
    parents = {}
    for j in range(kintree_table.shape[1]):
        p = int(kintree_table[0, j])
        parents[j] = p if p != j else None
    return parents


def build_part_adjacency(group_dict=PART_GROUPS_COARSE, kintree_table=None):
    """
    Returns a set of frozenset({partA, partB}) pairs that are anatomically
    ADJACENT -- i.e. contain joints that are direct parent/child of each
    other in the kinematic tree, or are the same part. These pairs are
    expected to touch and must be EXCLUDED from interpenetration checks.

    A part is also trivially adjacent to itself (not included in the
    returned set; callers should skip self-pairs separately).
    """
    if kintree_table is None:
        kintree_table = load_kintree_table()

    joint_to_part = {}
    for part_name, joint_ids in group_dict.items():
        for j in joint_ids:
            joint_to_part[j] = part_name

    parents = _joint_parent_map(kintree_table)

    adjacent_pairs = set()
    for j, p in parents.items():
        if p is None:
            continue
        part_j = joint_to_part.get(j)
        part_p = joint_to_part.get(p)
        if part_j is None or part_p is None:
            continue
        if part_j != part_p:
            adjacent_pairs.add(frozenset({part_j, part_p}))

    return adjacent_pairs


def get_non_adjacent_pairs(group_dict=PART_GROUPS_COARSE, kintree_table=None):
    """
    Returns a list of (partA, partB) tuples for every pair of DISTINCT
    parts in group_dict that are NOT anatomically adjacent -- i.e. pairs
    that should never occupy the same space in a correctly-fit specimen.
    This is exactly the set of pairs fitter_3d.penetration_loss checks.
    """
    adjacent = build_part_adjacency(group_dict, kintree_table)
    names = list(group_dict.keys())
    non_adjacent = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if frozenset({a, b}) not in adjacent:
                non_adjacent.append((a, b))
    return non_adjacent


if __name__ == "__main__":
    part_idx = get_part_vertex_indices(PART_GROUPS_COARSE)
    print("Coarse part vertex counts:")
    for name, idx in part_idx.items():
        print(f"  {name:<10} {len(idx):>6} vertices")

    print("\nAnatomically ADJACENT part pairs (expected contact, excluded from checks):")
    for pair in sorted(build_part_adjacency(PART_GROUPS_COARSE), key=lambda s: sorted(s)):
        print(f"  {' <-> '.join(sorted(pair))}")

    print("\nNON-ADJACENT part pairs (checked for interpenetration):")
    for a, b in get_non_adjacent_pairs(PART_GROUPS_COARSE):
        print(f"  {a} <-> {b}")
