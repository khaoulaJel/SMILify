"""
Metric 6 -- Pose plausibility (joint rotation outlier detection)
====================================================================

Reasoning
---------
Every metric so far -- chamfer, F-score, displacement magnitude, per-part
precision -- is a SURFACE metric: it looks at where mesh vertices or
faces end up in space. The 23-37 case (legs individually smooth and
intact, but draped over the gaster like straps) proved this is a real
blind spot: a pose error can produce geometry that is surface-plausible
(close to real scan points, not penetrating anything) while being
anatomically absurd. No surface-distance check can catch this, because
the surface itself isn't wrong -- only where it's been rotated to.

This metric checks the SKELETON instead: `joint_rot`, the axis-angle
rotation for each of the 54 non-root joints, already saved in every
fitted .npz, needing no new geometry computation and no re-fitting.

Why cross-specimen comparison, not a fixed threshold
-------------------------------------------------------
We have no documented biomechanical range for ant leg articulation, so
an absolute "joint should not rotate more than X degrees" threshold
would be invented, not grounded -- exactly the kind of unjustified
assumption we've been avoiding throughout this project.

What we DO have: multiple specimens fit with the same template, same
loss weights, same stage schedule. A joint whose rotation magnitude is a
statistical outlier relative to the SAME joint across the other
specimens is a defensible, ground-truth-free signal: it says this
specimen's optimizer found a solution for that joint unlike every other
specimen's solution for the same joint -- consistent with a joint having
been wound into an anatomically wrong position rather than converging,
like the others, to a normal (if individually variable) pose.

Why per-joint, not a single per-specimen summary
----------------------------------------------------
Averaging rotation magnitude across all 54 joints would dilute one badly
-rotated leg joint into 53 normal ones -- the same dilution failure we
already hit twice (chamfer's area-weighted mean, per-part precision's
one-directional blindness). This metric deliberately keeps every joint
separate and reports outliers individually, then rolls them up by
anatomical part only for summary display.

Known, confirmed limitation -- NOT fixable at this metric's level
------------------------------------------------------------------------
A meaningful chunk of joints never receive enough gradient from the
vertex-based losses to have their fitted value reflect the true pose at
all:
  - w_1_r/w_2_r/w_1_l/w_2_l (zero skinning weight, see
    part_groups.get_zero_weight_joint_rot_indices): ZERO gradient, fitted
    value frozen at init.
  - Other joints (e.g. l_1_fe_r) get gradient but can still be
    substantially suppressed by pose regularization, so the fitted value
    never diverges enough to become an outlier even when the true pose
    did.

This means the metric's fitted-value z-score cannot ever detect an error
at the zero-gradient joints (there's no variance to detect, by
construction) and is fundamentally limited at heavily-regularized joints
too (the information about true severity can be lost during fitting,
before this metric ever runs). No threshold or weighting change to this
function can recover information the optimizer never captured. A real
fix would require changing the fitter/rigging (e.g. giving the waist
joints some skinning influence), not this evaluation code.

Zero-gradient joints are explicitly excluded from flagging below (not
silently miscounted as z=0/"normal") since scoring them is provably
meaningless, not merely low-confidence.
"""

import numpy as np

from fitter_3d.part_groups import get_zero_weight_joint_rot_indices


def joint_index_to_part_name(joint_rot_index: int, joint_names: list, group_dict: dict) -> str:
    """
    Maps a joint_rot array index (0-53) to its anatomical part name.
    joint_rot[i] corresponds to joint_names[i + 1], since joint_rot
    excludes the root joint (index 0, "b_t"), which is controlled
    separately by global_rot.
    """
    full_joint_idx = joint_rot_index + 1
    for part_name, joint_ids in group_dict.items():
        if full_joint_idx in joint_ids:
            return part_name
    return None


def rotation_magnitudes(joint_rot: np.ndarray) -> np.ndarray:
    """
    joint_rot: (54, 3) axis-angle vectors for one specimen.
    Returns: (54,) rotation magnitude in radians (the norm of each
    axis-angle vector IS the rotation angle, by definition of the
    axis-angle representation).
    """
    return np.linalg.norm(joint_rot, axis=1)


def cross_specimen_joint_outliers(
    specimen_joint_rots: dict,
    joint_names: list,
    group_dict: dict,
    z_threshold: float = 1.5,
) -> dict:
    """
    Compares each joint's rotation magnitude across all specimens and
    flags per-specimen, per-joint outliers via z-score.

    Args:
        specimen_joint_rots: dict specimen_name -> (54, 3) joint_rot array
        joint_names:         the 55-entry JOINT_NAMES list from part_groups.py
        group_dict:           PART_GROUPS_COARSE or PART_GROUPS_FINE
        z_threshold:          |z| above this is flagged as an outlier.
                              1.5 is deliberately permissive given a small
                              specimen count -- with few specimens, z-scores
                              are noisy; treat flags as "worth a visual
                              check", not as proof on their own.

    Returns:
        dict: specimen_name -> {
            "per_joint_magnitude": (54,) array,
            "per_joint_zscore": (54,) array,
            "flagged_joints": list of (joint_idx, part_name, magnitude, zscore),
            "flagged_by_part": dict part_name -> count of flagged joints,
        }
    """
    specimen_names = list(specimen_joint_rots.keys())

    # (n_specimens, 54) matrix of rotation magnitudes
    all_magnitudes = np.stack(
        [rotation_magnitudes(specimen_joint_rots[name]) for name in specimen_names], axis=0
    )

    joint_mean = all_magnitudes.mean(axis=0)   # (54,)
    joint_std = all_magnitudes.std(axis=0) + 1e-8  # (54,) avoid div by zero

    # Joints with zero skinning weight get zero gradient from every vertex-based
    # loss, so their fitted value never moves from init regardless of true pose --
    # see this module's docstring ("Known, confirmed limitation"). Excluded from
    # flagging, not just left to fall out naturally, so this is legible as a
    # deliberate exclusion rather than an accidental "z=0, looks normal" reading.
    zero_gradient_indices = set(get_zero_weight_joint_rot_indices().tolist())

    results = {}
    for s_idx, name in enumerate(specimen_names):
        magnitudes = all_magnitudes[s_idx]           # (54,)
        zscores = (magnitudes - joint_mean) / joint_std  # (54,)

        flagged = []
        flagged_by_part = {}
        for j in range(54):
            if j in zero_gradient_indices:
                continue
            if abs(zscores[j]) > z_threshold:
                part_name = joint_index_to_part_name(j, joint_names, group_dict)
                flagged.append((j, part_name, float(magnitudes[j]), float(zscores[j])))
                if part_name is not None:
                    flagged_by_part[part_name] = flagged_by_part.get(part_name, 0) + 1

        results[name] = {
            "per_joint_magnitude": magnitudes,
            "per_joint_zscore": zscores,
            "flagged_joints": flagged,
            "flagged_by_part": flagged_by_part,
            "excluded_zero_gradient_joints": sorted(zero_gradient_indices),
        }

    return results
