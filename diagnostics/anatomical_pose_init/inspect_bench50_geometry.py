#!/usr/bin/env python3
"""Stage 4A/4B geometry inspection: for all 50 bench50_clean raw scans, compute the
correspondence-free rigid alignment (estimate_global_pose, ACTUAL PCA rotation, not the
identity shortcut cheap_anatomical_init.py uses for canonically-pre-aligned synth_clean),
the 6 analytic coxa anchors, and per-point distance-to-nearest-coxa vs distance-to-root.
Purely descriptive -- no threshold chosen yet, no fitting, no learned-initializer call.
"""
import glob
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.transforms import matrix_to_axis_angle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
from geom_leg_init import leg_chains, analytic_coxa_anchors  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cheap_anatomical_init import _pca_axes, estimate_global_pose, _rotmat_to_aa  # noqa: E402

MODEL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"


def load_normalized(path):
    """Same convention as fitter_3d.utils.load_meshes: center, divide by max abs coord."""
    v, _, _ = load_obj(path, load_textures=False)
    v = v.numpy().astype(np.float64)
    c = v.mean(0)
    v = v - c
    scale = np.abs(v).max()
    return v / scale


def main():
    with open(MODEL, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    jnames = list(dd["J_names"])
    Jr = np.asarray(dd["J_regressor"])
    vt = np.asarray(dd["v_template"])
    rest_J_np = Jr @ vt
    rest_J = torch.as_tensor(rest_J_np, dtype=torch.float32)
    root_rest_np = rest_J_np[0]
    template_axes, template_centroid = _pca_axes(vt.astype(np.float64))
    chains = leg_chains(jnames)

    files = sorted(glob.glob("diagnostics/moonshot/bench50_clean/*.obj"))
    print(f"{len(files)} bench50 specimens")

    all_ratios = []
    all_leg_counts = []
    for f in files:
        name = os.path.basename(f)
        target_np = load_normalized(f)

        R, trans_np = estimate_global_pose(target_np, template_axes, template_centroid, root_rest_np)
        global_rot_aa_np = _rotmat_to_aa(R)  # ACTUAL PCA rotation -- not identity

        global_rot_aa = torch.as_tensor(global_rot_aa_np, dtype=torch.float32)
        trans = torch.as_tensor(trans_np, dtype=torch.float32)
        target_pts = torch.as_tensor(target_np, dtype=torch.float32)

        anchors, R_root = analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
        # root (thorax) position under the SAME rigid transform, for the body-core criterion
        root_pos = trans + root_rest_np  # rotation about root pivots root itself, so R has no effect on root_pos

        anchor_keys = list(anchors.keys())
        anchor_stack = torch.stack([anchors[k] for k in anchor_keys])  # (6,3)
        d_to_coxa = (target_pts[:, None, :] - anchor_stack[None, :, :]).pow(2).sum(-1).sqrt()  # (P,6)
        d_nearest_coxa, nearest_idx = d_to_coxa.min(dim=1)
        d_to_root = (target_pts - root_pos[None, :]).pow(2).sum(-1).sqrt()

        ratio = (d_nearest_coxa / d_to_root.clamp_min(1e-6)).numpy()
        all_ratios.append(ratio)

        leg_counts = [int((nearest_idx == i).sum()) for i in range(6)]
        all_leg_counts.append(leg_counts)

        print(f"{name:<55} leg_pt_counts(min/max)={min(leg_counts):>5}/{max(leg_counts):>5}  "
              f"ratio p10/p50/p90={np.percentile(ratio,10):.2f}/{np.percentile(ratio,50):.2f}/{np.percentile(ratio,90):.2f}")

    all_ratios_flat = np.concatenate(all_ratios)
    all_leg_counts = np.array(all_leg_counts)
    print()
    print("=== pooled across all 50 specimens ===")
    print("d_nearest_coxa / d_to_root percentiles:",
          {p: round(float(np.percentile(all_ratios_flat, p)), 3) for p in [5, 10, 25, 50, 75, 90, 95]})
    print("per-leg point count stats (min/median/max across specimens, per leg slot):")
    for i in range(6):
        col = all_leg_counts[:, i]
        print(f"  leg slot {i}: min={col.min()} median={np.median(col):.0f} max={col.max()}")


if __name__ == "__main__":
    main()
