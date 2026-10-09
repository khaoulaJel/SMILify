#!/usr/bin/env python3
"""Phase 3 (2026-08-20): produce a fitter-compatible learned_init.npz for the 50 bench50_clean
raw scans, using the validated correspondence-free representation (see
infer_bench50_raw_scan.py's module docstring for the full pipeline) and the EXISTING trained
checkpoint. Single deterministic sample per specimen (seed fixed) -- the earlier stability
check showed resample variance is small (~0.48 deg mean), so one sample is a reasonable,
non-degenerate choice for the actual initialization fed to the fitter.

Output convention matches cheap_init.npz/learned_init.npz: joint_rot (N,54,3) axis-angle,
zeros for non-leg joints, names (N,) -- a drop-in for optimise_hierarchical.py
--init_joint_rot_from.
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
from geom_leg_init import (
    leg_chains, rest_chain_points, body_core_chain_points, assign_points_to_parts_chain,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cheap_anatomical_init import _pca_axes, estimate_global_pose, _rotmat_to_aa
from train_leg_pose_regressor import LegPoseRegressor, body_core_canonicalize, rot6d_to_matrix

MODEL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
CHECKPOINT = "diagnostics/anatomical_pose_init/learned_init_lowpose05/best_model.pt"
BENCH50_DIR = "diagnostics/moonshot/bench50_clean"
OUT = "diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc/bench50_learned_init_lowpose05.npz"
SEED = 0


def load_normalized(path):
    v, _, _ = load_obj(path, load_textures=False)
    v = v.numpy().astype(np.float64)
    c = v.mean(0)
    v = v - c
    scale = np.abs(v).max()
    return v / scale


def main():
    np.random.seed(SEED)
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

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    leg_rows = ckpt["leg_rows"]
    n_points = ckpt["n_points"]
    n_chains = ckpt["n_chains"]
    model = LegPoseRegressor(ckpt["in_dim"], ckpt["n_leg_joints"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    files = sorted(glob.glob(os.path.join(BENCH50_DIR, "*.obj")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    print(f"{len(files)} bench50 raw scans")

    all_joint_rot = np.zeros((len(files), 54, 3), dtype=np.float32)
    n_skipped = 0
    for i, f in enumerate(files):
        target_np = load_normalized(f)
        R, trans_np = estimate_global_pose(target_np, template_axes, template_centroid, root_rest_np)
        global_rot_aa = torch.as_tensor(_rotmat_to_aa(R), dtype=torch.float32)
        trans = torch.as_tensor(trans_np, dtype=torch.float32)
        target_pts = torch.as_tensor(target_np, dtype=torch.float32)

        rest_poly = rest_chain_points(rest_J, chains, global_rot_aa, trans)
        core_poly = body_core_chain_points(rest_J, jnames, global_rot_aa, trans)
        labels, _ = assign_points_to_parts_chain(target_pts, rest_poly, core_poly)
        labels_np = labels.numpy()

        pts_np = target_np.astype(np.float32)
        pts_canon = body_core_canonicalize(pts_np, labels_np).astype(np.float32)

        leg_idx_all = np.where(labels_np > 0)[0]
        core_idx_all = np.where(labels_np == 0)[0]
        if len(leg_idx_all) < 50 or len(core_idx_all) < 50:
            print(f"[skip -> zero init] {names[i]}: degenerate labeling "
                  f"({len(leg_idx_all)} leg / {len(core_idx_all)} core pts)")
            n_skipped += 1
            continue

        n_leg = min(len(leg_idx_all), n_points // 2)
        n_core = n_points - n_leg
        sel_leg = np.random.choice(leg_idx_all, n_leg, replace=False)
        sel_core = np.random.choice(core_idx_all, min(n_core, len(core_idx_all)),
                                     replace=len(core_idx_all) < n_core)
        sel = np.concatenate([sel_leg, sel_core])
        xyz = pts_canon[sel]
        onehot = np.eye(n_chains + 1, dtype=np.float32)[labels_np[sel]]
        feat = np.concatenate([xyz, onehot], axis=1)
        feat_t = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            d6 = model(feat_t)
            R_pred = rot6d_to_matrix(d6)[0]
            jr_pred = matrix_to_axis_angle(R_pred).numpy()
        all_joint_rot[i][leg_rows] = jr_pred
        print(f"[ok] {names[i]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, joint_rot=all_joint_rot, names=np.array(names))
    print(f"\nwrote {OUT} ({len(files)} specimens, {n_skipped} fell back to zero-init due to degenerate labeling)")


if __name__ == "__main__":
    main()
