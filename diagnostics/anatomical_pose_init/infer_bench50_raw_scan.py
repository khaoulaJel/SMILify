#!/usr/bin/env python3
"""Phase 3+4 (2026-08-20): apply the EXISTING trained checkpoint
(learned_init_20260820_25pc/best_model.pt) to raw bench50_clean scans, using the validated
correspondence-free representation:

  raw scan .obj
    -> normalize (load_meshes convention: center, divide by max abs coord -- verified this
       matches the template's own native scale, bbox diag ~2.35 both ways)
    -> estimate_global_pose (ACTUAL PCA rotation, NOT the identity shortcut synth_clean-only
       code uses -- per explicit instruction, since real scans are not canonically pre-aligned)
    -> 7-way body-core+leg chain assignment (assign_points_to_parts_chain, validated:
       macro-F1 0.608->0.697, ML/MR leg recall 0.349->0.768 vs the old coxa method)
    -> body_core_canonicalize (same function training used, generic over any same-length
       points+labels, no template-vertex-index dependence)
    -> same point sampling / onehot feature scheme as training
    -> trained PointNet regressor
    -> predicted leg-joint axis-angle

NOT fitting yet (explicit earlier agreement): this only checks whether predictions look
anatomically meaningful -- magnitude sanity, left/right and front/mid/hind structure, and
run-to-run stability under independent random point subsamples (a stability/confidence proxy
in place of a ground-truth comparison, which doesn't exist for real unposed scans).
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
from geom_leg_init import (  # noqa: E402
    leg_chains, rest_chain_points, body_core_chain_points, assign_points_to_parts_chain,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cheap_anatomical_init import _pca_axes, estimate_global_pose, _rotmat_to_aa  # noqa: E402
from train_leg_pose_regressor import LegPoseRegressor, body_core_canonicalize, rot6d_to_matrix  # noqa: E402

MODEL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
CHECKPOINT = "diagnostics/anatomical_pose_init/learned_init_20260820_25pc/best_model.pt"
BENCH50_DIR = "diagnostics/moonshot/bench50_clean"
N_RESAMPLE_TRIALS = 3  # for stability check


def load_normalized(path):
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
    chain_keys = list(chains.keys())

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    leg_rows = ckpt["leg_rows"]
    n_points = ckpt["n_points"]
    n_chains = ckpt["n_chains"]
    model = LegPoseRegressor(ckpt["in_dim"], ckpt["n_leg_joints"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"checkpoint: n_points={n_points} n_chains={n_chains} n_leg_joints={ckpt['n_leg_joints']}")

    files = sorted(glob.glob(os.path.join(BENCH50_DIR, "*.obj")))
    print(f"{len(files)} bench50 raw scans\n")

    all_mean_mag = []
    all_stability = []
    per_leg_mag = {k: [] for k in chain_keys}

    for f in files:
        name = os.path.basename(f)
        target_np = load_normalized(f)

        R, trans_np = estimate_global_pose(target_np, template_axes, template_centroid, root_rest_np)
        global_rot_aa = torch.as_tensor(_rotmat_to_aa(R), dtype=torch.float32)  # ACTUAL PCA rotation
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
            print(f"{name:<55} SKIPPED (degenerate labeling: {len(leg_idx_all)} leg / {len(core_idx_all)} core pts)")
            continue

        trial_preds = []
        for trial in range(N_RESAMPLE_TRIALS):
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
                jr_pred = matrix_to_axis_angle(R_pred).numpy()  # (L,3)
            trial_preds.append(jr_pred)

        trial_preds = np.stack(trial_preds)  # (T,L,3)
        mean_pred = trial_preds.mean(axis=0)
        mag_deg = np.degrees(np.linalg.norm(mean_pred, axis=-1))  # (L,)
        stability_deg = np.degrees(np.linalg.norm(trial_preds - mean_pred[None], axis=-1)).mean()

        all_mean_mag.append(mag_deg.mean())
        all_stability.append(stability_deg)

        # per-leg mean magnitude (average over that leg's 6 joint rows)
        for i, k in enumerate(chain_keys):
            rows_this_leg = [r for r, j in enumerate(leg_rows) if j in [c - 1 for c in chains[k]]]
            if rows_this_leg:
                per_leg_mag[k].append(mag_deg[rows_this_leg].mean())

        print(f"{name:<55} mean_pred_mag={mag_deg.mean():5.1f} deg   "
              f"resample_stability={stability_deg:5.2f} deg   "
              f"leg/core pts={len(leg_idx_all)}/{len(core_idx_all)}")

    print()
    print("=== pooled over all scored bench50 specimens ===")
    print(f"mean predicted rotation magnitude: {np.mean(all_mean_mag):.2f} deg "
          f"(training-corpus target band was ~23 deg mean init error)")
    print(f"mean resample stability (lower=more stable): {np.mean(all_stability):.2f} deg")
    print()
    print("per-leg mean predicted magnitude (check for anatomically sane L/R symmetry):")
    for k in chain_keys:
        v = per_leg_mag[k]
        print(f"  {k}: mean={np.mean(v):.2f} deg  std={np.std(v):.2f} deg  n={len(v)}")


if __name__ == "__main__":
    main()
