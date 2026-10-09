#!/usr/bin/env python3
"""
Apply the trained part-based leg-pose regressor to a corpus and score it
with the identical metric used throughout this experiment series, so the
output slots directly into your comparison table next to:
  zero_init            23.087 deg
  cheap_pca_heuristic  28.520 deg
  retrieval (dead)     32.040 deg  (== random pairing, 32.658 +/- 2.512)

Usage
-----
python infer_leg_pose_init.py \
    --corpus diagnostics/moonshot/synth_clean/ground_truth.npz \
    --checkpoint diagnostics/anatomical_pose_init/learned_init_20260820/best_model.pt \
    --out diagnostics/anatomical_pose_init/out_ceiling_20260820/learned_init.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from train_leg_pose_regressor import (
    LegPoseRegressor, body_core_canonicalize, rot6d_to_matrix,
    axis_angle_to_matrix, geodesic_loss_deg,
)


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Inverse of axis_angle_to_matrix, batched. R: (..., 3, 3)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    denom = 2.0 * torch.sin(theta).clamp(min=1e-8)
    rx = (R[..., 2, 1] - R[..., 1, 2]) / denom
    ry = (R[..., 0, 2] - R[..., 2, 0]) / denom
    rz = (R[..., 1, 0] - R[..., 0, 1]) / denom
    axis = torch.stack([rx, ry, rz], dim=-1)
    return axis * theta.unsqueeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_clean/ground_truth.npz")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="diagnostics/anatomical_pose_init/out_ceiling_20260820/learned_init.json")
    ap.add_argument(
        "--out_init_npz",
        default=None,
        help=(
            "If set, also write a fitter-compatible joint_rot (N,54,3) + names npz here "
            "(--init_joint_rot_from drop-in), same convention as cheap_init.npz: predicted "
            "leg rows, zero (rest pose) everywhere else."
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vertex_labels = ckpt["vertex_labels"]
    leg_rows = ckpt["leg_rows"]
    n_points = ckpt["n_points"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LegPoseRegressor(ckpt["in_dim"], ckpt["n_leg_joints"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gt = np.load(args.corpus, allow_pickle=True)
    verts_all = gt["verts"]
    # newer corpora (synth_large*) store both joint_rot (matrices) and joint_rot_aa
    # (axis-angle); legacy corpora (synth_clean) store only joint_rot, already axis-angle.
    jr_gt_all = gt["joint_rot_aa"] if "joint_rot_aa" in gt.files else gt["joint_rot"]
    names = [str(n) for n in gt["names"]]

    n_chains = ckpt["n_chains"]
    leg_idx_all = np.where(vertex_labels > 0)[0]
    core_idx_all = np.where(vertex_labels == 0)[0]

    per_specimen_err = {}
    all_joint_rot = np.zeros((len(names), jr_gt_all.shape[1], 3), dtype=np.float32)
    with torch.no_grad():
        for i, name in enumerate(names):
            pts_canon = body_core_canonicalize(verts_all[i].astype(np.float32), vertex_labels).astype(np.float32)

            n_leg = min(len(leg_idx_all), n_points // 2)
            n_core = n_points - n_leg
            sel_leg = np.random.choice(leg_idx_all, n_leg, replace=False)
            sel_core = np.random.choice(core_idx_all, min(n_core, len(core_idx_all)),
                                         replace=len(core_idx_all) < n_core)
            sel = np.concatenate([sel_leg, sel_core])

            xyz = pts_canon[sel]
            onehot = np.eye(n_chains + 1, dtype=np.float32)[vertex_labels[sel]]
            feat = np.concatenate([xyz, onehot], axis=1)
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)

            d6 = model(feat_t)                       # (1, L, 6)
            R_pred = rot6d_to_matrix(d6)[0]           # (L, 3, 3)
            jr_pred = matrix_to_axis_angle(R_pred).cpu().numpy()  # (L, 3)

            jr_true = jr_gt_all[i][leg_rows]
            R_gt = axis_angle_to_matrix(torch.from_numpy(jr_true.astype(np.float32)))
            err_deg = geodesic_loss_deg(R_pred, R_gt).item()
            per_specimen_err[name] = err_deg
            all_joint_rot[i][leg_rows] = jr_pred

    errs = np.array(list(per_specimen_err.values()))
    result = {
        "method": "learned_partbased_pointnet_bodycore_canon_6drot",
        "n_specimens": len(names),
        "per_specimen_leg_rot_err_deg": per_specimen_err,
        "summary": {
            "mean_leg_rot_err_deg": float(errs.mean()),
            "median_leg_rot_err_deg": float(np.median(errs)),
            "frac_specimens_under_15deg": float((errs <= 15.0).mean()),
        },
        "comparison": {
            "zero_init_mean_deg": 23.087,
            "cheap_pca_heuristic_init_mean_deg": 28.520,
            "retrieval_mean_deg": 32.040,
            "retrieval_random_baseline_mean_deg": 32.658,
            "robust_basin_upper_bound_deg": 15.0,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[infer] mean leg-joint error = {errs.mean():.3f} deg "
          f"(median {np.median(errs):.3f}), {(errs<=15).mean()*100:.0f}% under 15 deg")
    print(f"[infer] wrote {args.out}")

    if args.out_init_npz:
        os.makedirs(os.path.dirname(args.out_init_npz), exist_ok=True)
        np.savez(
            args.out_init_npz,
            joint_rot=all_joint_rot,
            names=np.array(names),
        )
        print(f"[infer] wrote {args.out_init_npz} (fitter-compatible --init_joint_rot_from)")


if __name__ == "__main__":
    main()