#!/usr/bin/env python3
"""Oracle check for `solve_chain_ik_multi` (tip + one mid-chain waypoint), same discipline as
`ik_init_oracle_PROBE.py`: feed the TRUE positions (tip and waypoint after segment 2, i.e. the
'ti' joint) and see how much closer this gets to GT than the single-tip version did (~21 deg
mean per-joint gap, confirmed genuine redundancy, not a solver bug, in the prior two probes).
"""
import os
import sys

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from geom_leg_init import leg_chains, rest_joint_positions, solve_chain_ik_multi  # noqa: E402
from joint_limits import joint_limit_tensors  # noqa: E402

import pickle


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def fk_all(global_rot_mat, coxa_pos, rest_dirs, theta):
    """Returns list of 5 world positions, one after each segment (index 4 = tip)."""
    R_g = global_rot_mat
    pos = coxa_pos
    out = []
    for i in range(5):
        R_g = R_g @ axis_angle_to_matrix(theta[i].unsqueeze(0))[0]
        pos = pos + R_g @ rest_dirs[i]
        out.append(pos)
    return out


def geodesic_deg(aa_a, aa_b):
    Ra = axis_angle_to_matrix(torch.as_tensor(aa_a, dtype=torch.float32).unsqueeze(0))[0]
    Rb = axis_angle_to_matrix(torch.as_tensor(aa_b, dtype=torch.float32).unsqueeze(0))[0]
    rel = Ra.T @ Rb
    aa_rel = matrix_to_axis_angle(rel.unsqueeze(0))[0]
    return float(torch.rad2deg(aa_rel.norm()))


def main():
    model_path = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
    corpus_path = "diagnostics/moonshot/synth_clean/ground_truth.npz"
    waypoint_seg_idx = 2  # position after fe->ti, the middle joint of the 6-joint chain

    dd = load_model(model_path)
    jnames = list(dd["J_names"])
    chains = leg_chains(jnames)
    rest_J = rest_joint_positions(dd["J_regressor"], dd["v_template"])
    min_limits_all, max_limits_all = joint_limit_tensors(dd, device="cpu")

    gt = np.load(corpus_path, allow_pickle=True)
    jr_gt_aa = gt["joint_rot"]
    N = jr_gt_aa.shape[0]

    R_root = torch.eye(3)
    root_rest = rest_J[0]

    pos_residuals = []
    joint_deg_errs = {seg: [] for seg in ("co", "tr", "fe", "ti", "ta")}

    for s in range(N):
        for key, chain_idx in chains.items():
            coxa_pos = root_rest + R_root @ (rest_J[chain_idx[0]] - root_rest)
            rest_dirs = torch.stack(
                [rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]] for i in range(5)]
            )
            rows = [chain_idx[i] - 1 for i in range(5)]
            theta_gt = torch.as_tensor(jr_gt_aa[s, rows], dtype=torch.float32)

            positions = fk_all(R_root, coxa_pos, rest_dirs, theta_gt)
            target_waypoint = positions[waypoint_seg_idx].detach()
            target_tip = positions[4].detach()

            min_l = min_limits_all[rows]
            max_l = max_limits_all[rows]
            constraints = [(waypoint_seg_idx, target_waypoint), (4, target_tip)]
            theta_ik = solve_chain_ik_multi(
                R_root, coxa_pos, rest_J, chain_idx, constraints, min_l, max_l, n_iters=300
            )

            reached = fk_all(R_root, coxa_pos, rest_dirs, theta_ik)
            res = float((reached[waypoint_seg_idx] - target_waypoint).norm()) + \
                  float((reached[4] - target_tip).norm())
            pos_residuals.append(res)

            for i, seg in enumerate(("co", "tr", "fe", "ti", "ta")):
                joint_deg_errs[seg].append(geodesic_deg(theta_gt[i], theta_ik[i]))

    pos_residuals = np.array(pos_residuals)
    print("=" * 78)
    print(f"ORACLE CHECK: solve_chain_ik_multi (tip + waypoint at seg_idx={waypoint_seg_idx}, 'ti')")
    print("=" * 78)
    print(f"Position residual (sum of both constraints' errors): mean={pos_residuals.mean():.6f} "
          f"max={pos_residuals.max():.6f}")
    print()
    print("Per-joint angular deviation from TRUE GT pose (deg) -- compare to single-tip's "
          "~19-24 deg across the board:")
    all_e = []
    for seg in ("co", "tr", "fe", "ti", "ta"):
        e = np.array(joint_deg_errs[seg])
        all_e.extend(e.tolist())
        print(f"  {seg:4s} mean={e.mean():6.2f}  median={np.median(e):6.2f}  "
              f"p95={np.percentile(e, 95):6.2f}  max={e.max():6.2f}")
    all_e = np.array(all_e)
    print(f"\nOverall mean: {all_e.mean():.2f} deg  (single-tip oracle was 21.15 deg)")


if __name__ == "__main__":
    main()
