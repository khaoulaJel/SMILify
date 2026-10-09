#!/usr/bin/env python3
"""Q3 follow-up: does the ~21 deg oracle redundancy gap (1 tip constraint) close meaningfully as
MORE oracle (true) constraint points are added along the chain, or does it plateau? Sweeps
n_constraints = 1..5 (5 = every segment boundary constrained, the theoretical ceiling for a
position-only objective on this 5-joint chain), all using TRUE positions (oracle, not
scan-derived) so this isolates the redundancy question itself from estimation noise -- exactly
the "too little information vs fundamentally broken category" question.
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

    # n_constraints=k -> use seg_idx = evenly spaced indices from {0,1,2,3,4}, always including
    # the tip (4). k=1: [4]. k=2: [2,4]. k=3: [1,2,4]... k=5: [0,1,2,3,4] (every joint's position
    # constrained -- the ceiling case for a position-only objective on this chain).
    constraint_sets = {
        1: [4],
        2: [2, 4],
        3: [1, 3, 4],
        4: [0, 1, 3, 4],
        5: [0, 1, 2, 3, 4],
    }

    print("=" * 78)
    print("IK CONSTRAINT-COUNT SWEEP (oracle/true positions throughout)")
    print("=" * 78)
    for k, seg_idxs in constraint_sets.items():
        all_e = []
        for s in range(N):
            for key, chain_idx in chains.items():
                coxa_pos = root_rest + R_root @ (rest_J[chain_idx[0]] - root_rest)
                rest_dirs = torch.stack(
                    [rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]] for i in range(5)]
                )
                rows = [chain_idx[i] - 1 for i in range(5)]
                theta_gt = torch.as_tensor(jr_gt_aa[s, rows], dtype=torch.float32)
                positions = fk_all(R_root, coxa_pos, rest_dirs, theta_gt)
                constraints = [(si, positions[si].detach()) for si in seg_idxs]

                min_l = min_limits_all[rows]
                max_l = max_limits_all[rows]
                theta_ik = solve_chain_ik_multi(
                    R_root, coxa_pos, rest_J, chain_idx, constraints, min_l, max_l, n_iters=300
                )
                for i in range(5):
                    all_e.append(geodesic_deg(theta_gt[i], theta_ik[i]))
        all_e = np.array(all_e)
        n_unknowns = 15
        n_eqns = 3 * k
        print(f"k={k} constraint(s) (seg_idx={seg_idxs}, {n_eqns} eqns vs {n_unknowns} unknowns): "
              f"mean={all_e.mean():.2f} deg  median={np.median(all_e):.2f} deg  "
              f"p95={np.percentile(all_e,95):.2f} deg")

    print()
    print("k=1 reproduces the original single-tip oracle (~21 deg); k=2 reproduces the")
    print("tip+waypoint oracle (~19 deg). k=5 is the ceiling: every joint's OWN position is")
    print("directly constrained by a true point, which is strictly more information than any")
    print("realistic point-cloud-derived estimate could reliably provide per joint.")


if __name__ == "__main__":
    main()
