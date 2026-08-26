#!/usr/bin/env python3
"""Oracle correctness check for `fitter_3d.geom_leg_init.solve_chain_ik` (2026-08-25 basin-map
follow-up), BEFORE it is ever fed realistic, error-prone scan-derived tip estimates.

Question this answers: is the IK math itself correct, and given the true tip position of a leg
(the best input this method could ever receive), how close does it get to actually reaching that
position, and how far is the specific pose it picks (among the many that also reach that same
position -- 5 joints x 3 axes = 15 unknowns, only 3 position constraints, genuinely redundant)
from the true GT pose?

This is deliberately run BEFORE the realistic (scan-derived-tip) test, matching this project's
established discipline: validate the mechanism in isolation against ground truth before it is
ever exposed to the confound of an imperfect upstream estimate (same pattern as
`estimate_leg_curve`'s own rest-pose self-consistency check).

Uses IDENTITY global orientation and ZERO translation throughout, matching every other arm in
this investigation series' explicit isolation of the pose-only question from the deferred
global-orientation hypothesis (see RESULTS_correspondence_free_representation.md).
"""
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from geom_leg_init import leg_chains, rest_joint_positions, solve_chain_ik  # noqa: E402
from joint_limits import joint_limit_tensors  # noqa: E402


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def fk_tip(global_rot_mat, coxa_pos, rest_dirs, theta):
    """Same forward-kinematics convention as `solve_chain_ik`'s internal loop -- kept in sync
    deliberately (not imported) so a bug in one implementation cannot silently cancel against
    the same bug in the other; a real convention mismatch would show up as the oracle solve
    failing to reach a target it was itself constructed with, which is the point of this check."""
    R_g = global_rot_mat
    pos = coxa_pos
    for i in range(5):
        R_g = R_g @ axis_angle_to_matrix(theta[i].unsqueeze(0))[0]
        pos = pos + R_g @ rest_dirs[i]
    return pos


def geodesic_deg(aa_a, aa_b):
    Ra = axis_angle_to_matrix(torch.as_tensor(aa_a, dtype=torch.float32).unsqueeze(0))[0]
    Rb = axis_angle_to_matrix(torch.as_tensor(aa_b, dtype=torch.float32).unsqueeze(0))[0]
    rel = Ra.T @ Rb
    from pytorch3d.transforms import matrix_to_axis_angle
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
    assert min_limits_all is not None, "model has no authored joint_limits -- oracle check needs real bounds"

    gt = np.load(corpus_path, allow_pickle=True)
    jr_gt_aa = gt["joint_rot"]  # (N,54,3)
    names = [str(n) for n in gt["names"]]
    N = jr_gt_aa.shape[0]

    R_root = torch.eye(3)  # identity global orientation, matches this investigation's convention
    root_rest = rest_J[0]

    pos_residuals = []
    joint_deg_errs = []  # per (specimen, leg, joint) GT-vs-recovered geodesic error
    limit_violations = 0
    total_joint_checks = 0

    for s in range(N):
        for key, chain_idx in chains.items():
            coxa_pos = root_rest + R_root @ (rest_J[chain_idx[0]] - root_rest)
            rest_dirs = torch.stack(
                [rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]] for i in range(5)]
            )
            rows = [chain_idx[i] - 1 for i in range(5)]
            theta_gt = torch.as_tensor(jr_gt_aa[s, rows], dtype=torch.float32)  # (5,3)

            target_tip = fk_tip(R_root, coxa_pos, rest_dirs, theta_gt).detach()

            min_l = min_limits_all[rows]
            max_l = max_limits_all[rows]
            theta_ik = solve_chain_ik(
                R_root, coxa_pos, rest_J, chain_idx, target_tip, min_l, max_l, n_iters=300
            )

            reached = fk_tip(R_root, coxa_pos, rest_dirs, theta_ik).detach()
            pos_residuals.append(float((reached - target_tip).norm()))

            for i in range(5):
                joint_deg_errs.append(
                    (key, jnames[chain_idx[i]], geodesic_deg(theta_gt[i], theta_ik[i]))
                )
                total_joint_checks += 1
                if (theta_ik[i] < min_l[i] - 1e-3).any() or (theta_ik[i] > max_l[i] + 1e-3).any():
                    limit_violations += 1

    pos_residuals = np.array(pos_residuals)
    errs_by_joint = {}
    for key, jname, e in joint_deg_errs:
        seg = jname.split("_")[2]  # co/tr/fe/ti/ta
        errs_by_joint.setdefault(seg, []).append(e)

    print("=" * 78)
    print("ORACLE CHECK: solve_chain_ik, TRUE tip position as target")
    print("=" * 78)
    print(f"specimens={N}, legs/specimen={len(chains)}, total leg-solves={len(pos_residuals)}")
    print()
    print("Position residual ||reached - target_tip|| (should be ~0 if the solver converges):")
    print(f"  mean={pos_residuals.mean():.6f}  median={np.median(pos_residuals):.6f}  "
          f"p95={np.percentile(pos_residuals, 95):.6f}  max={pos_residuals.max():.6f}  "
          f"(model units, template extent ~1.79)")
    print()
    print("Per-joint angular deviation from TRUE GT pose (deg) -- non-zero is EXPECTED here, "
          "since 15 unknowns vs 3 position constraints is genuinely redundant; this measures "
          "how large that redundancy gap actually is, per joint, not solver failure:")
    for seg in ("co", "tr", "fe", "ti", "ta"):
        e = np.array(errs_by_joint.get(seg, []))
        if len(e):
            print(f"  {seg:4s} mean={e.mean():6.2f}  median={np.median(e):6.2f}  "
                  f"p95={np.percentile(e, 95):6.2f}  max={e.max():6.2f}")
    print()
    print(f"Joint-limit violations (should be 0, hinge weight w_limit=1.0 by default): "
          f"{limit_violations}/{total_joint_checks}")


if __name__ == "__main__":
    main()
