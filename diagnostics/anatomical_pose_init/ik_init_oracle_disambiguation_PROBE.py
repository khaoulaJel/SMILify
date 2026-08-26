#!/usr/bin/env python3
"""Follow-up to `ik_init_oracle_PROBE.py`: that probe found ~20 deg mean per-joint deviation from
GT even when solve_chain_ik is given the EXACT true tip position. This script isolates WHY,
because "genuine kinematic redundancy" (many poses reach the same tip -- an unavoidable fact of
15 unknowns vs 3 constraints) and "the minimum-norm-toward-rest regularizer specifically pulling
the answer away from GT" (a hyperparameter/implementation choice, fixable) are different claims
that were conflated in the first pass, and only one of them is actually inherent to the method.

Three conditions per (specimen, leg), all against the exact same true tip target used before:

  A. theta_init=0,        w_reg=0.02   -- REPRODUCES the original oracle run (baseline for this script)
  B. theta_init=theta_GT, w_reg=0.02   -- if GT is already a local minimum of this SAME objective,
                                          seeding there should barely move. If it drifts back toward
                                          A's answer anyway, that specifically implicates the
                                          regularizer overpowering the position term.
  C. theta_init=0,        w_reg=1e-6   -- near-zero regularizer pull. If A's ~20deg gap is mostly
                                          the explicit regularizer, this should close it substantially.
                                          If it does NOT close (still far from GT), that implicates
                                          genuine multi-modality / implicit gradient-descent bias
                                          toward small-norm solutions, independent of w_reg.

Interpretation guide printed at the end: which combination of (B stays near GT) and (C closes the
gap) supports which explanation.
"""
import os
import sys

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from geom_leg_init import leg_chains, rest_joint_positions, solve_chain_ik  # noqa: E402
from joint_limits import joint_limit_tensors  # noqa: E402

import pickle


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def fk_tip(global_rot_mat, coxa_pos, rest_dirs, theta):
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

    conditions = {
        "A_zero_init_wreg0.02": dict(theta_init=None, w_reg=0.02),
        "B_gt_init_wreg0.02": dict(theta_init="gt", w_reg=0.02),
        "C_zero_init_wreg1e-6": dict(theta_init=None, w_reg=1e-6),
    }
    results = {c: [] for c in conditions}
    pos_res = {c: [] for c in conditions}

    for s in range(N):
        for key, chain_idx in chains.items():
            coxa_pos = root_rest + R_root @ (rest_J[chain_idx[0]] - root_rest)
            rest_dirs = torch.stack(
                [rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]] for i in range(5)]
            )
            rows = [chain_idx[i] - 1 for i in range(5)]
            theta_gt = torch.as_tensor(jr_gt_aa[s, rows], dtype=torch.float32)
            target_tip = fk_tip(R_root, coxa_pos, rest_dirs, theta_gt).detach()
            min_l = min_limits_all[rows]
            max_l = max_limits_all[rows]

            for cname, cfg in conditions.items():
                t_init = theta_gt if cfg["theta_init"] == "gt" else None
                theta_ik = solve_chain_ik(
                    R_root, coxa_pos, rest_J, chain_idx, target_tip, min_l, max_l,
                    n_iters=300, w_reg=cfg["w_reg"], theta_init=t_init,
                )
                reached = fk_tip(R_root, coxa_pos, rest_dirs, theta_ik).detach()
                pos_res[cname].append(float((reached - target_tip).norm()))
                for i in range(5):
                    results[cname].append(geodesic_deg(theta_gt[i], theta_ik[i]))

    print("=" * 78)
    print("DISAMBIGUATION: redundancy vs regularizer-bias-toward-rest")
    print("=" * 78)
    for cname in conditions:
        e = np.array(results[cname])
        p = np.array(pos_res[cname])
        print(f"\n{cname}:")
        print(f"  per-joint deg-from-GT: mean={e.mean():.2f} median={np.median(e):.2f} p95={np.percentile(e,95):.2f}")
        print(f"  position residual:     mean={p.mean():.6f} max={p.max():.6f}")

    a_mean = np.array(results["A_zero_init_wreg0.02"]).mean()
    b_mean = np.array(results["B_gt_init_wreg0.02"]).mean()
    c_mean = np.array(results["C_zero_init_wreg1e-6"]).mean()

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(f"A (baseline, reproduces prior oracle run): {a_mean:.2f} deg")
    print(f"B (GT-seeded, same regularizer):            {b_mean:.2f} deg")
    print(f"C (zero-seeded, negligible regularizer):     {c_mean:.2f} deg")
    print()
    if b_mean < 0.3 * a_mean:
        print("B stayed close to GT -> GT IS a local minimum of this objective. The 20deg gap in A")
        print("is gradient descent from zero finding a DIFFERENT local minimum than GT -- i.e. this")
        print("is genuine landscape multi-modality, not a regularizer artifact.")
    else:
        print("B drifted back away from GT even when seeded there -> the regularizer (or the")
        print("optimization dynamics) is actively pulling the solution away from a GT-adjacent")
        print("minimum. This implicates w_reg specifically, not pure kinematic redundancy.")
    if c_mean < 0.3 * a_mean:
        print("C closed most of the gap -> the explicit w_reg=0.02 term was doing most of the")
        print("work pulling toward rest; with it removed, zero-init converges much closer to GT.")
    else:
        print("C did NOT close the gap -> even with negligible explicit regularization, starting")
        print("from zero still lands far from GT. This points to either genuine multi-modality or")
        print("gradient descent's own implicit minimum-norm bias on this underdetermined problem,")
        print("not the explicit regularizer term.")


if __name__ == "__main__":
    main()
