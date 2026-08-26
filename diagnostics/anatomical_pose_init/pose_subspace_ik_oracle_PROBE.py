#!/usr/bin/env python3
"""Oracle validation for a WHOLE-CHAIN PCA pose-subspace IK candidate (2026-08-25), built to
directly address the structural limitation `coherent_multi_estimator_oracle_PROBE.py` proved:
true leg poses distribute ~50 deg of bend across tr/fe/ti/ta as well as co (coxa_only_gap
mean=50.03 deg), so NO coxa-only estimator -- however good its direction estimate -- can recover
the true per-joint decomposition. PCA_coherent and Cluster_coherent both then lost to zero-init
in aggregate leg_acc (-0.064, -0.041) despite being internally coherent by construction, which is
consistent with that same shared limitation, not a third independent failure to explain away.

MECHANISM, why this is a genuinely different candidate rather than a 4th coxa-direction guess:
instead of restricting the candidate to "coxa rotates freely, tr/fe/ti/ta pinned at rest" (a
1-joint family) or "all 5 joints rotate freely, redundancy resolved by a generic minimum-norm
regularizer toward rest" (`solve_chain_ik`'s 15-D unconstrained family, oracle-measured ~19-21 deg
irreducible gap even given the TRUE tip/waypoint position -- see `ik_init_constraint_sweep_PROBE.py`),
this restricts the candidate to a LEARNED LOW-DIMENSIONAL SUBSPACE of realistic whole-chain poses:
theta = mean_pose + z @ basis, z in R^n_modes (n_modes << 15), where mean_pose/basis are fit via
PCA on the corpus's OWN true [co,tr,fe,ti,ta] joint-rotation vectors (pooled across all 6 legs,
LEAVE-ONE-OUT over specimens so the held-out test specimen's own ground truth never touches its
own training fold -- this module DOES consume ground-truth pose data, unlike `geom_leg_init.py`
which explicitly forbids it; this is a trained-prior candidate, methodologically the same
leave-one-out-generalization discipline `probe_19` already uses for a different question).

This directly targets the "no coxa-only estimator can help" limitation: real leg bends are not
arbitrary combinations of 5 independent joint rotations (that's what made raw IK's redundancy
irreducible), they are drawn from a much smaller family of anatomically-realistic coupled poses.
A subspace constraint replaces "any pose, prefer small" with "only poses this leg-articulation
actually exhibits, learned directly from the corpus" -- a strictly stronger and more specific
prior than either the coherent family's "zero bend downstream" or raw IK's generic L2 regularizer.

Oracle-only: uses TRUE tip/waypoint POSITIONS as constraints (best possible input), sweeping
n_modes, to isolate "does the subspace CONSTRAINT help" from "can we estimate tip/waypoint from a
noisy scan" (deferred to a later, scan-derived script, exactly the established discipline).
"""
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from geom_leg_init import leg_chains, rest_joint_positions  # noqa: E402


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


def build_pose_subspace(jr_gt_aa, chains, held_out_idx, n_modes):
    """Leave-one-out PCA basis over [co,tr,fe,ti,ta] (15-D) joint-rotation vectors, pooled across
    ALL 6 legs of every specimen EXCEPT held_out_idx (11 specimens x 6 legs = 66 training rows).
    Pooling across leg POSITIONS (l1/l2/l3, left/right), not fitting one basis per leg identity,
    is a deliberate choice forced by data scarcity (66 pooled rows vs 11 rows if kept per-position
    -- too few for even a 5-mode basis without severe overfitting); this is a known limitation
    (front/mid/hind legs likely have systematically different characteristic poses that a single
    pooled basis will blend together), not a hidden one.

    Returns (mean_pose (15,) np.float32, basis (n_modes,15) np.float32, explained_var_ratio (n_modes,)).
    """
    N = jr_gt_aa.shape[0]
    rows = []
    for s in range(N):
        if s == held_out_idx:
            continue
        for key, chain_idx in chains.items():
            leg_rows = [chain_idx[i] - 1 for i in range(5)]
            rows.append(jr_gt_aa[s, leg_rows].reshape(-1))  # (15,)
    X = np.stack(rows, axis=0).astype(np.float64)  # (66,15)
    mean_pose = X.mean(axis=0)
    Xc = X - mean_pose
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = (S**2).sum()
    basis = Vt[:n_modes]
    evr = (S[:n_modes] ** 2) / total_var
    return mean_pose.astype(np.float32), basis.astype(np.float32), evr


def solve_chain_ik_pca(global_rot_mat, coxa_pos, rest_J, chain_idx, constraints, mean_pose, basis,
                        n_iters=300, lr=0.08, w_reg=0.001):
    """Same FK-based IK as `solve_chain_ik_multi`, but theta is constrained to the learned
    subspace: theta = (mean_pose + z @ basis).reshape(5,3), optimizing z in R^n_modes instead of
    the raw 15-D theta. `w_reg` is a mild L2 pull on z toward 0 (i.e. toward mean_pose) -- much
    weaker role than `solve_chain_ik`'s w_reg, since the subspace itself is now the dominant
    constraint, not the regularizer.
    """
    device = coxa_pos.device
    rest_dirs = torch.stack(
        [rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]] for i in range(5)]
    ).to(device)

    mean_pose_t = torch.as_tensor(mean_pose, dtype=torch.float32, device=device)
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)  # (n_modes,15)
    n_modes = basis_t.shape[0]

    z = torch.zeros(n_modes, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    constraints = [(seg_idx, tgt.detach().to(device)) for seg_idx, tgt in constraints]

    for _ in range(n_iters):
        opt.zero_grad()
        theta = (mean_pose_t + z @ basis_t).reshape(5, 3)
        R_local = axis_angle_to_matrix(theta)
        R_g = global_rot_mat
        pos = coxa_pos
        pos_loss = 0.0
        for i in range(5):
            R_g = R_g @ R_local[i]
            pos = pos + R_g @ rest_dirs[i]
            for seg_idx, tgt in constraints:
                if seg_idx == i:
                    pos_loss = pos_loss + ((pos - tgt) ** 2).sum()
        reg_loss = (z**2).mean()
        loss = pos_loss + w_reg * reg_loss
        loss.backward()
        opt.step()

    theta_final = (mean_pose_t + z.detach() @ basis_t).reshape(5, 3)
    return theta_final


def main():
    model_path = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
    corpus_path = "diagnostics/moonshot/synth_clean/ground_truth.npz"

    dd = load_model(model_path)
    jnames = list(dd["J_names"])
    chains = leg_chains(jnames)
    rest_J = rest_joint_positions(dd["J_regressor"], dd["v_template"])

    gt = np.load(corpus_path, allow_pickle=True)
    jr_gt_aa = gt["joint_rot"]
    N = jr_gt_aa.shape[0]

    R_root = torch.eye(3)
    root_rest = rest_J[0]

    constraint_defs = {"tip_only (k=1)": [4], "tip+waypoint (k=2)": [2, 4]}
    mode_counts = [2, 3, 5, 8, 15]

    print("=" * 90)
    print("PCA POSE-SUBSPACE ORACLE (leave-one-out, TRUE tip/waypoint positions, N=%d specimens)" % N)
    print("=" * 90)
    print("Reference (raw box-constrained IK oracle, from ik_init_constraint_sweep_PROBE.py):")
    print("  tip_only (k=1): mean=21.15 deg   tip+waypoint (k=2): mean=19.34 deg")
    print()

    for label, seg_idxs in constraint_defs.items():
        print(f"--- {label} ---")
        for n_modes in mode_counts:
            all_e = []
            all_evr_last = None
            for s in range(N):
                mean_pose, basis, evr = build_pose_subspace(jr_gt_aa, chains, held_out_idx=s, n_modes=n_modes)
                all_evr_last = evr
                for key, chain_idx in chains.items():
                    coxa_pos = root_rest + R_root @ (rest_J[chain_idx[0]] - root_rest)
                    rest_dirs = torch.stack(
                        [rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]] for i in range(5)]
                    )
                    rows = [chain_idx[i] - 1 for i in range(5)]
                    theta_gt = torch.as_tensor(jr_gt_aa[s, rows], dtype=torch.float32)
                    positions = fk_all(R_root, coxa_pos, rest_dirs, theta_gt)
                    constraints = [(si, positions[si].detach()) for si in seg_idxs]

                    theta_pca = solve_chain_ik_pca(
                        R_root, coxa_pos, rest_J, chain_idx, constraints, mean_pose, basis, n_iters=300
                    )
                    for i in range(5):
                        all_e.append(geodesic_deg(theta_gt[i], theta_pca[i]))
            all_e = np.array(all_e)
            evr_str = ",".join(f"{v:.2f}" for v in all_evr_last)
            print(f"  n_modes={n_modes:2d}: mean={all_e.mean():6.2f} deg  median={np.median(all_e):6.2f}  "
                  f"p95={np.percentile(all_e,95):6.2f}  (last-fold explained_var_ratio=[{evr_str}])")
        print()


if __name__ == "__main__":
    main()
