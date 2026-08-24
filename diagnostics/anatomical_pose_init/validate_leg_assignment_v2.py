#!/usr/bin/env python3
"""Phase 2A + posed-chain ceiling + body-core criterion (2026-08-20 follow-up to
validate_leg_assignment.py). All diagnostic-only, against synth_clean's known template
correspondence -- nothing here is deployable on a real scan (posed-chain uses GT pose, which
never exists for real inference; it's a ceiling probe only).

Four representations compared on the same 12 specimens:
  A. coxa hard        (assign_points_to_legs_coxa)
  B. chain hard        (assign_points_to_legs_chain, rest-pose polyline)
  C. chain soft         (softmax over rest-pose chain distances, several temperatures)
  D. GT posed-chain hard (ceiling probe: same nearest-polyline rule, but the polyline uses the
                          ACTUAL per-specimen articulated pose via forward kinematics, not the
                          straight rest pose)

Also: body-core vs leg criterion, evaluated against true_labels==0, before picking any rule.
"""
import glob
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.transforms import axis_angle_to_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
from geom_leg_init import (  # noqa: E402
    leg_chains, analytic_coxa_anchors, assign_points_to_legs_coxa,
    rest_chain_points, LEG_SEGMENTS,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cheap_anatomical_init import _pca_axes, estimate_global_pose, _rotmat_to_aa  # noqa: E402
from train_leg_pose_regressor import body_core_and_leg_masks  # noqa: E402

MODEL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
CORPUS_DIR = "diagnostics/moonshot/synth_clean"
TEMPERATURES = [0.02, 0.05, 0.10, 0.20]  # in template-normalized units (template bbox diag ~2.35)


def posed_chain_points(rest_J, chains, global_rot_aa, trans, joint_rot_aa_full, n_samples_per_segment=8):
    """Same polyline construction as rest_chain_points, but using the ACTUAL per-specimen
    articulated pose (forward kinematics down each chain) instead of the straight rest pose.
    Ceiling probe only -- joint_rot_aa_full ((54,3) GT axis-angle) is never available for a
    real scan. Composition convention matches analytic_coxa_anchors' own documented one:
    child_pos = parent_pos + R_cumulative @ (child_rest - parent_rest), R_cumulative updated
    by each joint's own LOCAL rotation as the chain is walked (standard SMAL/SPML kinematic
    chain convention, consistent with SMAL3DFitter.forward's batch_global_rigid_transformation)."""
    R_root = torch.as_tensor(axis_angle_to_matrix(global_rot_aa.unsqueeze(0))[0], dtype=torch.float32)
    root_rest = rest_J[0]

    out = {}
    for key, chain in chains.items():  # chain: 6 GLOBAL 1-indexed joint ids, co..pt
        R_cum = R_root
        parent_pos = trans + root_rest
        parent_rest = root_rest
        joint_positions = []
        for j in chain:
            child_rest = rest_J[j - 1] if False else rest_J[j - 1]  # placeholder, fixed below
            joint_positions.append(None)
        # recompute properly: chain holds 1-indexed rows into (54,3); rest_J is (55,3) incl root at [0]
        R_cum = R_root
        parent_pos = trans + root_rest
        parent_rest = root_rest
        positions = []
        for seg_i, j in enumerate(chain):
            child_rest = rest_J[j]  # rest_J indexed by GLOBAL joint id (0=root, 1..54=the rest)
            child_pos = parent_pos + R_cum @ (child_rest - parent_rest)
            positions.append(child_pos)
            R_local = torch.as_tensor(
                axis_angle_to_matrix(joint_rot_aa_full[j - 1].unsqueeze(0))[0], dtype=torch.float32
            )  # joint_rot_aa_full is 0-indexed (54,3): row j-1 == this joint's own local rotation
            R_cum = R_cum @ R_local
            parent_pos = child_pos
            parent_rest = child_rest
        joint_pos = torch.stack(positions)  # (6,3)
        pts = [joint_pos]
        for i in range(len(joint_pos) - 1):
            t = torch.linspace(0, 1, n_samples_per_segment + 2)[1:-1]
            seg = joint_pos[i][None, :] * (1 - t[:, None]) + joint_pos[i + 1][None, :] * t[:, None]
            pts.append(seg)
        out[key] = torch.cat(pts, dim=0)
    return out


def macro_f1_and_recall(true_lab, pred_lab, chain_keys, n_chains):
    leg_mask = true_lab > 0
    t = true_lab[leg_mask] - 1
    p = pred_lab[leg_mask] - 1
    f1s, recs = {}, {}
    for i, k in enumerate(chain_keys):
        tp = int(((t == i) & (p == i)).sum())
        fp = int(((t != i) & (p == i)).sum())
        fn = int(((t == i) & (p != i)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        recs[k] = rec
        f1s[k] = 2 * prec * rec / max(prec + rec, 1e-8)
    return float(np.mean(list(f1s.values()))), recs


def soft_diagnostics(true_lab, d2_stack_np, chain_keys, tau):
    """d2_stack_np: (V,6) squared distances. Returns top1/2/3 acc, mean entropy, restricted to
    ground-truth leg vertices (true_lab>0)."""
    leg_mask = true_lab > 0
    t = true_lab[leg_mask] - 1
    d = np.sqrt(d2_stack_np[leg_mask])  # (Vleg,6) distances, NOT squared, for stable softmax scale
    logits = -d / tau
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    order = np.argsort(-p, axis=1)
    top1 = (order[:, 0] == t).mean()
    top2 = np.any(order[:, :2] == t[:, None], axis=1).mean()
    top3 = np.any(order[:, :3] == t[:, None], axis=1).mean()
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1).mean()
    return float(top1), float(top2), float(top3), float(entropy), p


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
    true_labels, n_chains = body_core_and_leg_masks(dd, jnames, chains)

    gt_npz = np.load(os.path.join(CORPUS_DIR, "ground_truth.npz"))
    jr_gt_all = gt_npz["joint_rot"]  # (12,54,3) axis-angle, synth_clean convention
    names_npz = [str(n) for n in gt_npz["names"]]

    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.obj")))

    rows = {"coxa": [], "chain_hard": [], "posed_hard": []}
    soft_rows = {tau: [] for tau in TEMPERATURES}
    body_core_dist_leg, body_core_dist_core = [], []

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        v, _, _ = load_obj(f, load_textures=False)
        target_np = v.numpy().astype(np.float64)
        s_idx = names_npz.index(name)
        jr_gt = torch.as_tensor(jr_gt_all[s_idx], dtype=torch.float32)  # (54,3)

        R, trans_np = estimate_global_pose(target_np, template_axes, template_centroid, root_rest_np)
        global_rot_aa = torch.as_tensor(_rotmat_to_aa(R), dtype=torch.float32)
        trans = torch.as_tensor(trans_np, dtype=torch.float32)
        target_pts = torch.as_tensor(target_np, dtype=torch.float32)

        anchors, _ = analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
        d2_coxa = torch.stack([(target_pts - anchors[k][None, :]).pow(2).sum(-1) for k in chain_keys], dim=1)
        pred_coxa = (d2_coxa.argmin(-1) + 1).numpy()

        rest_poly = rest_chain_points(rest_J, chains, global_rot_aa, trans)
        d2_rest = torch.stack([
            (target_pts[:, None, :] - rest_poly[k][None, :, :]).pow(2).sum(-1).min(dim=1).values
            for k in chain_keys
        ], dim=1)
        pred_chain = (d2_rest.argmin(-1) + 1).numpy()

        posed_poly = posed_chain_points(rest_J, chains, global_rot_aa, trans, jr_gt)
        d2_posed = torch.stack([
            (target_pts[:, None, :] - posed_poly[k][None, :, :]).pow(2).sum(-1).min(dim=1).values
            for k in chain_keys
        ], dim=1)
        pred_posed = (d2_posed.argmin(-1) + 1).numpy()

        mf1_c, rec_c = macro_f1_and_recall(true_labels, pred_coxa, chain_keys, n_chains)
        mf1_h, rec_h = macro_f1_and_recall(true_labels, pred_chain, chain_keys, n_chains)
        mf1_p, rec_p = macro_f1_and_recall(true_labels, pred_posed, chain_keys, n_chains)
        rows["coxa"].append((mf1_c, (rec_c["l2_r"] + rec_c["l2_l"]) / 2))
        rows["chain_hard"].append((mf1_h, (rec_h["l2_r"] + rec_h["l2_l"]) / 2))
        rows["posed_hard"].append((mf1_p, (rec_p["l2_r"] + rec_p["l2_l"]) / 2))

        d2_rest_np = d2_rest.numpy()
        for tau in TEMPERATURES:
            top1, top2, top3, ent, p_arr = soft_diagnostics(true_labels, d2_rest_np, chain_keys, tau)
            soft_rows[tau].append((top1, top2, top3, ent))
            if name == "synth_011":
                l2r_idx = np.where((true_labels > 0) & (true_labels - 1 == chain_keys.index("l2_r")))[0]
                l2r_idx_in_leg = np.searchsorted(np.where(true_labels > 0)[0], l2r_idx)
                if len(l2r_idx_in_leg):
                    mean_p = p_arr[l2r_idx_in_leg].mean(axis=0)
                    print(f"  [tau={tau}] synth_011 l2_r points' mean prob per leg: "
                          + ", ".join(f"{k}={mean_p[i]:.2f}" for i, k in enumerate(chain_keys)))

        # body-core criterion inspection: distance to nearest REST chain vs true label
        d_nearest_np = np.sqrt(d2_rest_np.min(axis=1))
        body_core_dist_leg.append(d_nearest_np[true_labels > 0])
        body_core_dist_core.append(d_nearest_np[true_labels == 0])

        print(f"{name:<12} macro-F1  coxa={mf1_c:.3f} chain={mf1_h:.3f} posed={mf1_p:.3f}   "
              f"ML/MR recall  coxa={rows['coxa'][-1][1]:.3f} chain={rows['chain_hard'][-1][1]:.3f} "
              f"posed={rows['posed_hard'][-1][1]:.3f}")

    print()
    print("=== pooled (mean over 12 specimens) ===")
    for k, v in rows.items():
        arr = np.array(v)
        print(f"{k:<12} macro-F1={arr[:,0].mean():.3f}  ML/MR recall={arr[:,1].mean():.3f}")
    print()
    print("=== soft (chain-distance-based) diagnostics, mean over 12 specimens ===")
    for tau in TEMPERATURES:
        arr = np.array(soft_rows[tau])
        print(f"tau={tau:<5}  top1={arr[:,0].mean():.3f}  top2={arr[:,1].mean():.3f}  "
              f"top3={arr[:,2].mean():.3f}  mean_entropy={arr[:,3].mean():.3f}")

    print()
    print("=== body-core vs leg: nearest-chain-distance distributions ===")
    leg_d = np.concatenate(body_core_dist_leg)
    core_d = np.concatenate(body_core_dist_core)
    print(f"TRUE LEG points   nearest-chain-dist percentiles: "
          f"{ {p: round(float(np.percentile(leg_d,p)),4) for p in [5,25,50,75,95]} }")
    print(f"TRUE CORE points  nearest-chain-dist percentiles: "
          f"{ {p: round(float(np.percentile(core_d,p)),4) for p in [5,25,50,75,95]} }")
    for thresh in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        core_correct = (core_d > thresh).mean()  # correctly called body-core
        leg_correct = (leg_d <= thresh).mean()    # correctly called leg
        print(f"  thresh={thresh:<5}  core_correctly_excluded={core_correct:.3f}  leg_correctly_included={leg_correct:.3f}")


if __name__ == "__main__":
    main()
