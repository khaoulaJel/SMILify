#!/usr/bin/env python3
"""7-way soft body-core+leg classifier, unifying the body-core question with the same
validated chain-distance softmax used for legs (rather than an arbitrary distance threshold,
which validate_leg_assignment_v2.py showed does not cleanly separate the two classes).

Class 0 = body-core, distance = distance to root/thorax point (a single point is defensible
here since the thorax is compact and roughly centered, unlike an extended limb).
Classes 1-6 = the 6 legs, distance = nearest-point-on-rest-pose-polyline (validated in
validate_leg_assignment_v2.py: chain beats coxa, macro-F1 0.656->0.831).
"""
import glob
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
from geom_leg_init import leg_chains, rest_chain_points, axis_angle_to_matrix_np_safe  # noqa: E402

# Body-core rig chain: head -> thorax(root) -> 5 gaster segments, mirroring leg_chains' own
# structure. Same JOINT_NAMES convention as fitter_3d/part_groups.py.
BODY_CORE_JOINT_NAMES = ["b_h", "b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]


def body_core_chain_points(rest_J, jnames, root_rot_aa, trans, n_samples_per_segment=8):
    """Body-core polyline: head -> thorax(root) -> gaster segments, rigidly transformed by the
    SAME global (rotation, translation) leg chains use. Mirrors rest_chain_points but for the
    body axis instead of a leg -- a single point (root/thorax) badly under-represents an
    elongated body (validated: core_recall 0.366 with a single point), the same failure mode
    coxa-only leg assignment had."""
    name_to_idx = {n: i for i, n in enumerate(jnames)}
    R_root = axis_angle_to_matrix_np_safe(root_rot_aa)
    root_rest = rest_J[0]
    positions = []
    for nm in BODY_CORE_JOINT_NAMES:
        idx = name_to_idx[nm]  # index into jnames / rest_J directly (55 rows incl. root)
        rest_pos = rest_J[idx]
        pos = trans + root_rest + R_root @ (rest_pos - root_rest)
        positions.append(pos)
    joint_pos = torch.stack(positions)
    pts = [joint_pos]
    for i in range(len(joint_pos) - 1):
        t = torch.linspace(0, 1, n_samples_per_segment + 2)[1:-1]
        seg = joint_pos[i][None, :] * (1 - t[:, None]) + joint_pos[i + 1][None, :] * t[:, None]
        pts.append(seg)
    return torch.cat(pts, dim=0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cheap_anatomical_init import _pca_axes, estimate_global_pose, _rotmat_to_aa  # noqa: E402
from train_leg_pose_regressor import body_core_and_leg_masks  # noqa: E402

MODEL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
CORPUS_DIR = "diagnostics/moonshot/synth_clean"
TEMPERATURES = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]


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
    all_keys = ["body_core"] + chain_keys
    true_labels, n_chains = body_core_and_leg_masks(dd, jnames, chains)

    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.obj")))

    results = {tau: {"correct7": [], "core_recall": [], "leg_recall_mean": []} for tau in TEMPERATURES}
    hard7_f1 = []

    for f in files:
        name = os.path.basename(f)
        v, _, _ = load_obj(f, load_textures=False)
        target_np = v.numpy().astype(np.float64)

        R, trans_np = estimate_global_pose(target_np, template_axes, template_centroid, root_rest_np)
        global_rot_aa = torch.as_tensor(_rotmat_to_aa(R), dtype=torch.float32)
        trans = torch.as_tensor(trans_np, dtype=torch.float32)
        target_pts = torch.as_tensor(target_np, dtype=torch.float32)

        rest_poly = rest_chain_points(rest_J, chains, global_rot_aa, trans)
        d_leg = torch.stack([
            (target_pts[:, None, :] - rest_poly[k][None, :, :]).pow(2).sum(-1).min(dim=1).values.sqrt()
            for k in chain_keys
        ], dim=1)  # (P,6)
        core_poly = body_core_chain_points(rest_J, jnames, global_rot_aa, trans)
        d_core = (target_pts[:, None, :] - core_poly[None, :, :]).pow(2).sum(-1).min(dim=1).values.sqrt()[:, None]  # (P,1)
        d_all = torch.cat([d_core, d_leg], dim=1).numpy()  # (P,7), col 0 = body_core

        hard_pred = d_all.argmin(axis=1)  # 0=core, 1..6=legs (matches true_labels convention directly)
        conf = np.zeros((7, 7), dtype=np.int64)
        for ti in range(7):
            for pi in range(7):
                conf[ti, pi] = int(((true_labels == ti) & (hard_pred == ti if False else hard_pred == pi)).sum()) if False else 0
        # (simpler direct per-class P/R/F1, skip full conf matrix construction here)
        f1s = []
        for c in range(7):
            tp = int(((true_labels == c) & (hard_pred == c)).sum())
            fp = int(((true_labels != c) & (hard_pred == c)).sum())
            fn = int(((true_labels == c) & (hard_pred != c)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
        hard7_f1.append(np.mean(f1s))

        for tau in TEMPERATURES:
            logits = -d_all / tau
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            pred = p.argmax(axis=1)
            correct7 = (pred == true_labels).mean()
            core_mask = true_labels == 0
            core_recall = (pred[core_mask] == 0).mean() if core_mask.sum() else np.nan
            leg_recalls = []
            for c in range(1, 7):
                m = true_labels == c
                if m.sum():
                    leg_recalls.append((pred[m] == c).mean())
            results[tau]["correct7"].append(correct7)
            results[tau]["core_recall"].append(core_recall)
            results[tau]["leg_recall_mean"].append(np.mean(leg_recalls))

        print(f"{name:<12} hard-7way macro-F1={hard7_f1[-1]:.3f}")

    print()
    print(f"=== hard 7-way (argmin) pooled macro-F1: {np.mean(hard7_f1):.3f} ===")
    print()
    print("=== 7-way accuracy by temperature (pooled over 12 specimens) ===")
    for tau in TEMPERATURES:
        r = results[tau]
        print(f"tau={tau:<5}  overall_acc={np.mean(r['correct7']):.3f}  "
              f"core_recall={np.mean(r['core_recall']):.3f}  leg_recall_mean={np.mean(r['leg_recall_mean']):.3f}")


if __name__ == "__main__":
    main()
