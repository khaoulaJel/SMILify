#!/usr/bin/env python3
"""Phase 1 validation (2026-08-20, raw-scan anatomical-labeling follow-up): does
assign_points_to_legs_chain actually repair the mesothoracic-starvation defect found in
assign_points_to_legs_coxa? Ground truth = template skinning weights (per-VERTEX true leg
membership), available because synth_clean is topology-matched to the template by
construction. Both assignment methods are run "blind" (as they would be on a raw scan: no
vertex correspondence used, only the estimated rigid pose + raw point positions), then scored
against the known-correspondence ground truth.

Per the project's diagnostics rule: developed/validated ONLY on synth_clean (known
correspondence), NOT tuned against bench50 -- frozen before being applied there.
"""
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
from geom_leg_init import (  # noqa: E402
    leg_chains, analytic_coxa_anchors, assign_points_to_legs_coxa,
    assign_points_to_legs_chain, rest_chain_points,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cheap_anatomical_init import _pca_axes, estimate_global_pose, _rotmat_to_aa  # noqa: E402
from train_leg_pose_regressor import body_core_and_leg_masks  # noqa: E402

MODEL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
CORPUS_DIR = "diagnostics/moonshot/synth_clean"


def confusion_and_scores(true_lab, pred_lab, chain_keys, n_chains):
    """true_lab, pred_lab: (V,) int in {0..n_chains} (0=body-core). Only scores the 6 leg
    classes (body-core excluded -- these methods don't attempt a body-core call yet, all
    points get forced into one of 6 legs by construction)."""
    K = n_chains
    conf = np.zeros((K, K), dtype=np.int64)
    leg_mask = true_lab > 0  # ground-truth leg vertices only (fair comparison: both methods
    # force every point into one of 6 legs, so scoring body-core points would inject noise
    # neither method is designed to avoid at this stage)
    t = true_lab[leg_mask] - 1
    p = pred_lab[leg_mask] - 1
    for ti in range(K):
        for pi in range(K):
            conf[ti, pi] = int(((t == ti) & (p == pi)).sum())

    precision, recall, f1 = {}, {}, {}
    for i, k in enumerate(chain_keys):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        precision[k] = tp / max(tp + fp, 1)
        recall[k] = tp / max(tp + fn, 1)
        f1[k] = 2 * precision[k] * recall[k] / max(precision[k] + recall[k], 1e-8)
    macro_f1 = float(np.mean(list(f1.values())))
    return conf, precision, recall, f1, macro_f1


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

    true_labels, n_chains = body_core_and_leg_masks(dd, jnames, chains)  # (V,), template-vertex-indexed GT

    import glob
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.obj")))
    print(f"Validating on {len(files)} synth_clean specimens (known template correspondence)\n")

    macro_f1_coxa, macro_f1_chain = [], []
    ml_mr_recall_coxa, ml_mr_recall_chain = [], []
    for f in files:
        name = os.path.basename(f)
        v, _, _ = load_obj(f, load_textures=False)
        target_np = v.numpy().astype(np.float64)  # synth_clean already template-scale, vertex order == true_labels order

        R, trans_np = estimate_global_pose(target_np, template_axes, template_centroid, root_rest_np)
        global_rot_aa = torch.as_tensor(_rotmat_to_aa(R), dtype=torch.float32)
        trans = torch.as_tensor(trans_np, dtype=torch.float32)
        target_pts = torch.as_tensor(target_np, dtype=torch.float32)

        # -- coxa method --
        anchors, _ = analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
        d2_coxa = torch.stack([
            (target_pts - anchors[k][None, :]).pow(2).sum(-1) for k in chain_keys
        ], dim=1)
        pred_coxa = (d2_coxa.argmin(-1) + 1).numpy()  # (V,), 1-indexed to match true_labels convention

        # -- chain method --
        polylines = rest_chain_points(rest_J, chains, global_rot_aa, trans)
        d2_chain = torch.stack([
            (target_pts[:, None, :] - polylines[k][None, :, :]).pow(2).sum(-1).min(dim=1).values
            for k in chain_keys
        ], dim=1)
        pred_chain = (d2_chain.argmin(-1) + 1).numpy()

        _, _, rec_coxa, _, mf1_coxa = confusion_and_scores(true_labels, pred_coxa, chain_keys, n_chains)
        _, _, rec_chain, _, mf1_chain = confusion_and_scores(true_labels, pred_chain, chain_keys, n_chains)
        macro_f1_coxa.append(mf1_coxa)
        macro_f1_chain.append(mf1_chain)
        ml_mr_recall_coxa.append((rec_coxa["l2_r"] + rec_coxa["l2_l"]) / 2)
        ml_mr_recall_chain.append((rec_chain["l2_r"] + rec_chain["l2_l"]) / 2)
        print(f"{name:<16} macro-F1 coxa={mf1_coxa:.3f} chain={mf1_chain:.3f}   "
              f"ML/MR recall coxa={ml_mr_recall_coxa[-1]:.3f} chain={ml_mr_recall_chain[-1]:.3f}")

    print()
    print(f"=== pooled over {len(files)} specimens ===")
    print(f"macro-F1:      coxa mean={np.mean(macro_f1_coxa):.3f}   chain mean={np.mean(macro_f1_chain):.3f}")
    print(f"ML/MR recall:  coxa mean={np.mean(ml_mr_recall_coxa):.3f}   chain mean={np.mean(ml_mr_recall_chain):.3f}")

    # full confusion + per-leg P/R/F1 on the LAST specimen, printed in detail
    print(f"\n=== detailed confusion matrix, last specimen ({name}) ===")
    for method_name, pred in [("coxa", pred_coxa), ("chain", pred_chain)]:
        conf, prec, rec, f1, mf1 = confusion_and_scores(true_labels, pred, chain_keys, n_chains)
        print(f"\n--- {method_name} ---")
        print("true\\pred  " + "  ".join(f"{k:>7}" for k in chain_keys))
        for i, k in enumerate(chain_keys):
            print(f"{k:<10}" + "  ".join(f"{conf[i,j]:>7}" for j in range(n_chains)))
        for k in chain_keys:
            print(f"  {k}: precision={prec[k]:.3f} recall={rec[k]:.3f} f1={f1[k]:.3f}")
        print(f"  macro-F1={mf1:.3f}")


if __name__ == "__main__":
    main()
