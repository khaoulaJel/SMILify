#!/usr/bin/env python3
"""Oracle validation for the 3 coherent-candidate direction estimators (PCA / cluster-axis /
tip-direction) BEFORE any real-scan use -- same discipline as `ik_init_oracle_PROBE.py`.

Uses the TRUE joint_rot from ground_truth.npz to forward-kinematically build each leg's clean
point chain, under IDENTITY global orientation / ZERO translation -- `ground_truth.npz` stores no
separate global_rot/trans field (checked directly: keys are verts/names/betas/joint_rot/
log_beta_scales/noise/extent), and every oracle probe in this series (`ik_init_oracle_PROBE.py`
etc.) already established identity/zero as the correct convention for isolating the pose-only
question. This gives "clean" points (no H0 estimation error, no scan noise) so we can check
whether each estimator's implied COXA rotation is close to the leg's TRUE coxa rotation --
isolating "is this estimator geometrically sound given clean points" from "does scan noise/H0
error break it" (the latter is a separate, later question).

Also reports pairwise ANGULAR AGREEMENT between the 3 estimators' raw directions per leg, since
the multi-start plan's Option C wants to use cross-estimator spread as a cheap, GT-free
confidence signal -- this establishes whether that spread actually correlates with which
estimator is closest to truth, on clean data, before trusting it on noisy real data.
"""
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import geom_leg_init as gli  # noqa: E402


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def geodesic_deg(aa_a, aa_b):
    Ra = axis_angle_to_matrix(torch.as_tensor(aa_a, dtype=torch.float32).unsqueeze(0))[0]
    Rb = axis_angle_to_matrix(torch.as_tensor(aa_b, dtype=torch.float32).unsqueeze(0))[0]
    rel = Ra.T @ Rb
    aa_rel = matrix_to_axis_angle(rel.unsqueeze(0))[0]
    return float(torch.rad2deg(aa_rel.norm()))


def direction_angle_deg(a, b):
    a = a / a.norm().clamp_min(1e-8)
    b = b / b.norm().clamp_min(1e-8)
    cos = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cos)))


def fk_verts(jr_aa, R_root, rest_J, chains, jnames):
    """We don't need a full skinned mesh -- for THIS oracle we only need, per leg, a clean point
    set to feed the estimators. Use the leg's own posed joint-chain samples (rest_chain_points
    logic but with the TRUE per-joint rotations applied, not rest) as a stand-in for "clean scan
    points along that leg" -- sufficient to test whether the estimators recover the true COXA
    direction from uncorrupted points, which is exactly what they consume downstream in the real
    pipeline (assigned points from *some* point cloud along the leg). IDENTITY global orientation
    (R_root=eye(3)), matching `ik_init_oracle_PROBE.py`'s established convention."""
    root_rest = rest_J[0]
    out = {}
    for key, chain_idx in chains.items():
        R_g = R_root
        pos = root_rest + R_root @ (rest_J[chain_idx[0]] - root_rest)
        pts = [pos[None, :]]
        for i in range(5):
            row = chain_idx[i] - 1
            R_local = axis_angle_to_matrix(torch.as_tensor(jr_aa[row], dtype=torch.float32).unsqueeze(0))[0]
            R_g = R_g @ R_local
            rest_dir = rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]]
            # densify each segment with 8 interpolated points, like rest_chain_points does
            seg_end = pos + R_g @ rest_dir
            t = torch.linspace(0, 1, 10)[1:-1]
            seg_pts = pos[None, :] * (1 - t[:, None]) + seg_end[None, :] * t[:, None]
            pts.append(seg_pts)
            pos = seg_end
            pts.append(pos[None, :])
        out[key] = torch.cat(pts, dim=0)
    return out


def main():
    model_path = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/moonshot/synth_clean/ground_truth.npz"

    dd = load_model(model_path)
    jnames = list(dd["J_names"])
    chains = gli.leg_chains(jnames)
    rest_J = gli.rest_joint_positions(dd["J_regressor"], dd["v_template"])

    gt = np.load(corpus_path, allow_pickle=True)
    jr_gt_aa = gt["joint_rot"]
    names = [str(n) for n in gt["names"]]
    N = jr_gt_aa.shape[0]
    R_root = torch.eye(3)  # identity global orientation, matches ik_init_oracle_PROBE.py's convention

    estimators = {
        "pca": gli.estimate_leg_pca_direction,
        "cluster": gli.estimate_leg_cluster_axis_direction,
        "tipdir": gli.estimate_leg_tip_direction,
    }

    err_by_est = {k: [] for k in estimators}
    dir_err_by_est = {k: [] for k in estimators}  # vs TRUE OVERALL direction (coxa->true tip), not true coxa rotation
    coxa_only_gap = []  # true coxa rotation vs true overall (coxa->tip) direction -- disentangles the two questions
    pairwise_agree = {"pca_vs_cluster": [], "pca_vs_tipdir": [], "cluster_vs_tipdir": []}
    n_valid = {k: 0 for k in estimators}
    n_total = 0

    root_rest = rest_J[0]
    for s in range(N):
        jr_aa = jr_gt_aa[s]

        leg_pts_clean = fk_verts(jr_aa, R_root, rest_J, chains, jnames)
        anchors = {
            key: root_rest + R_root @ (rest_J[chain_idx[0]] - root_rest)
            for key, chain_idx in chains.items()
        }

        for key, chain_idx in chains.items():
            n_total += 1
            row_co = chain_idx[0] - 1
            true_co_aa = jr_aa[row_co]
            rest_dir0 = rest_J[chain_idx[1]] - rest_J[chain_idx[0]]

            # TRUE overall direction: coxa anchor -> true tip position (last point of the clean
            # FK chain), i.e. what a coherent (coxa-only) candidate is ACTUALLY trying to match --
            # the leg's overall pointing direction, NOT the coxa joint's own decomposed rotation.
            true_tip = leg_pts_clean[key][-1]
            true_overall_dir = true_tip - anchors[key]

            # Disentangling measurement: if we used the TRUE coxa rotation itself (not an
            # estimator) as a coherent candidate, how far off would ITS implied direction be from
            # the true overall direction? This isolates "is coxa-only fundamentally lossy here"
            # from "are these specific estimators bad."
            R_local_co_true = axis_angle_to_matrix(torch.as_tensor(true_co_aa, dtype=torch.float32).unsqueeze(0))[0]
            R_global_co_true = R_root @ R_local_co_true
            true_coxa_only_dir = R_global_co_true @ rest_dir0
            coxa_only_gap.append(direction_angle_deg(true_coxa_only_dir, true_overall_dir))

            dirs = {}
            for est_name, fn in estimators.items():
                direction, valid = fn(anchors[key], leg_pts_clean[key])
                if not valid:
                    continue
                n_valid[est_name] += 1
                R_global_co = gli._align_rotation(rest_dir0.unsqueeze(0), direction.unsqueeze(0))[0]
                R_local_co = R_root.transpose(-1, -2) @ R_global_co
                pred_co_aa = matrix_to_axis_angle(R_local_co.unsqueeze(0))[0].numpy()
                err_by_est[est_name].append(geodesic_deg(pred_co_aa, true_co_aa))
                dir_err_by_est[est_name].append(direction_angle_deg(direction, true_overall_dir))
                dirs[est_name] = direction

            if "pca" in dirs and "cluster" in dirs:
                pairwise_agree["pca_vs_cluster"].append(direction_angle_deg(dirs["pca"], dirs["cluster"]))
            if "pca" in dirs and "tipdir" in dirs:
                pairwise_agree["pca_vs_tipdir"].append(direction_angle_deg(dirs["pca"], dirs["tipdir"]))
            if "cluster" in dirs and "tipdir" in dirs:
                pairwise_agree["cluster_vs_tipdir"].append(direction_angle_deg(dirs["cluster"], dirs["tipdir"]))

    print("=" * 78)
    print("COHERENT MULTI-ESTIMATOR ORACLE (clean FK points, identity global orientation, N=%d specimens, %d legs)" % (N, n_total))
    print("=" * 78)
    cog = np.array(coxa_only_gap)
    print("Disentangling check -- TRUE coxa rotation's own implied direction vs TRUE overall "
          "(coxa->tip) direction (0 deg would mean the true pose IS a pure coxa-only rotation; "
          "large values mean tr/fe/ti/ta genuinely contribute, so no coxa-only estimator, even a "
          "perfect one, could recover the true CO ROTATION from overall shape alone):")
    print(f"  mean={cog.mean():.2f}  median={np.median(cog):.2f}  p95={np.percentile(cog,95):.2f}")
    print()
    for est_name in estimators:
        e = np.array(err_by_est[est_name])
        de = np.array(dir_err_by_est[est_name])
        print(f"{est_name:10s}: valid {n_valid[est_name]}/{n_total}  "
              f"coxa-ROTATION error vs TRUE co: mean={e.mean():.2f}  median={np.median(e):.2f}  p95={np.percentile(e,95):.2f}  |  "
              f"DIRECTION error vs TRUE overall (coxa->tip): mean={de.mean():.2f}  median={np.median(de):.2f}  p95={np.percentile(de,95):.2f}")
    print()
    print("Pairwise RAW-DIRECTION agreement (deg, lower = more agreement):")
    for k, v in pairwise_agree.items():
        v = np.array(v)
        print(f"  {k:20s}: n={len(v):4d}  mean={v.mean():.2f}  median={np.median(v):.2f}  p95={np.percentile(v,95):.2f}")


if __name__ == "__main__":
    main()
