#!/usr/bin/env python3
"""Realistic (scan-derived) multi-start coherent candidates for synth_clean: cluster-axis and
tip-direction, siblings of the existing PCA-coherent candidate (`generate_pca_coherent_init.py`).
Same H0-reuse convention (H0 only fits 'body', independent of leg-pose init choice).

Also records, per specimen/leg, the pairwise angular SPREAD among all 3 estimators' raw
directions -- the oracle probe (`coherent_multi_estimator_oracle_PROBE.py`) established a ~3-5 deg
floor for this spread on CLEAN data, so spread meaningfully above that on real (noisy, H0-error-
prone) scans is a candidate GT-free confidence signal (Option C), not measured before now.
"""
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import geom_leg_init as gli  # noqa: E402


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def geodesic_deg_batch(aa_pred, aa_gt):
    Rp = axis_angle_to_matrix(torch.as_tensor(aa_pred, dtype=torch.float32))
    Rg = axis_angle_to_matrix(torch.as_tensor(aa_gt, dtype=torch.float32))
    rel = Rp.transpose(-1, -2) @ Rg
    aa_rel = matrix_to_axis_angle(rel)
    return torch.rad2deg(aa_rel.norm(dim=-1))


def direction_angle_deg(a, b):
    a = a / a.norm().clamp_min(1e-8)
    b = b / b.norm().clamp_min(1e-8)
    cos = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cos)))


def main():
    model_path = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
    corpus_path = "diagnostics/moonshot/synth_clean/ground_truth.npz"
    mesh_dir = "diagnostics/moonshot/synth_clean"
    h0_path = "diagnostics/moonshot/runs/BASIN_random_15deg_hier/H0_body.npz"
    out_dir = "diagnostics/anatomical_pose_init/out_ik_20260825"
    os.makedirs(out_dir, exist_ok=True)

    dd = load_model(model_path)
    jnames = list(dd["J_names"])
    chains = gli.leg_chains(jnames)
    rest_J = gli.rest_joint_positions(dd["J_regressor"], dd["v_template"])
    leg_rows_all = sorted({c - 1 for chain in chains.values() for c in chain[:5]})

    gt = np.load(corpus_path, allow_pickle=True)
    jr_gt_aa = gt["joint_rot"]
    names = [str(n) for n in gt["names"]]

    h0 = np.load(h0_path, allow_pickle=True)
    h0_names = [str(n).removesuffix(".obj") for n in h0["labels"]]
    h0_by_name = {n: i for i, n in enumerate(h0_names)}

    n_pose = jr_gt_aa.shape[1]
    out_cluster = np.zeros((len(names), n_pose, 3), dtype=np.float32)
    out_tipdir = np.zeros((len(names), n_pose, 3), dtype=np.float32)
    errs_cluster, errs_tipdir = [], []
    spread_rows = []  # (specimen, leg, pca_vs_cluster, pca_vs_tipdir, cluster_vs_tipdir, n_valid)

    estimators = {
        "pca": gli.estimate_leg_pca_direction,
        "cluster": gli.estimate_leg_cluster_axis_direction,
        "tipdir": gli.estimate_leg_tip_direction,
    }

    for i, name in enumerate(names):
        h0_i = h0_by_name[name]
        global_rot_aa = torch.as_tensor(h0["global_rot"][h0_i], dtype=torch.float32)
        trans = torch.as_tensor(h0["trans"][h0_i], dtype=torch.float32)

        verts, _, _ = load_obj(os.path.join(mesh_dir, f"{name}.obj"), load_textures=False)
        target_pts = verts.to(torch.float32)

        jr_cluster = gli.init_joint_rot_for_specimen_cluster_coherent(
            global_rot_aa, trans, rest_J, jnames, target_pts
        )
        jr_tipdir = gli.init_joint_rot_for_specimen_tipdir_coherent(
            global_rot_aa, trans, rest_J, jnames, target_pts
        )
        out_cluster[i] = jr_cluster.numpy()
        out_tipdir[i] = jr_tipdir.numpy()

        gt_legs = jr_gt_aa[i][leg_rows_all]
        e_c = geodesic_deg_batch(jr_cluster[leg_rows_all].numpy(), gt_legs)
        e_t = geodesic_deg_batch(jr_tipdir[leg_rows_all].numpy(), gt_legs)
        errs_cluster.append(float(e_c.mean()))
        errs_tipdir.append(float(e_t.mean()))

        # cross-estimator spread, computed directly on the same real-scan assigned points
        anchors, R_root = gli.analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
        assigned = gli.assign_points_to_legs_chain(
            target_pts, gli.rest_chain_points(rest_J, chains, global_rot_aa, trans)
        )
        for key in chains:
            dirs = {}
            for est_name, fn in estimators.items():
                direction, valid = fn(anchors[key], assigned[key])
                if valid:
                    dirs[est_name] = direction
            pc = direction_angle_deg(dirs["pca"], dirs["cluster"]) if "pca" in dirs and "cluster" in dirs else None
            pt = direction_angle_deg(dirs["pca"], dirs["tipdir"]) if "pca" in dirs and "tipdir" in dirs else None
            ct = direction_angle_deg(dirs["cluster"], dirs["tipdir"]) if "cluster" in dirs and "tipdir" in dirs else None
            spread_rows.append((name, key, pc, pt, ct, len(dirs)))

        print(f"[{name}] cluster-coherent={errs_cluster[-1]:.2f} deg  tipdir-coherent={errs_tipdir[-1]:.2f} deg")

    np.savez(os.path.join(out_dir, "cluster_coherent_init.npz"), joint_rot=out_cluster, names=np.array(names))
    np.savez(os.path.join(out_dir, "tipdir_coherent_init.npz"), joint_rot=out_tipdir, names=np.array(names))

    errs_cluster = np.array(errs_cluster)
    errs_tipdir = np.array(errs_tipdir)
    print()
    print(f"cluster-coherent: mean={errs_cluster.mean():.2f} deg  median={np.median(errs_cluster):.2f} deg")
    print(f"tipdir-coherent : mean={errs_tipdir.mean():.2f} deg  median={np.median(errs_tipdir):.2f} deg")
    print(f"wrote {out_dir}/cluster_coherent_init.npz, {out_dir}/tipdir_coherent_init.npz")

    print()
    print("Cross-estimator SPREAD on REAL scan-derived points (oracle clean-data floor was ~3-5 deg):")
    valid_rows = [r for r in spread_rows if r[5] == 3]
    print(f"  legs with all 3 estimators valid: {len(valid_rows)}/{len(spread_rows)}")
    for label, idx in [("pca_vs_cluster", 2), ("pca_vs_tipdir", 3), ("cluster_vs_tipdir", 4)]:
        vals = np.array([r[idx] for r in valid_rows if r[idx] is not None])
        print(f"  {label:20s}: n={len(vals):4d}  mean={vals.mean():.2f}  median={np.median(vals):.2f}  p95={np.percentile(vals,95):.2f}  max={vals.max():.2f}")

    spread_csv = os.path.join(out_dir, "cross_estimator_spread.csv")
    with open(spread_csv, "w") as f:
        f.write("specimen,leg,pca_vs_cluster,pca_vs_tipdir,cluster_vs_tipdir,n_valid_estimators\n")
        for row in spread_rows:
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")
    print(f"wrote {spread_csv}")


if __name__ == "__main__":
    main()
