#!/usr/bin/env python3
"""Realistic (scan-derived) PCA-coherent init for synth_clean, using
`init_joint_rot_for_specimen_pca_coherent` -- the 4th, genuinely distinct GT-free candidate
added to the selection-experiment pool (2026-08-25). Same H0 reuse convention as
`generate_ik_init.py` (H0 only fits 'body', independent of leg-pose init choice).
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
    out_pca = np.zeros((len(names), n_pose, 3), dtype=np.float32)
    errs = []

    for i, name in enumerate(names):
        h0_i = h0_by_name[name]
        global_rot_aa = torch.as_tensor(h0["global_rot"][h0_i], dtype=torch.float32)
        trans = torch.as_tensor(h0["trans"][h0_i], dtype=torch.float32)

        verts, _, _ = load_obj(os.path.join(mesh_dir, f"{name}.obj"), load_textures=False)
        target_pts = verts.to(torch.float32)

        jr_pca = gli.init_joint_rot_for_specimen_pca_coherent(
            global_rot_aa, trans, rest_J, jnames, target_pts
        )
        out_pca[i] = jr_pca.numpy()

        gt_legs = jr_gt_aa[i][leg_rows_all]
        e = geodesic_deg_batch(jr_pca[leg_rows_all].numpy(), gt_legs)
        errs.append(float(e.mean()))
        print(f"[{name}] pca-coherent={errs[-1]:.2f} deg")

    np.savez(os.path.join(out_dir, "pca_coherent_init.npz"), joint_rot=out_pca, names=np.array(names))
    errs = np.array(errs)
    print()
    print(f"pca-coherent: mean={errs.mean():.2f} deg  median={np.median(errs):.2f} deg")
    print(f"wrote {out_dir}/pca_coherent_init.npz")


if __name__ == "__main__":
    main()
