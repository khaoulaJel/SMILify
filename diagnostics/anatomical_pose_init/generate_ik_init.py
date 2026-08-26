#!/usr/bin/env python3
"""Realistic (non-oracle) IK-based leg-pose init for synth_clean, using `geom_leg_init.py`'s
`init_joint_rot_for_specimen_ik` (tip-only) and `_ik2` (tip+waypoint), per the 2026-08-25
basin-map follow-up. Two oracle probes already validated the IK math and characterised its
fundamental redundancy (`ik_init_oracle_PROBE.py`, `ik_init_oracle_disambiguation_PROBE.py`,
`ik_init_oracle_multiconstraint_PROBE.py`) using the TRUE tip/waypoint positions -- this script
is the first time either method sees REALISTIC, scan-derived estimates instead.

H0's fitted global_rot/trans: this leg-pose method requires a prior whole-body rigid alignment
(the same precondition `init_joint_rot_for_specimen`'s leakage audit documents), but does not
itself need to RE-FIT it -- H0 only moves the 'body' group, never legs, so its output is
independent of which leg-pose init is used downstream. Reused directly from
`diagnostics/moonshot/runs/BASIN_random_15deg_hier/H0_body.npz` (one of the basin-map run's
already-completed hierarchical stages) rather than re-run, since re-fitting H0 here would give
numerically different but not meaningfully different global_rot/trans for the same reason.

Writes `ik_tip_init.npz` / `ik_tip_waypoint_init.npz` (fitter-compatible --init_joint_rot_from)
and prints the achieved mean leg-joint pose error vs GT for both, in the SAME convention used
throughout this experiment series (zero-init 23.09 deg, cheap_pca_heuristic 28.52 deg, learned
22.79 deg) -- a cheap sanity gate to run before spending GPU time on a full D1 fit.
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
from joint_limits import joint_limit_tensors  # noqa: E402


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def geodesic_deg_batch(aa_pred, aa_gt):
    """(K,3),(K,3) axis-angle -> (K,) geodesic error in degrees."""
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
    min_limits_all, max_limits_all = joint_limit_tensors(dd, device="cpu")
    leg_rows_all = sorted({c - 1 for chain in chains.values() for c in chain[:5]})

    gt = np.load(corpus_path, allow_pickle=True)
    jr_gt_aa = gt["joint_rot"]
    names = [str(n) for n in gt["names"]]

    h0 = np.load(h0_path, allow_pickle=True)
    h0_names = [str(n).removesuffix(".obj") for n in h0["labels"]]
    h0_by_name = {n: i for i, n in enumerate(h0_names)}

    n_pose = jr_gt_aa.shape[1]
    out_tip = np.zeros((len(names), n_pose, 3), dtype=np.float32)
    out_tipwpt = np.zeros((len(names), n_pose, 3), dtype=np.float32)
    errs_tip, errs_tipwpt = [], []

    for i, name in enumerate(names):
        h0_i = h0_by_name[name]
        global_rot_aa = torch.as_tensor(h0["global_rot"][h0_i], dtype=torch.float32)
        trans = torch.as_tensor(h0["trans"][h0_i], dtype=torch.float32)

        verts, _, _ = load_obj(os.path.join(mesh_dir, f"{name}.obj"), load_textures=False)
        target_pts = verts.to(torch.float32)

        jr_tip = gli.init_joint_rot_for_specimen_ik(
            global_rot_aa, trans, rest_J, jnames, target_pts, min_limits_all, max_limits_all
        )
        jr_tipwpt = gli.init_joint_rot_for_specimen_ik2(
            global_rot_aa, trans, rest_J, jnames, target_pts, min_limits_all, max_limits_all
        )

        out_tip[i] = jr_tip.numpy()
        out_tipwpt[i] = jr_tipwpt.numpy()

        gt_legs = jr_gt_aa[i][leg_rows_all]
        e_tip = geodesic_deg_batch(jr_tip[leg_rows_all].numpy(), gt_legs)
        e_tipwpt = geodesic_deg_batch(jr_tipwpt[leg_rows_all].numpy(), gt_legs)
        errs_tip.append(float(e_tip.mean()))
        errs_tipwpt.append(float(e_tipwpt.mean()))
        print(f"[{name}] tip-only={errs_tip[-1]:.2f} deg   tip+waypoint={errs_tipwpt[-1]:.2f} deg")

    np.savez(os.path.join(out_dir, "ik_tip_init.npz"), joint_rot=out_tip, names=np.array(names))
    np.savez(os.path.join(out_dir, "ik_tip_waypoint_init.npz"), joint_rot=out_tipwpt, names=np.array(names))

    errs_tip, errs_tipwpt = np.array(errs_tip), np.array(errs_tipwpt)
    print()
    print("=" * 78)
    print(f"tip-only:       mean={errs_tip.mean():.2f} deg  median={np.median(errs_tip):.2f} deg")
    print(f"tip+waypoint:   mean={errs_tipwpt.mean():.2f} deg  median={np.median(errs_tipwpt):.2f} deg")
    print("For reference (same corpus/recipe, RESULTS_ABC_DEF.md):")
    print("  zero-init 23.09 deg | cheap_pca_heuristic (old, failed) 28.52 deg | learned 22.79 deg")
    print(f"wrote {out_dir}/ik_tip_init.npz, {out_dir}/ik_tip_waypoint_init.npz")


if __name__ == "__main__":
    main()
