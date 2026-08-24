#!/usr/bin/env python3
"""Arms D/E/F (2026-08-20 basin-structure follow-up to the Anatomical Initialization Ceiling
Test): construct theta_0 = theta_GT + epsilon for the 12 synth_clean specimens, matched in mean
per-joint geodesic magnitude (~23 deg, same convention as leg_rot_err_deg elsewhere in this
experiment series) but with different STRUCTURE:

  D (random)              -- all 36 leg joints i.i.d. magnitude from the same pooled distribution.
  E (proximal-concentrated) -- coxa/trochanter/femur (18 joints) mean 34.5 deg, tibia/tarsus/
                                pretarsus (18 joints) mean 11.5 deg. Weighted mean 23 deg.
  F (distal-concentrated)   -- mirror of E (distal joints get the higher mean).

Perturbation construction (grounded in standard SO(3) sampling practice: sampling angle ~
Uniform and axis ~ uniform on S^2 independently biases toward small angles under the Haar
measure; the way to get an EXACT, direction-unbiased geodesic distance is to fix the angle
magnitude directly and only randomize the axis uniformly on S^2 -- see e.g. Shoemake 1992,
Kuffner 2004 "Effective Sampling and Distance Metrics for 3D Rigid Body Path Planning"):
  R_pert(axis, angle) applied as R_init = R_pert @ R_GT, axis ~ Uniform(S^2), angle = magnitude
  drawn per-joint from the group's Gamma-ish (here: clipped Gaussian) distribution.

This gives geodesic_distance(R_init, R_GT) == angle EXACTLY (bi-invariant metric on SO(3): left
multiplication by R_pert does not change the geodesic distance it induces from R_GT).

Non-leg joints (head/gaster/wing/antenna) are left at zero rotation (rest pose), matching the
convention of cheap_init.npz and learned_init.npz -- these arms isolate the leg-pose init only.

Output: <out_dir>/<condition>_init.npz with joint_rot (N,54,3) + names (N,), a drop-in for
`optimise_hierarchical.py --init_joint_rot_from`, identical convention to cheap_init.npz /
learned_init.npz.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fitter_3d"))
from geom_leg_init import leg_chains, LEG_SEGMENTS  # noqa: E402

PROXIMAL = {"co", "tr", "fe"}
DISTAL = {"ti", "ta", "pt"}


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def sample_perturbation_aa(rng, mean_deg, R_gt_np):
    """One perturbation axis-angle vector (3,) s.t. geodesic_distance(R_pert @ R_gt, R_gt) is
    drawn from a clipped-Gaussian magnitude (mean_deg, std=mean_deg/4, clipped to >=1 deg to
    avoid a degenerate identity perturbation), uniform-random axis on S^2."""
    mag_deg = float(np.clip(rng.normal(mean_deg, mean_deg / 4.0), 1.0, None))
    mag_rad = np.deg2rad(mag_deg)
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    aa_pert = axis * mag_rad
    R_pert = axis_angle_to_matrix(torch.from_numpy(aa_pert.astype(np.float32)).unsqueeze(0))[0]
    R_gt = torch.from_numpy(R_gt_np.astype(np.float32))
    R_init = R_pert @ R_gt
    aa_init = matrix_to_axis_angle(R_init.unsqueeze(0))[0].numpy()
    return aa_init, mag_deg


def build_condition(condition, jr_gt_aa, names, chains, rng):
    """jr_gt_aa: (N,54,3) axis-angle GT. Returns joint_rot (N,54,3), per_specimen_mean_err_deg."""
    N, n_joints, _ = jr_gt_aa.shape
    out = np.zeros_like(jr_gt_aa)
    per_specimen_err = []
    for s in range(N):
        errs = []
        for leg_key, chain in chains.items():
            for seg_i, joint_1idx in enumerate(chain):
                row = joint_1idx - 1
                seg = LEG_SEGMENTS[seg_i]
                if condition == "random":
                    mean_deg = 23.0
                elif condition == "proximal":
                    mean_deg = 34.5 if seg in PROXIMAL else 11.5
                elif condition == "distal":
                    mean_deg = 34.5 if seg in DISTAL else 11.5
                else:
                    raise ValueError(condition)
                R_gt = axis_angle_to_matrix(
                    torch.from_numpy(jr_gt_aa[s, row].astype(np.float32)).unsqueeze(0)
                )[0].numpy()
                aa_init, mag_deg = sample_perturbation_aa(rng, mean_deg, R_gt)
                out[s, row] = aa_init
                errs.append(mag_deg)
        per_specimen_err.append(float(np.mean(errs)))
    return out, per_specimen_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_clean/ground_truth.npz")
    ap.add_argument("--model", default="3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_ceiling_20260820_25pc")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dd = load_model(args.model)
    jnames = list(dd["J_names"])
    chains = leg_chains(jnames)

    gt = np.load(args.corpus, allow_pickle=True)
    jr_gt_aa = gt["joint_rot"]  # synth_clean: already axis-angle, (N,54,3)
    names = [str(n) for n in gt["names"]]

    os.makedirs(args.out_dir, exist_ok=True)
    for condition in ("random", "proximal", "distal"):
        rng = np.random.default_rng(args.seed)
        joint_rot, per_specimen_err = build_condition(condition, jr_gt_aa, names, chains, rng)
        mean_err = float(np.mean(per_specimen_err))
        print(f"[perturb] {condition}: per-specimen mean leg-joint error = "
              f"{mean_err:.2f} deg (band {min(per_specimen_err):.1f}-{max(per_specimen_err):.1f})")
        np.savez(
            os.path.join(args.out_dir, f"{condition}_init.npz"),
            joint_rot=joint_rot.astype(np.float32),
            names=np.array(names),
        )
    print(f"[perturb] wrote {args.out_dir}/{{random,proximal,distal}}_init.npz")


if __name__ == "__main__":
    main()
