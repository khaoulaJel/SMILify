"""Pose-error sweep (2026-08-20 follow-up to the Anatomical Initialization Ceiling Test).

Starts from the exact ground-truth leg joint_rot (`diagnostics/moonshot/synth_clean/
ground_truth.npz`) and injects a FIXED-MAGNITUDE random rotation of `level` degrees onto each
of the 36 leg joints (co/tr/fe/ti/ta/pt x 6 legs), leaving every other joint (head/gaster/
wing/antenna/mandible) at the exact ground-truth value. Only legs are perturbed because that is
what the ceiling test's own results (RESULTS.md) showed diverging: gt_init recovers cleanly,
cheap_anatomical's leg pose error is what correlates with its regression -- this isolates degree
of LEG pose error as the single swept variable, with everything else held at the GT ceiling, so
the resulting basin curve answers "how much leg pose error can the D1 pipeline tolerate" without
conflating it with antenna/head pose error (which the ceiling test never actually varied either).

NOISE MODEL (frozen -- do not tune per level):
  For specimen i, leg joint row j, level L (degrees): draw a uniform-random unit axis a (via
  `rng.normal(size=3)`, normalised -- the standard method for a uniform point on S^2) and set
  the LOCAL perturbing rotation R_delta = axis_angle_to_matrix(a * radians(L)) -- i.e. the
  perturbation's rotation ANGLE is exactly L degrees, not drawn from a distribution, only its
  AXIS is random. Composed as R_noisy = R_delta @ R_gt (delta applied in the joint's own local
  frame, left-multiplied), matching how `simple_leg_heuristic`/`geom_leg_init` already compose
  local rotations in this codebase. A single `np.random.default_rng(seed=1000*level + i)` is
  used per (level, specimen) so the whole sweep is reproducible from this file alone; the SAME
  seed formula is reused for every level so per-specimen noise draws are independent across
  levels (not nested/correlated), and the axis draw order is joint-major (co,tr,fe,ti,ta,pt per
  leg, legs in `leg_chains()` dict order) so re-running this script reproduces byte-identical
  output.

Level 0 is not written (it is exactly the existing `SYN_clean_pose25_gtinit_w5` run -- reused,
not re-fit, same as the ceiling test's Arm C).
"""

import argparse
import os
import sys

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fitter_3d.geom_leg_init import leg_chains  # noqa: E402 (pure, not the abandoned IK)

LEVELS = [5, 10, 15, 20, 25, 30]


def perturb(jr_gt, jnames, level, seed_base):
    """jr_gt: (N,54,3) axis-angle. Returns perturbed copy, legs only."""
    chains = leg_chains(jnames)
    leg_rows = sorted({idx - 1 for chain in chains.values() for idx in chain})  # joint_rot rows
    out = jr_gt.copy()
    N = jr_gt.shape[0]
    for i in range(N):
        rng = np.random.default_rng(seed_base + i)
        for row in leg_rows:
            axis = rng.normal(size=3)
            axis = axis / np.linalg.norm(axis)
            delta_aa = torch.tensor(axis * np.radians(level), dtype=torch.float32)
            R_delta = axis_angle_to_matrix(delta_aa.unsqueeze(0))[0]
            R_gt = axis_angle_to_matrix(torch.tensor(jr_gt[i, row], dtype=torch.float32).unsqueeze(0))[0]
            R_noisy = R_delta @ R_gt
            out[i, row] = matrix_to_axis_angle(R_noisy.unsqueeze(0))[0].numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_clean")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_ceiling_20260820/pose_noise")
    ap.add_argument("--model", default="3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
    args = ap.parse_args()

    import pickle

    with open(args.model, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    jnames = list(dd["J_names"])

    gt = np.load(os.path.join(args.corpus, "ground_truth.npz"), allow_pickle=True)
    jr_gt = gt["joint_rot"]
    names = gt["names"]

    os.makedirs(args.out_dir, exist_ok=True)
    for level in LEVELS:
        jr_noisy = perturb(jr_gt, jnames, level, seed_base=1000 * level)
        p = os.path.join(args.out_dir, f"noise{level}.npz")
        np.savez(p, joint_rot=jr_noisy.astype(np.float32), names=names)
        # sanity: confirm the injected error is exactly `level` degrees per leg joint
        import scipy.spatial.transform as st

        leg_rows = sorted({idx - 1 for chain in leg_chains(jnames).values() for idx in chain})
        Ra = st.Rotation.from_rotvec(jr_noisy[:, leg_rows].reshape(-1, 3))
        Rb = st.Rotation.from_rotvec(jr_gt[:, leg_rows].reshape(-1, 3))
        err = (Ra.inv() * Rb).magnitude() * 180 / np.pi
        print(f"[pose_noise] level={level}: wrote {p}, injected leg-joint error mean={err.mean():.3f} "
              f"deg (std={err.std():.3f}, expect mean==level exactly since axis-only randomness)")


if __name__ == "__main__":
    main()
