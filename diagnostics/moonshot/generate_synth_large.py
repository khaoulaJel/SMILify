#!/usr/bin/env python3

import argparse
import os

import numpy as np
import torch

import config
from fitter_3d.trainer import SMAL3DFitter


def axis_angle_to_matrix(axis_angle):
    """
    Rodrigues conversion.

    axis_angle: (..., 3)
    returns:    (..., 3, 3)
    """
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / theta.clamp_min(1e-8)

    x, y, z = axis.unbind(dim=-1)

    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(*axis.shape[:-1], 3, 3)

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)

    cos = torch.cos(theta)[..., None]
    sin = torch.sin(theta)[..., None]

    return eye + sin * K + (1.0 - cos) * (K @ K)


def sample_batch(
    smil,
    batch_size,
    pose_scale=0.50,
    shape_scale=2.0,
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = smil.device

    # Shape variation
    betas = (
        smil.mean_betas.unsqueeze(0)
        + shape_scale
        * torch.randn(
            batch_size,
            smil.n_betas,
            device=device,
        )
    )

    # Pose variation.
    #
    # IMPORTANT:
    # We deliberately use a broader distribution than the repository
    # sampler's default pose_scale=0.25 because the purpose here is
    # to train a pose initializer and cover the basin of attraction.
    joint_rot_aa = pose_scale * torch.randn(
        batch_size,
        config.N_POSE,
        3,
        device=device,
    )

    # No global orientation variation.
    #
    # The learned initializer is supposed to receive a canonicalized
    # point cloud. Global orientation therefore should not become a
    # nuisance variable during training.
    global_rot = torch.zeros(
        batch_size,
        3,
        device=device,
    )

    # No meaningful translation variation.
    trans = torch.zeros(
        batch_size,
        3,
        device=device,
    )

    # Limb scaling if enabled by the model.
    log_beta_scales = torch.zeros(
        batch_size,
        smil.n_joints,
        3,
        device=device,
    )

    if config.ALLOW_LIMB_SCALING:
        log_beta_scales[:, 1:] = 0.25 * torch.randn(
            batch_size,
            smil.n_joints - 1,
            3,
            device=device,
        )

    # Set parameters
    smil.betas.data = betas
    smil.joint_rot.data = joint_rot_aa
    smil.global_rot.data = global_rot
    smil.trans.data = trans

    if config.ALLOW_LIMB_SCALING:
        smil.log_beta_scales.data = log_beta_scales

    # Generate meshes
    with torch.no_grad():
        verts, joints = smil.forward(return_joints=True)

    # Convert axis-angle GT rotations to matrices.
    joint_rot_matrix = axis_angle_to_matrix(joint_rot_aa)

    return (
        verts.detach().cpu().numpy(),
        joint_rot_matrix.detach().cpu().numpy(),
        joint_rot_aa.detach().cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-specimens",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--pose-scale",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--shape-scale",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260820,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    parser.add_argument(
        "--shape-family",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=config.SMAL_FILE,
        help=(
            "Model .pkl this corpus should be generated from. config.SMAL_FILE is "
            "resolved at import time from the SMILIFY_SMAL_FILE env var, before "
            "argparse runs, so this flag only asserts the two agree -- it cannot "
            "switch the model itself. Set SMILIFY_SMAL_FILE=<path> before invoking "
            "python if you need a non-default model."
        ),
    )

    args = parser.parse_args()

    if os.path.abspath(args.model) != os.path.abspath(config.SMAL_FILE):
        raise RuntimeError(
            f"--model {args.model!r} does not match config.SMAL_FILE "
            f"{config.SMAL_FILE!r} (resolved from SMILIFY_SMAL_FILE at import time, "
            f"before this argument was parsed). N_BETAS and other config-derived "
            f"constants are already locked in from config.SMAL_FILE by now, so "
            f"passing --model alone cannot switch the model. Re-run with "
            f"SMILIFY_SMAL_FILE={args.model} set in the environment."
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = (
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print(f"Using device: {device}")
    print(f"Generating {args.n_specimens} specimens")
    print(f"Batch size: {args.batch_size}")
    print(f"Pose scale: {args.pose_scale}")
    print(f"Shape scale: {args.shape_scale}")

    if args.shape_family is not None:
        config.SHAPE_FAMILY = args.shape_family

    smil = SMAL3DFitter(
        batch_size=args.batch_size,
        device=device,
        shape_family=config.SHAPE_FAMILY,
    )

    all_verts = []
    all_joint_rot = []
    all_joint_rot_aa = []
    all_names = []

    remaining = args.n_specimens
    specimen_idx = 0
    batch_idx = 0

    while remaining > 0:
        n = min(args.batch_size, remaining)

        # SMAL3DFitter was initialized with args.batch_size and its
        # internal tensors therefore expect that exact batch size.
        # Always generate a full batch, then keep only the required
        # specimens from the final batch.
        verts, joint_rot, joint_rot_aa = sample_batch(
            smil,
            batch_size=args.batch_size,
            pose_scale=args.pose_scale,
            shape_scale=args.shape_scale,
            seed=args.seed + batch_idx,
        )

        all_verts.append(verts[:n])
        all_joint_rot.append(joint_rot[:n])
        all_joint_rot_aa.append(joint_rot_aa[:n])

        all_names.extend(
            [f"synth_large_{i:05d}" for i in range(
                specimen_idx,
                specimen_idx + n,
            )]
        )

        specimen_idx += n
        remaining -= n
        batch_idx += 1

        print(
            f"Generated {specimen_idx}/{args.n_specimens}",
            flush=True,
        )

    verts = np.concatenate(all_verts, axis=0)
    joint_rot = np.concatenate(all_joint_rot, axis=0)
    joint_rot_aa = np.concatenate(all_joint_rot_aa, axis=0)
    names = np.asarray(all_names)

    print()
    print("Final corpus:")
    print("  verts:", verts.shape)
    print("  joint_rot:", joint_rot.shape)
    print("  joint_rot_aa:", joint_rot_aa.shape)
    print("  names:", names.shape)

    np.savez_compressed(
        args.output,
        verts=verts,
        joint_rot=joint_rot,
        joint_rot_aa=joint_rot_aa,
        names=names,
    )

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
