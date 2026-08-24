#!/usr/bin/env python3
"""
Part-based learned leg-pose initializer for SMILify.
2026-08-20, closes out the raw-geometry-retrieval branch (32.0 deg,
statistically == random pairing at 32.7 deg +/- 2.5 deg -- see
all_pairs_diagnostic.py).

Design, and WHY, straight from the retrieval post-mortem + literature
(PTF, CVPR'21; ArtEq, CVPR'23; SMAL/PASyn/Animal3D for small-data animal
pose):

  1. Canonicalize orientation from BODY-CORE vertices only (thorax/head/
     gaster), never from the full point cloud. The retrieval failure was
     caused exactly by canonicalizing with legs included -- leg pose was
     leaking into the "reference frame" that was supposed to be pose-
     invariant. Body-core is rigid across specimens, so it gives a stable
     frame with no circularity.
  2. PART-BASED input, not one global descriptor. Each point carries its
     (canonical-frame) xyz + a one-hot leg-chain-id feature (from the same
     skinning weights you already load for leg_chains). This lets the
     network localize "which leg, which segment" instead of pooling
     everything into a single histogram that the body dominates by point
     count (PTF's and ArtEq's core argument for part-based regression).
  3. 6D continuous rotation representation (Zhou et al. CVPR'19) per leg
     joint -> Gram-Schmidt -> rotation matrix, trained with the SAME
     geodesic loss used in your eval metric, so train loss and reported
     error are the same quantity.
  4. Train on a LARGE synthetic corpus from your existing generator, not
     the 12-specimen synth_clean set. 12 specimens is not just bad for
     retrieval (see all_pairs_diagnostic.py) -- it's far too small to
     train any regressor's weights, let alone validate it. Point this
     script at a bigger ground_truth.npz (same schema: verts, joint_rot,
     names) with broad, non-clustered leg-pose sampling. Reserve the
     original 12 as a held-out test set only (see --held_out_corpus).

Repo-specific adapter
----------------------
Exactly ONE function needs filling in: `body_core_and_leg_masks`, which
must return, from your model dict `dd`, a per-vertex integer array
(V,) where 0 = body-core, 1..K = leg-chain id (matching the ordering of
leg_chains(...)'s chains). A default implementation is provided assuming
a standard SMAL/SMPL-style `dd["weights"]` (V, n_joints) skinning-weight
matrix -- if your OmniAnt pkl uses a different key/shape, edit the body
marked `# >>> ADAPT`; everything else is complete and does not need
editing.

Usage
-----
python train_leg_pose_regressor.py \
    --train_corpus diagnostics/moonshot/synth_large/ground_truth.npz \
    --held_out_corpus diagnostics/moonshot/synth_clean/ground_truth.npz \
    --model 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl \
    --out_dir diagnostics/anatomical_pose_init/learned_init_20260820 \
    --epochs 200 --batch_size 16 --n_points 2048
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fitter_3d.geom_leg_init import leg_chains  # noqa: E402


# ---------------------------------------------------------------------------
# Model-dict helpers
# ---------------------------------------------------------------------------

def load_model_dict(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def leg_rows_from_names(jnames):
    chains = leg_chains(jnames)
    return chains, sorted({idx - 1 for chain in chains.values() for idx in chain})


def body_core_and_leg_masks(dd, jnames, chains):
    """Return (V,) int array: 0 = body-core vertex, k = 1-indexed leg-chain
    id for the k-th chain in `chains` (dict order).

    # >>> ADAPT if your pkl's skinning-weight key/shape differs.
    Default assumes dd["weights"] is (V, n_joints) skinning weights, the
    standard SMAL/SMPL convention, and assigns each vertex to its
    highest-weight joint, then buckets that joint into a leg chain (or
    body-core if it isn't in any chain).
    """
    weights = np.asarray(dd["weights"])  # (V, n_joints)
    nearest_joint = weights.argmax(axis=1)  # (V,)

    joint_to_chain = {}
    for chain_idx, (chain_name, joint_idxs) in enumerate(chains.items(), start=1):
        for j in joint_idxs:
            joint_to_chain[j - 1] = chain_idx  # chains store 1-indexed joints

    labels = np.zeros(len(nearest_joint), dtype=np.int64)
    for v, j in enumerate(nearest_joint):
        labels[v] = joint_to_chain.get(int(j), 0)
    return labels, len(chains)


# ---------------------------------------------------------------------------
# Canonicalization: body-core PCA frame only
# ---------------------------------------------------------------------------

def body_core_canonicalize(points: np.ndarray, vertex_labels: np.ndarray):
    """Compute a PCA frame from body-core (label==0) points only, then
    apply it to ALL points. Sign-resolved via 3rd-moment skew computed on
    body-core points only (not contaminated by leg pose)."""
    core = points[vertex_labels == 0]
    if len(core) < 10:
        raise ValueError("Too few body-core vertices found; check body_core_and_leg_masks.")
    mu = core.mean(axis=0, keepdims=True)
    core_c = core - mu
    _, _, vt = np.linalg.svd(core_c, full_matrices=False)
    axes = vt  # (3,3)

    core_proj = core_c @ axes.T
    skew = np.mean(core_proj ** 3, axis=0)
    signs = np.where(skew < 0, -1.0, 1.0)
    axes = axes * signs[:, None]

    all_proj = (points - mu) @ axes.T
    return all_proj


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LegPoseDataset(Dataset):
    def __init__(self, npz_path, vertex_labels, leg_rows, n_chains, n_points=2048):
        gt = np.load(npz_path, allow_pickle=True)
        self.verts = gt["verts"]        # (N, V, 3)
        self.joint_rot = gt["joint_rot_aa"]  # (N, 54, 3)
        self.names = [str(n) for n in gt["names"]]
        self.vertex_labels = vertex_labels
        self.leg_rows = leg_rows
        self.n_chains = n_chains
        self.n_points = n_points

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        pts_raw = self.verts[idx].astype(np.float32)
        pts_canon = body_core_canonicalize(pts_raw, self.vertex_labels).astype(np.float32)

        # random subsample for compute; keep leg points over-represented
        # since they're the minority class and the actual signal
        leg_idx = np.where(self.vertex_labels > 0)[0]
        core_idx = np.where(self.vertex_labels == 0)[0]
        n_leg = min(len(leg_idx), self.n_points // 2)
        n_core = self.n_points - n_leg
        sel_leg = np.random.choice(leg_idx, n_leg, replace=False)
        sel_core = np.random.choice(core_idx, min(n_core, len(core_idx)),
                                     replace=len(core_idx) < n_core)
        sel = np.concatenate([sel_leg, sel_core])

        xyz = pts_canon[sel]  # (n_points, 3)
        onehot = np.eye(self.n_chains + 1, dtype=np.float32)[self.vertex_labels[sel]]
        feat = np.concatenate([xyz, onehot], axis=1)  # (n_points, 3 + n_chains+1)

        jr_gt = self.joint_rot[idx][self.leg_rows].astype(np.float32)  # (L, 3)
        return torch.from_numpy(feat), torch.from_numpy(jr_gt)


# ---------------------------------------------------------------------------
# Model: shared-MLP PointNet + per-leg-joint 6D rotation heads
# ---------------------------------------------------------------------------

class PointNetBackbone(nn.Module):
    def __init__(self, in_dim, feat_dim=256):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, feat_dim, 1), nn.BatchNorm1d(feat_dim), nn.ReLU(),
        )

    def forward(self, x):  # x: (B, N, in_dim)
        x = x.transpose(1, 2)          # (B, in_dim, N)
        point_feat = self.mlp1(x)      # (B, feat_dim, N)
        global_feat = point_feat.max(dim=2)[0]  # (B, feat_dim)
        return global_feat


class LegPoseRegressor(nn.Module):
    def __init__(self, in_dim, n_leg_joints, feat_dim=256):
        super().__init__()
        self.backbone = PointNetBackbone(in_dim, feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_leg_joints * 6),
        )
        self.n_leg_joints = n_leg_joints

    def forward(self, x):
        g = self.backbone(x)
        out = self.head(g)
        return out.view(-1, self.n_leg_joints, 6)


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. CVPR'19 Gram-Schmidt 6D -> rotation matrix. d6: (..., 6)."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)  # (..., 3, 3)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """aa: (..., 3) axis-angle -> (..., 3, 3) rotation matrix (Rodrigues)."""
    theta = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(*aa.shape[:-1], 3, 3)
    theta = theta.unsqueeze(-1)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def geodesic_loss_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Mean geodesic rotation error in degrees, batched over (..., 3, 3)."""
    R_rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    return theta.mean() * 180.0 / np.pi


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total_err, n_batches = 0.0, 0
    for feat, jr_gt in loader:
        feat, jr_gt = feat.to(device), jr_gt.to(device)
        R_gt = axis_angle_to_matrix(jr_gt)  # (B, L, 3, 3)

        with torch.set_grad_enabled(train):
            d6 = model(feat)                # (B, L, 6)
            R_pred = rot6d_to_matrix(d6)     # (B, L, 3, 3)
            err_deg = geodesic_loss_deg(R_pred, R_gt)

            if train:
                optimizer.zero_grad()
                err_deg.backward()
                optimizer.step()

        total_err += err_deg.item()
        n_batches += 1
    return total_err / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_corpus", required=True,
                     help="Large synthetic ground_truth.npz for training/val split.")
    ap.add_argument("--held_out_corpus", default="diagnostics/moonshot/synth_clean/ground_truth.npz",
                     help="Original 12-specimen corpus, used ONLY as final held-out test.")
    ap.add_argument("--model", default="3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/learned_init_20260820")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    dd = load_model_dict(args.model)
    jnames = list(dd["J_names"])
    chains, leg_rows = leg_rows_from_names(jnames)
    vertex_labels, n_chains = body_core_and_leg_masks(dd, jnames, chains)
    print(f"[train] {len(leg_rows)} leg-joint rows, {n_chains} leg chains, "
          f"{(vertex_labels == 0).sum()} body-core verts / {(vertex_labels > 0).sum()} leg verts")

    full_ds = LegPoseDataset(args.train_corpus, vertex_labels, leg_rows, n_chains, args.n_points)
    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    print(f"[train] {n_train} train / {n_val} val specimens (from {args.train_corpus})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = 3 + (n_chains + 1)
    model = LegPoseRegressor(in_dim, len(leg_rows)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    ckpt_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(args.epochs):
        train_err = run_epoch(model, train_loader, optimizer, device, train=True)
        val_err = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step()

        if val_err < best_val:
            best_val = val_err
            torch.save({
                "model_state": model.state_dict(),
                "in_dim": in_dim,
                "n_leg_joints": len(leg_rows),
                "leg_rows": leg_rows,
                "n_chains": n_chains,
                "vertex_labels": vertex_labels,
                "n_points": args.n_points,
            }, ckpt_path)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[train] epoch {epoch:4d}  train {train_err:.3f} deg  "
                  f"val {val_err:.3f} deg  (best {best_val:.3f})")

    print(f"[train] done. best val = {best_val:.3f} deg. checkpoint: {ckpt_path}")
    print(f"[train] now run infer_leg_pose_init.py against --held_out_corpus "
          f"{args.held_out_corpus} for the number directly comparable to your "
          f"23.1 / 28.5 / 32.0 deg table.")


if __name__ == "__main__":
    main()