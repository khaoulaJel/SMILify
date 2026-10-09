"""B1 (Phase 10, Variant 1) -- DensePose-style dense correspondence network for SMIL point clouds.

WHY THIS EXISTS, AND WHAT IT FIXES (see PHASE10_DESIGN_correspondence_network_20260825.md and
SESSION_SYNTHESIS_20260825.md #1, #12): `smil_pointnet.py`'s `SMILPointNet2` decodes every output
parameter from `sa3`'s single `group_all=True` pooled 1024-d vector -- discarding `sa1`/`sa2`'s
still-localized per-region features before any parameter is decoded. This network reuses the SAME
verified sa1/sa2/sa3 backbone shapes (320/640/1024 channels, matching `SMILPointNet2` exactly --
see its own docstring comments), but decodes correspondence from PER-POINT features recovered via
`PointNetFeaturePropagation` (the standard PointNet++ part-segmentation upsampling path, already
implemented in `pointnet2_utils.py`, unused until now), never from the pooled vector directly.

DensePose-style split (finding #12: DensePose = discrete part classification + continuous
within-part regression -- independently confirmed present in this exact codebase's existing,
already-trained SMILPointNet2 checkpoint by `architecture_probe_A3_sa1_sa2_20260825.py`, 0.85-0.87
leg-identity accuracy / R^2 0.71-0.84 chain-position from LINEAR probes on frozen sa1/sa2 alone):
  (i) per-point segment classification: body vs one of (n_legs * n_leg_segments) leg segments,
      cross-entropy, using the SAME leg/segment taxonomy `leg_acc`/`seg_acc` already score against
      (`geom_leg_init.leg_chains`/`LEG_SEGMENTS`, dominant-skinning-weight-joint vertex labeling,
      matching `architecture_probe_A3_sa1_sa2_20260825.py::build_vertex_labels`).
  (ii) within-segment chain-position regression (co=0 .. pt=1), masked to points whose TRUE label
      is a leg segment (not body) at train time -- the same continuous target A3 already validated
      is linearly recoverable, so B1's head is trained to do explicitly what A3 showed the frozen
      features already support implicitly.

`decode_to_template_vertex` turns a (segment, chain_pos) prediction into a single template vertex
index, for consumption by PHASE10_DESIGN's proposed fitter hook (`--correspondence_prior_from` /
`w_dense_gt`-style additive term) and by C2's per-leg rigid-alignment reconnection -- NOT wired
into either yet (out of scope for B1 itself, kept here since both need the same decode utility).
"""
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pointnet2_utils import (  # noqa: E402
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, REPO)


def load_model_dict(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def build_segment_taxonomy(jnames):
    """Same taxonomy leg_acc/seg_acc score against. Returns (class_names, name_to_id) with
    class 0 = 'body', classes 1..K = 'lK_side_seg' for every leg joint in jnames."""
    import geom_leg_init as gli

    classes = ["body"]
    for nm in jnames:
        if not nm.startswith("l_"):
            continue
        bits = nm.split("_")
        k, seg, side = bits[1], bits[2], bits[-1]
        cls = f"l{k}_{side}_{seg}"
        if cls not in classes:
            classes.append(cls)
    name_to_id = {c: i for i, c in enumerate(classes)}
    return classes, name_to_id


def build_vertex_labels(dd, class_names, name_to_id):
    """(V,) segment-class-id array + (V,) chain_pos array (co=0..pt=1, -1 for body), via
    dominant skinning-weight joint per vertex -- identical rule to
    architecture_probe_A3_sa1_sa2_20260825.py::build_vertex_labels, factored here so B1/B2 and A3
    share one implementation instead of two copies that could silently drift apart."""
    import geom_leg_init as gli

    weights = np.asarray(dd["weights"])
    dominant_joint = weights.argmax(axis=1)
    n_verts = weights.shape[0]
    jnames = list(dd["J_names"])
    seg_class_id = np.zeros(n_verts, dtype=np.int64)  # default 'body' = 0
    chain_pos = np.full(n_verts, -1.0)
    for j, nm in enumerate(jnames):
        if not nm.startswith("l_"):
            continue
        bits = nm.split("_")
        k, seg, side = bits[1], bits[2], bits[-1]
        mask = dominant_joint == j
        seg_class_id[mask] = name_to_id[f"l{k}_{side}_{seg}"]
        chain_pos[mask] = gli.LEG_SEGMENTS.index(seg) / (len(gli.LEG_SEGMENTS) - 1)
    return seg_class_id, chain_pos


class SMILCorrespondenceNet(nn.Module):
    """PointNet++ MSG backbone (sa1/sa2/sa3, channel dims identical to SMILPointNet2) + standard
    PointNet++ part-segmentation feature-propagation decoder (fp3/fp2/fp1) back to per-point
    resolution, then two per-point heads. Input (B,N,3) -> (seg_logits (B,N,n_classes),
    chain_pos (B,N) in [0,1])."""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 0,
                                              [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.fp3 = PointNetFeaturePropagation(1024 + 640, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, n_classes, 1),
        )
        self.pos_head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, N, 3). Returns seg_logits (B, N, n_classes), chain_pos (B, N)."""
        xyz = x.transpose(2, 1)  # (B,3,N)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)  # (B,128,N)

        seg_logits = self.seg_head(l0_points).transpose(2, 1)  # (B,N,n_classes)
        chain_pos = self.pos_head(l0_points).transpose(2, 1).squeeze(-1)  # (B,N)
        return seg_logits, chain_pos


def correspondence_loss(seg_logits, chain_pos_pred, seg_true, chain_pos_true, class_weight=None):
    """seg_logits (B,N,C), chain_pos_pred (B,N), seg_true (B,N) long, chain_pos_true (B,N).
    Regression loss is masked to true leg points (seg_true != 0, class 0 reserved for 'body') --
    'position along the chain' is undefined for body points, so they must not contribute gradient
    to the regression head (silently regressing body points toward the padding value 0.0/-1.0
    used upstream would corrupt the loss, not just be a no-op)."""
    B, N, C = seg_logits.shape
    cls_loss = F.cross_entropy(seg_logits.reshape(-1, C), seg_true.reshape(-1), weight=class_weight)

    leg_mask = seg_true != 0
    if leg_mask.any():
        pos_loss = F.mse_loss(chain_pos_pred[leg_mask], chain_pos_true[leg_mask])
    else:
        pos_loss = torch.zeros((), device=seg_logits.device)
    return cls_loss, pos_loss


def build_template_vertex_lookup(dd, class_names, name_to_id):
    """Per segment class (excluding body), sorted (chain_pos, vertex_idx) pairs -- used by
    decode_to_template_vertex for nearest-chain_pos lookup within a predicted class."""
    seg_class_id, chain_pos = build_vertex_labels(dd, class_names, name_to_id)
    lookup = {}
    for cls_id in range(1, len(class_names)):
        verts_in_class = np.where(seg_class_id == cls_id)[0]
        if len(verts_in_class) == 0:
            continue
        order = np.argsort(chain_pos[verts_in_class])
        lookup[cls_id] = (chain_pos[verts_in_class][order], verts_in_class[order])
    return lookup


def decode_to_template_vertex(seg_pred, chain_pos_pred, lookup):
    """seg_pred (N,) int, chain_pos_pred (N,) float in [0,1], both numpy. Returns (N,) template
    vertex indices, or -1 for points predicted as 'body' (no single corresponding leg vertex --
    callers needing a body correspondence too should use a separate nearest-body-vertex rule, out
    of scope here since body correspondence was never the problem this network targets)."""
    out = np.full(len(seg_pred), -1, dtype=np.int64)
    for cls_id in np.unique(seg_pred):
        if cls_id == 0 or cls_id not in lookup:
            continue
        mask = seg_pred == cls_id
        positions, vert_ids = lookup[cls_id]
        idx = np.searchsorted(positions, chain_pos_pred[mask])
        idx = np.clip(idx, 0, len(positions) - 1)
        # also check idx-1 in case it's a closer match than the searchsorted insertion point
        idx_prev = np.clip(idx - 1, 0, len(positions) - 1)
        use_prev = np.abs(positions[idx_prev] - chain_pos_pred[mask]) < np.abs(positions[idx] - chain_pos_pred[mask])
        idx = np.where(use_prev, idx_prev, idx)
        out[mask] = vert_ids[idx]
    return out
