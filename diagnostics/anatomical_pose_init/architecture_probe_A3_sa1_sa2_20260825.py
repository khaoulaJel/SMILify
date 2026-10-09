"""A3 -- architecture probe on smil_pointnet.py's SMILPointNet2, sharpened by the DensePose split
(finding #12: DensePose = discrete part classification + continuous within-part regression, which
maps onto this investigation's own leg_acc/seg_acc split, #10-11).

PROVENANCE NOTE (probe-before-editing): the handoff assumed an "already computed" SMILPointNet2
checkpoint existed to probe. No checkpoint under this repo's `diagnostics/anatomical_pose_init/`
best_model.pt files is actually SMILPointNet2 -- those all belong to a DIFFERENT, simpler network
(`train_leg_pose_regressor.py`'s `LegPoseRegressor`, a single global-max-pool PointNet, no sa1/sa2
hierarchy at all). The actual SMILPointNet2 checkpoint lives at `checkpoints/checkpoint_epoch_10.pth`
(state_dict keys `sa1.conv_blocks.*` confirm the PointNetSetAbstractionMsg architecture; n_pose=54,
n_betas=13, output_size=511 => n_joints=55, rotation_representation='6d', include_scales=True --
all inferred from checkpoint tensor shapes and cross-checked against config.N_POSE=54 before use,
not assumed). Uses this checkpoint's ALREADY-TRAINED weights -- genuinely "no new training", per
the A3 spec, just not the specific file path implied.

WHAT sa1/sa2's OWN CENTROID SELECTION GUARANTEES (checked directly in pointnet2_utils.py, not
assumed): PointNetSetAbstractionMsg's `new_xyz` centroids are produced by farthest-point-sampling
INDICES into the input point cloud, i.e. l1_xyz/l2_xyz are a literal subset of real input point
coordinates (never interpolated) -- so each region's TRUE anatomical label can be recovered
exactly via 1-NN back to the (label-tracked) input points, no approximation needed for the probe's
own ground truth (separate from, and not related to, the KNN label-generation rule A2 validated
for training targets).

Two linear probes on the FROZEN, already-trained network's sa1/sa2 features (never sa3 -- sa3's
group_all=True collapse is exactly the flaw this whole investigation's finding #1 diagnosed):
  (i) leg-identity separability (classification): does a per-region feature already separate
      points by which leg (of 6) or body they belong to? Mirrors DensePose's classification head.
  (ii) within-leg-position separability (regression): for points already known to be on a leg,
      does a per-region feature predict normalized position along the chain (co=0 .. pt=1)?
      Mirrors DensePose's regression head.
A LINEAR probe is used deliberately (not an MLP) so a positive result means the information is
already close to linearly readable in the frozen feature -- exactly what a DensePose-style added
head would exploit -- and a negative result isn't confounded by the probe's own capacity.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.ops import knn_points
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
import geom_leg_init as gli  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.pointcloud2smil.sample_smil_model import generate_random_parameters  # noqa: E402
from fitter_3d.pointcloud2smil.smil_pointnet import SMILPointNet2  # noqa: E402

# reuse A2's validated area-weighted-with-face-idx sampler (same file, imported by path)
sys.path.insert(0, os.path.dirname(__file__))
from validate_dense_label_pipeline_A2_20260825 import area_weighted_sample_with_face_idx  # noqa: E402


def load_model_dict(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def build_vertex_labels(jnames, dd):
    weights = np.asarray(dd["weights"])
    dominant_joint = weights.argmax(axis=1)
    n_verts = weights.shape[0]
    leg_label = np.array(["body"] * n_verts, dtype=object)
    chain_pos = np.full(n_verts, -1.0)  # normalized 0(co)..1(pt) along chain, -1 for non-leg
    for j, nm in enumerate(jnames):
        if not nm.startswith("l_"):
            continue
        bits = nm.split("_")
        k, seg, side = bits[1], bits[2], bits[-1]
        mask = dominant_joint == j
        leg_label[mask] = f"l{k}_{side}"
        chain_pos[mask] = gli.LEG_SEGMENTS.index(seg) / (len(gli.LEG_SEGMENTS) - 1)
    return leg_label, chain_pos


def load_smilpointnet2(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    msd = ckpt["model_state_dict"]
    n_pose = ckpt["n_pose"]
    n_betas = ckpt["n_betas"]
    assert n_pose == config.N_POSE, f"checkpoint n_pose={n_pose} != config.N_POSE={config.N_POSE}"
    model = SMILPointNet2(n_betas=n_betas, n_pose=n_pose, include_scales=True, rotation_representation="6d")
    model.set_joint_scales_size(55)  # inferred from fc3 output_size=511, cross-checked vs J_names count
    model.load_state_dict(msd)
    model.eval()
    return model


def extract_region_features(model, xyz_batch):
    """xyz_batch: (B,N,3) torch. Returns dict with l1_xyz/l1_points (B,3,512)/(B,320,512) and
    l2_xyz/l2_points (B,3,128)/(B,640,128), calling sa1/sa2 directly (never sa3)."""
    xyz = xyz_batch.transpose(2, 1)
    with torch.no_grad():
        l1_xyz, l1_points = model.sa1(xyz, None)
        l2_xyz, l2_points = model.sa2(l1_xyz, l1_points)
    return l1_xyz, l1_points, l2_xyz, l2_points


def labels_for_region(region_xyz, input_xyz, input_leg_label, input_chain_pos):
    """region_xyz: (B,3,S) torch (subset of input_xyz coords, per pointnet2_utils's FPS-by-index
    guarantee). Recovers exact source-point index via 1-NN, then reads off that point's true
    labels. Returns (leg_label (B*S,) str array, chain_pos (B*S,) float array)."""
    B, _, S = region_xyz.shape
    region_pts = region_xyz.transpose(2, 1)  # (B,S,3)
    idx = knn_points(region_pts, input_xyz, K=1).idx[..., 0].numpy()  # (B,S)
    leg_label = input_leg_label[idx.reshape(-1) + np.repeat(np.arange(B) * input_xyz.shape[1], S)]
    chain_pos = input_chain_pos[idx.reshape(-1) + np.repeat(np.arange(B) * input_xyz.shape[1], S)]
    return leg_label, chain_pos


def main():
    dd = load_model_dict(config.SMAL_FILE)
    jnames = list(dd["J_names"])
    leg_label_per_vertex, chain_pos_per_vertex = build_vertex_labels(jnames, dd)

    device = "cpu"
    model = load_smilpointnet2(os.path.join(REPO, "checkpoints", "checkpoint_epoch_10.pth"), device)

    fitter = SMAL3DFitter(batch_size=1, device=device, shape_family=-1)
    N_SPECIMENS = 24
    N_POINTS = 2048
    rng = np.random.default_rng(20260825)

    all_input_leg = []
    all_input_chainpos = []
    xyz_list = []
    for s in range(N_SPECIMENS):
        generate_random_parameters(fitter, seed=30000 + s, random_dist="uniform",
                                    shape_scale=2.0, pose_scale=0.25, trans_scale=0.01,
                                    scale_scale=0.25, global_rot_scale=0.0)
        with torch.no_grad():
            verts = fitter()[0]
        faces = fitter.faces[0]
        pts_np, face_idx = area_weighted_sample_with_face_idx(verts, faces, N_POINTS, rng)
        v0 = faces[face_idx, 0].numpy()  # representative vertex per sampled point (its face's first vertex)
        all_input_leg.append(leg_label_per_vertex[v0])
        all_input_chainpos.append(chain_pos_per_vertex[v0])
        xyz_list.append(pts_np)

    xyz_batch = torch.as_tensor(np.stack(xyz_list), dtype=torch.float32)  # (B,N,3)
    input_leg = np.stack(all_input_leg)  # (B,N) str
    input_chainpos = np.stack(all_input_chainpos)  # (B,N) float

    l1_xyz, l1_points, l2_xyz, l2_points = extract_region_features(model, xyz_batch)

    input_xyz_for_knn = xyz_batch  # (B,N,3), matches labels indexing
    results = {}
    for region_name, region_xyz, region_feat in [("sa1", l1_xyz, l1_points), ("sa2", l2_xyz, l2_points)]:
        leg_lab, chain_pos = labels_for_region(region_xyz, input_xyz_for_knn, input_leg.reshape(-1), input_chainpos.reshape(-1))
        feat = region_feat.transpose(2, 1).reshape(-1, region_feat.shape[1]).numpy()  # (B*S, C)
        results[region_name] = dict(feat=feat, leg_lab=leg_lab, chain_pos=chain_pos)

    out_dir = os.path.join(os.path.dirname(__file__), "out_A3_architecture_probe_20260825")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    report_lines = ["=" * 78, "A3 ARCHITECTURE PROBE (SMILPointNet2, checkpoints/checkpoint_epoch_10.pth, sa1/sa2 only)", "=" * 78]

    for row, region_name in enumerate(["sa1", "sa2"]):
        feat = results[region_name]["feat"]
        leg_lab = results[region_name]["leg_lab"]
        chain_pos = results[region_name]["chain_pos"]

        # --- (i) leg-identity classification ---
        Xtr, Xte, ytr, yte = train_test_split(feat, leg_lab, test_size=0.3, random_state=0, stratify=leg_lab)
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
        clf.fit(scaler.transform(Xtr), ytr)
        pred = clf.predict(scaler.transform(Xte))
        classes = sorted(set(leg_lab))
        acc = (pred == yte).mean()
        cm = confusion_matrix(yte, pred, labels=classes, normalize="true")
        report_lines.append(f"\n[{region_name}] leg-identity classification: n_train={len(ytr)} n_test={len(yte)} "
                             f"classes={classes} accuracy={acc:.4f}")

        ax = axes[row, 0]
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{region_name}: leg-identity confusion (acc={acc:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.046)

        # --- (ii) within-leg chain-position regression (leg points only) ---
        leg_mask = leg_lab != "body"
        Xl, yl = feat[leg_mask], chain_pos[leg_mask]
        Xltr, Xlte, yltr, ylte = train_test_split(Xl, yl, test_size=0.3, random_state=0)
        scaler2 = StandardScaler().fit(Xltr)
        reg = Ridge(alpha=1.0)
        reg.fit(scaler2.transform(Xltr), yltr)
        pred_pos = reg.predict(scaler2.transform(Xlte))
        r2 = r2_score(ylte, pred_pos)
        report_lines.append(f"[{region_name}] within-leg chain-position regression: n_train={len(yltr)} "
                             f"n_test={len(ylte)} R2={r2:.4f}")

        ax = axes[row, 1]
        ax.scatter(ylte, pred_pos, s=3, alpha=0.15, color="#4C72B0")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("true chain position (0=co, 1=pt)")
        ax.set_ylabel("predicted")
        ax.set_title(f"{region_name}: chain-position regression (R2={r2:.3f})")
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)

    fig.suptitle("A3: linear probes on FROZEN, already-trained SMILPointNet2 sa1/sa2 features")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "fig_A3_sa1_sa2_linear_probes.png")
    fig.savefig(out_path, dpi=150)

    report_lines.append(f"\n[A3] wrote {out_path}")
    print("\n".join(report_lines))
    with open(os.path.join(out_dir, "A3_probe_report.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
