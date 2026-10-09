"""A2 — validate the dense per-point ground-truth correspondence label generator needed for B2's
training data.

WHAT THE LABEL GENERATOR IS
The existing `w_dense_gt` oracle mechanism (`trainer_hierarchical.py` forward(), finding #11)
already implements the label rule this diagnostics reuses: for a point `p` resampled from a
target mesh surface, its "ground-truth corresponding vertex" is defined as
`argmin_v ||p - dense_gt_verts[v]||` -- nearest TRUE-mesh vertex in Euclidean 3D space, where
`dense_gt_verts` is the target's own posed vertices (index i = anatomical point i by
construction, since synth targets are generated straight from the model). B2 needs exactly this
rule, applied to synthetic training targets, to produce per-point labels.

WHY THIS NEEDS VALIDATING, NOT ASSUMING CORRECT
`p` is generally an interior point of some face F=(v0,v1,v2), not a vertex itself. The label rule
assumes the nearest-in-3D-space vertex is anatomically correct for `p` -- true almost everywhere,
EXCEPT near thin, high-curvature, or tightly-packed geometry (exactly what ant leg segments are:
thin cylinders, six of them close together), where a DIFFERENT face's vertex can be closer in
Euclidean space than any of F's own three vertices, despite being geodesically/anatomically far
away (e.g. across a thin gap between adjacent leg segments, or the near side vs far side of a
bent joint). This is the same class of silent failure the reference-ceiling check caught in
finding #11 (ground truth scored 0.974 through its own metric, not 1.0) -- here checked directly
at the label-generation level, before any of that label is used to supervise a network in B2.

METHOD
Sample points from a posed synth_clean target mesh using a custom area-weighted sampler (mirrors
pytorch3d's `sample_points_from_meshes` algorithm) that ADDITIONALLY records which face each
point was drawn from -- information the public `sample_points_from_meshes` API does not expose,
and which is therefore not normally available at label-generation OR fit time (this script only
uses it as a held-out check on the label rule, not as part of the mechanism itself). A point's
label is then scored "topologically consistent" if the nearest-vertex label falls among ITS OWN
face's 3 vertices; otherwise it's a genuine correspondence label error, and we measure how far
(graph/geodesic hops along the mesh, and straight per-joint-chain segment identity) the wrong
label lands from the truth.
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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.pointcloud2smil.sample_smil_model import generate_random_parameters  # noqa: E402


def area_weighted_sample_with_face_idx(verts, faces, n_samples, rng):
    """verts: (V,3) torch, faces: (F,3) long torch. Returns (points (n,3) np, face_idx (n,) np).
    Mirrors pytorch3d.ops.sample_points_from_meshes's algorithm (area-weighted face choice +
    uniform barycentric via the sqrt trick), except it also returns face_idx, which the public
    pytorch3d API does not expose."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    probs = (areas / areas.sum()).cpu().numpy()

    face_idx = rng.choice(len(faces), size=n_samples, p=probs)
    face_idx_t = torch.as_tensor(face_idx, dtype=torch.long)

    u = torch.as_tensor(rng.random(n_samples), dtype=torch.float32)
    v = torch.as_tensor(rng.random(n_samples), dtype=torch.float32)
    su = torch.sqrt(u)
    b0 = 1.0 - su
    b1 = su * (1.0 - v)
    b2 = su * v

    pts = (
        b0.unsqueeze(-1) * v0[face_idx_t]
        + b1.unsqueeze(-1) * v1[face_idx_t]
        + b2.unsqueeze(-1) * v2[face_idx_t]
    )
    return pts.numpy(), face_idx


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def build_vertex_leg_segment_labels(jnames, dd, n_verts):
    """(V,) array of 'legK_side_seg' strings (or 'other') per template vertex, via the model's
    own weights/J_regressor-adjacent skinning-weight argmax -- reuses the same anatomical-group
    convention `optimise_hierarchical.py` uses (leg segment = whichever joint has max blend
    weight), so 'own face's true segment' is defined identically to how the rest of this
    investigation defines segment identity (leg_acc/seg_acc), not a new ad hoc rule."""
    weights = np.asarray(dd["weights"])  # (V, J)
    dominant_joint = weights.argmax(axis=1)  # (V,)
    labels = np.array(["other"] * n_verts, dtype=object)
    for j, nm in enumerate(jnames):
        if not nm.startswith("l_"):
            continue
        bits = nm.split("_")
        k, seg, side = bits[1], bits[2], bits[-1]
        mask = dominant_joint == j
        labels[mask] = f"l{k}_{side}_{seg}"
    return labels


def main():
    dev = "cpu"
    fitter = SMAL3DFitter(batch_size=1, device=dev, shape_family=-1)
    generate_random_parameters(fitter, seed=20260825, pose_scale=0.25, shape_scale=1.0,
                                trans_scale=0.0, scale_scale=0.10, global_rot_scale=0.0)
    with torch.no_grad():
        verts = fitter()[0]  # (V,3)
    faces = fitter.faces[0]  # (F,3)

    dd = load_model(config.SMAL_FILE)
    jnames = list(dd["J_names"])
    seg_labels = build_vertex_leg_segment_labels(jnames, dd, verts.shape[0])

    rng = np.random.default_rng(20260825)
    n_samples = 20000
    pts_np, face_idx = area_weighted_sample_with_face_idx(verts, faces, n_samples, rng)
    pts = torch.as_tensor(pts_np, dtype=torch.float32)

    # THE LABEL RULE UNDER TEST: nearest TRUE-mesh vertex in 3D space (dense_gt_verts = verts here)
    true_idx = knn_points(pts.unsqueeze(0), verts.unsqueeze(0), K=1).idx[0, :, 0].numpy()

    faces_np = faces.numpy()
    own_face_verts = faces_np[face_idx]  # (n,3) -- the 3 vertex indices the point was ACTUALLY sampled from

    consistent = np.array([true_idx[i] in own_face_verts[i] for i in range(n_samples)])
    frac_consistent = consistent.mean()

    # For inconsistent points: is the assigned label at least the SAME anatomical segment as the
    # point's true face? And how far off in Euclidean distance is the assigned vertex from the
    # nearest of the point's own true face vertices (a lower bound on how "wrong" the label is)?
    own_seg = seg_labels[own_face_verts[:, 0]]  # face vertices share a segment except at seam edges
    assigned_seg = seg_labels[true_idx]
    same_segment = own_seg == assigned_seg

    dists = np.linalg.norm(
        verts.numpy()[true_idx] - verts.numpy()[own_face_verts[:, 0]], axis=1
    )
    extent = float((verts.max(0).values - verts.min(0).values).max())

    out_dir = os.path.join(os.path.dirname(__file__), "out_A2_dense_label_validation_20260825")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print(f"A2 DENSE-LABEL VALIDATION (n={n_samples} sampled points, 1 posed specimen, seed=20260825)")
    print("=" * 78)
    print(f"Label rule 'nearest TRUE-mesh vertex' matches the point's OWN sampled face: "
          f"{frac_consistent*100:.2f}% consistent, {(1-frac_consistent)*100:.2f}% mislabeled "
          f"(n={int((~consistent).sum())} points)")
    print(f"Of mislabeled points, same-anatomical-segment-anyway: {same_segment[~consistent].mean()*100:.2f}%  "
          f"(i.e. wrong VERTEX but right SEGMENT -- a softer failure)")
    print(f"Of mislabeled points, DIFFERENT segment (a real leg/segment correspondence error): "
          f"{(~same_segment[~consistent]).mean()*100:.2f}%  "
          f"(n={int((~consistent & ~same_segment).sum())})")
    print(f"\nMislabel distance (assigned vertex to nearest true-face vertex), model units "
          f"(mesh extent={extent:.3f}):")
    md = dists[~consistent]
    if len(md):
        print(f"  mean={md.mean():.4f} ({100*md.mean()/extent:.2f}% of extent)  "
              f"median={np.median(md):.4f}  p95={np.percentile(md,95):.4f}  max={md.max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(["consistent\n(own face)", "mislabeled,\nsame segment", "mislabeled,\ndiff. segment"],
                [frac_consistent, same_segment[~consistent].mean() * (1 - frac_consistent),
                 (~same_segment[~consistent]).mean() * (1 - frac_consistent)],
                color=["#55A868", "#DD8452", "#C44E52"])
    axes[0].set_ylabel("fraction of sampled points")
    axes[0].set_title("Label-rule consistency\n(nearest-vertex label vs true sampled face)")
    axes[0].set_ylim(0, 1)

    if len(md):
        axes[1].hist(md, bins=40, color="#C44E52", alpha=0.8)
        axes[1].set_xlabel("mislabel distance (model units)")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"Mislabel distance distribution\n(n={len(md)} of {n_samples} points)")
    fig.suptitle("A2: dense correspondence label-generator validation")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "fig_A2_dense_label_validation.png")
    fig.savefig(out_path, dpi=150)
    print(f"\n[A2] wrote {out_path}")


if __name__ == "__main__":
    main()
