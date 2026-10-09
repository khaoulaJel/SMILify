#!/usr/bin/env python3
"""C2 -- convert the trained correspondence network's (B1/B2) dense per-point predictions into a
pose-init candidate, reusing `geom_leg_init.py`'s existing per-leg rigid-alignment/IK machinery
(NOT new IK code) exactly as PHASE10_DESIGN_correspondence_network_20260825.md's C2 step
specifies. Same H0-reuse convention as every other candidate generator in this series
(`generate_multistart_coherent_init.py`, `generate_pca_coherent_init.py`, `generate_ik_init.py`):
H0 only fits the rigid 'body' pose (global_rot/trans), independent of leg-pose init choice.

Mechanism: for each of a leg's 5 non-coxa joints (tr, fe, ti, ta, pt -- see `gli.LEG_SEGMENTS`),
gather the scan points the network predicts belong to that segment class (argmax over
`seg_logits`), take their centroid as a WORLD-SPACE position estimate for the joint immediately
distal to that segment, and feed it as one `solve_chain_ik_multi` constraint at
`seg_idx = LEG_SEGMENTS.index(seg) - 1` (that function's own convention: seg_idx=0 is the
position after co's rotation, i.e. tr; seg_idx=4 is the tip/pt -- see its docstring). This is
exactly `estimate_leg_tip`/`estimate_leg_waypoint`'s role in `init_joint_rot_for_specimen_ik2`,
except the point-to-segment assignment comes from the trained network instead of a geodesic-
distance heuristic -- the network supplies up to 5 constraints per leg (vs. ik2's fixed 2),
whichever segments have enough confidently-classified points (`MIN_PTS_PER_BAND`, same threshold
`estimate_leg_tip`/`estimate_leg_waypoint` already use, for consistency).

`--oracle_gt_labels` bypasses the network entirely and uses each point's TRUE segment label
instead (nearest-vertex lookup against the specimen's own posed mesh, identical rule
`CorrespondenceDataset` used to generate training labels -- valid here because synth_clean shares
exact template topology, so vertex index IS the correspondence, no KNN-to-template needed, only
KNN from a sampled surface point back to its own mesh's nearest vertex). This isolates
`solve_chain_ik_multi`'s own centroid-based conversion noise from the network's prediction
accuracy: Part 1c already proved IK has a real ~14-21deg floor even given the TRUE tip position,
so if this oracle version is only marginally better than the network's version, the IK-conversion
step -- not the network -- is the bottleneck; if it's much better, the network's own accuracy is
still the limiting factor.

Network inference runs on `--n_points` (2048, matching `train_correspondence_net_B2_20260825.py`'s
own default) points UNIFORMLY RESAMPLED from the specimen's posed mesh surface via
`sample_points_from_meshes` -- NOT the mesh's raw vertices directly. This was verified necessary,
not assumed: feeding the ~10235 raw template vertices straight to the network (denser, and
differently distributed near joints than a uniform-area surface sample) caused a catastrophic
collapse to predicting nearly the entire point cloud as one leg's segments (checked directly --
on synth_clean's raw vertices the network puts >95% of points into 2-3 classes belonging to a
single leg; on a training-corpus specimen resampled to 2048 points exactly as
`CorrespondenceDataset` does, seg_acc=0.80 with a sane distribution across all 37 classes). The
network's fixed-radius PointNet++ ball queries (0.1/0.2/0.4, absolute units) are tuned to the
point density `sample_points_from_meshes` produces, not raw vertex density -- an I/O-swap-style
mismatch (see project convention: verify byte/distribution-equivalence on any point-source swap),
caught here before committing a 12-specimen GPU fitting run to it.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)

import geom_leg_init as gli  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    SMILCorrespondenceNet,
    build_segment_taxonomy,
    build_vertex_labels,
)
from fitter_3d.joint_limits import joint_limit_tensors  # noqa: E402
from pytorch3d.ops import knn_points  # noqa: E402


def load_model_dict(path):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="diagnostics/anatomical_pose_init/out_B2_correspondence_net_20260826/best_model.pt")
    ap.add_argument("--model_path", default="3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
    ap.add_argument("--corpus_path", default="diagnostics/moonshot/synth_clean/ground_truth.npz")
    ap.add_argument("--mesh_dir", default="diagnostics/moonshot/synth_clean")
    ap.add_argument("--h0_path", default="diagnostics/moonshot/runs/ACI_E_proximal_hier/H0_body.npz")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_ik_20260825")
    ap.add_argument("--out_name", default="network_correspondence_init.npz")
    ap.add_argument("--n_iters", type=int, default=300)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle_gt_labels", action="store_true",
                     help="bypass the network; use each point's TRUE segment label instead, to "
                          "isolate the IK-conversion ceiling from the network's own accuracy")
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dd = load_model_dict(args.model_path)
    jnames = list(dd["J_names"])
    chains = gli.leg_chains(jnames)
    rest_J = gli.rest_joint_positions(dd["J_regressor"], dd["v_template"])
    leg_rows_all = sorted({c - 1 for chain in chains.values() for c in chain[:5]})
    min_limits_all, max_limits_all = joint_limit_tensors(dd, "cpu")
    min_limits_all = torch.as_tensor(min_limits_all, dtype=torch.float32)
    max_limits_all = torch.as_tensor(max_limits_all, dtype=torch.float32)

    class_names, name_to_id = build_segment_taxonomy(jnames)
    if args.oracle_gt_labels:
        net = None
        seg_class_id_template, _ = build_vertex_labels(dd, class_names, name_to_id)
        seg_class_id_template = torch.as_tensor(seg_class_id_template, dtype=torch.long)
    else:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        assert ckpt["class_names"] == class_names, (
            "checkpoint's segment taxonomy does not match the current model's -- "
            "stale checkpoint or wrong --model_path, refusing to silently mismatch class ids"
        )
        net = SMILCorrespondenceNet(n_classes=len(class_names)).to(device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()

    gt = np.load(args.corpus_path, allow_pickle=True)
    jr_gt_aa = gt["joint_rot"]
    names = [str(n) for n in gt["names"]]

    h0 = np.load(args.h0_path, allow_pickle=True)
    h0_names = [str(n).removesuffix(".obj") for n in h0["labels"]]
    h0_by_name = {n: i for i, n in enumerate(h0_names)}

    n_pose = jr_gt_aa.shape[1]
    out_jr = np.zeros((len(names), n_pose, 3), dtype=np.float32)
    errs, n_constraints_used = [], []

    for i, name in enumerate(names):
        h0_i = h0_by_name[name]
        global_rot_aa = torch.as_tensor(h0["global_rot"][h0_i], dtype=torch.float32)
        trans = torch.as_tensor(h0["trans"][h0_i], dtype=torch.float32)

        verts, faces_obj, _ = load_obj(os.path.join(args.mesh_dir, f"{name}.obj"), load_textures=False)
        verts = verts.to(torch.float32)
        torch.manual_seed(args.seed + i)
        mesh = Meshes(verts=verts.unsqueeze(0), faces=faces_obj.verts_idx.unsqueeze(0))
        target_pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]  # (n_points,3)

        if args.oracle_gt_labels:
            # verts IS the specimen's own posed mesh, same topology as the template by
            # construction (synth_clean) -- vertex index alone is the correspondence, so the
            # ONLY unknown is which of the specimen's own vertices each sampled surface point is
            # closest to (the same nearest-mesh-vertex step CorrespondenceDataset does at train
            # time), not a template lookup.
            nn_idx = knn_points(target_pts.unsqueeze(0), verts.unsqueeze(0), K=1).idx[0, :, 0]
            seg_pred = seg_class_id_template[nn_idx]
        else:
            with torch.no_grad():
                seg_logits, _ = net(target_pts.unsqueeze(0).to(device))
            seg_pred = seg_logits[0].argmax(-1).cpu()  # (n_points,)

        anchors, R_root = gli.analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)

        jr_specimen = torch.zeros(n_pose, 3, dtype=torch.float32)
        n_constraints_i = 0
        for key, chain_idx in chains.items():
            rows = [chain_idx[i] - 1 for i in range(5)]
            min_l = min_limits_all[rows]
            max_l = max_limits_all[rows]

            constraints = []
            for seg in gli.LEG_SEGMENTS[1:]:  # tr, fe, ti, ta, pt (co is the anchor, not a target)
                cls_id = name_to_id[f"{key}_{seg}"]
                pts_here = target_pts[seg_pred == cls_id]
                if pts_here.shape[0] < gli.MIN_PTS_PER_BAND:
                    continue
                seg_idx = gli.LEG_SEGMENTS.index(seg) - 1
                constraints.append((seg_idx, pts_here.mean(dim=0)))

            if constraints:
                local_aa = gli.solve_chain_ik_multi(
                    R_root, anchors[key], rest_J, chain_idx, constraints, min_l, max_l, n_iters=args.n_iters
                )
                n_constraints_i += len(constraints)
            else:
                local_aa = torch.zeros(5, 3, dtype=torch.float32)
            for r_i in range(5):
                jr_specimen[rows[r_i]] = local_aa[r_i]

        out_jr[i] = jr_specimen.numpy()
        n_constraints_used.append(n_constraints_i)

        gt_legs = jr_gt_aa[i][leg_rows_all]
        e = geodesic_deg_batch(jr_specimen[leg_rows_all].numpy(), gt_legs)
        errs.append(float(e.mean()))
        print(f"[{name}] pose_err={errs[-1]:.2f} deg  constraints_used={n_constraints_i}/30 (6 legs x 5 segs)")

    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez(out_path, joint_rot=out_jr, names=np.array(names))

    errs = np.array(errs)
    n_constraints_used = np.array(n_constraints_used)
    print()
    print(f"pose error vs GT: mean={errs.mean():.2f} deg  median={np.median(errs):.2f} deg")
    print(f"constraints used per specimen: mean={n_constraints_used.mean():.1f}/30  min={n_constraints_used.min()}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
