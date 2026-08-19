"""Cycle2 B3: properly-isolated capacity-vs-sampling ceiling (2x2 design from the task list).

The previous cycle's Rank 2 ceiling (overnight_20260818) froze pose but used the SAME
area-weighted chamfer sampler as the rest of the pipeline, so `log_beta_scales` for distal
joints still received a starved gradient signal -- that measurement conflated H3 (sampling)
with H6 (capacity) rather than separating them.

This script adds the missing cell: GT pose (frozen) + free per-segment bone scale (scheme
"shape" equivalent) + Rank 5's leg-protected TARGET sampler (`distal_quota_protected`), so
distal geometry gets guaranteed sampling density INDEPENDENT of pose accuracy AND independent
of a shared global sampling budget with non-leg anatomy.

Implementation: reuses `fitter_3d.trainer_hierarchical.HierarchicalStage` directly (not
`optimise_hierarchical.py`'s fixed H0-H3 schedule) as ONE custom stage:
  - `smal = SMAL3DFitter(..., init_joint_rot=<GT joint_rot>)` -- pose starts at GT.
  - `active_groups=[]` -- `joint_mask_for_groups(..., set())` is all-False, so
    `self.smal.joint_rot.grad[:, ~self.joint_mask, :] = 0.0` zeroes EVERY joint's gradient every
    step (trainer_hierarchical.py:637) -- joint_rot is pinned at GT for the entire run, exactly
    like Section 9.5.2's `scheme: shape` did, verified the same way (max abs diff check below).
  - `optimise_deform=False` (default) -- no deform_verts, matching the original ceiling design.
  - `partitioned=False` -- plain GLOBAL bidirectional chamfer (no TargetPartition, no per-group
    n_t<10 skip logic at all -- that mechanism is irrelevant here since we want the sampler's
    OWN guarantee, not group-membership-gated correspondence).
  - `distal_quota_protected=0.3` (same value as Rank 5, for consistency) -- ONLY changes which
    TARGET points `_sample_targets` draws (trainer_hierarchical.py:593-604); the global chamfer
    then uses this leg-protected sample instead of `sample_points_from_meshes`'s default
    area-weighted draw.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_hierarchical import HierarchicalStage, vertex_groups  # noqa: E402
from fitter_3d.stratified_sampling import distal_face_mask, leg_face_mask  # noqa: E402
import pickle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--nits", type=int, default=1200)
    ap.add_argument("--quota", type=float, default=0.3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    mesh_files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    names = [os.path.basename(f) for f in mesh_files]
    stems = [os.path.splitext(n)[0] for n in names]
    _, target_meshes = load_meshes(mesh_files=mesh_files, device=device)

    with open(config.SMAL_FILE, "rb") as f:
        dd = pickle.load(f, encoding="latin1")
    jnames = [str(x) for x in dd["J_names"]]

    gt = np.load(os.path.join(args.mesh_dir, "ground_truth.npz"))
    gt_names = [str(x) for x in gt["names"]]
    idx = [gt_names.index(s) for s in stems]
    init_joint_rot = gt["joint_rot"][idx]

    smal = SMAL3DFitter(batch_size=len(names), device=device, shape_family=-1, init_joint_rot=init_joint_rot)

    vg, group_names = vertex_groups(np.asarray(dd["weights"]), jnames, split_distal=False)
    dmask = distal_face_mask(dd, jnames).to(device)
    lmask = leg_face_mask(dd, jnames).to(device)
    template_faces = torch.tensor(np.asarray(dd["f"], dtype=np.int64), device=device)
    template_verts = torch.tensor(np.asarray(dd["v_template"], dtype=np.float32), device=device)

    stage = HierarchicalStage(
        name="B3_ceiling_protected",
        nits=args.nits,
        smal=smal,
        target_meshes=target_meshes,
        vertex_group=vg,
        group_names=group_names,
        joint_names=jnames,
        active_groups=[],  # empty set -> joint_mask all False -> joint_rot frozen at GT for good
        partitioned=False,  # plain global chamfer, no per-group skip logic
        lr=0.005,
        joint_lr=0.0,
        # HierarchicalStage's own defaults are nonzero for w_edge/w_normal/w_laplacian/
        # w_beta_prior (unlike MoonshotStage's, which the original Rank 2 ceiling used) --
        # explicitly zeroed here to match Rank 2's zero-regularization design as closely as
        # possible, so the ONLY intended difference vs Rank 2 is the sampler.
        loss_weights={"w_chamfer": 1.0, "w_edge": 0.0, "w_normal": 0.0, "w_laplacian": 0.0,
                      "w_sym": 0.5, "w_midline": 2.0, "w_beta_prior": 0.0, "w_limit": 0.0},
        n_sample=6000,
        out_dir=args.results_dir,
        device=device,
        optimise_deform=False,
        distal_quota_protected=args.quota,
        distal_face_mask=dmask,
        template_faces=template_faces,
        leg_face_mask=lmask,
        template_verts=template_verts,
    )

    before = smal.joint_rot.detach().clone()
    stage.run()
    after = smal.joint_rot.detach()
    max_diff = float((after - before).abs().max())
    print(f"[b3] max abs joint_rot drift from GT init after {args.nits} its: {max_diff}", flush=True)
    assert max_diff == 0.0, "joint_rot moved -- freeze mechanism broken, do not trust this ceiling number"

    stage.save_npz(labels=names)
    print(f"[b3] wrote {args.results_dir}/B3_ceiling_protected.npz", flush=True)


if __name__ == "__main__":
    main()
