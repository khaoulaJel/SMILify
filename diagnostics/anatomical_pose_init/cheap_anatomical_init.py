"""Arm B (cheap_anatomical) initialiser for the Anatomical Initialization Ceiling Test
(2026-08-20 experiment spec). Produces a deterministic, correspondence-free `joint_rot`
init from raw target geometry alone -- no ground truth, no learned model, no fitting run.

REUSE DISCLOSURE (explicit user decisions, 2026-08-20, two rounds): the experiment protocol
originally said "do not reuse any previous geometry-only initializer code." Round 1: the user
approved reusing `fitter_3d/geom_leg_init.py` (Task 7 "Intervention B")'s per-segment chain-IK
for the leg-direction step. Round 2: validating that mechanism end-to-end for the first time
(`probe_B0_geom_init_validation.py`, never actually run before despite its own docstring
implying it had been) showed it makes EVERY leg joint substantially WORSE than zero-init
(leg_distal position error 0.436 vs 0.163 zero-init on `synth_clean`, 3x worse) -- not a wiring
bug, reproduced across all 36 leg joints. Per the user's follow-up decision, geom_leg_init's
per-segment band-estimation + chain IK is NOT used. This script now uses
`diagnostics/anatomical_pose_init/simple_leg_heuristic.py`, a fresh single-rigid-rod-per-leg
heuristic (see that module's docstring) written after the above finding -- it still measured
worse than zero-init pre-optimization (leg_distal 0.333 vs 0.163, ~2x), which is reported
honestly as a Section-A finding rather than hidden; whether the FULL D1 pipeline (not just raw
init quality) recovers from that worse start is exactly what this experiment tests. This file
still adds the one piece that did not exist anywhere in the repo: PCA-based global-orientation
estimation (protocol step 1) -- though see step 1 below for why it is reported, not fed into
the leg step.

Pipeline per specimen:
  1. Global orientation: PCA on the full target point cloud (robust core -- see
     `_pca_axes`), aligned to the template's own PCA axes (computed once, fixed). Sign
     ambiguity of each eigenvector is resolved by keeping the axis whose dot product with
     the corresponding template axis is positive (standard closest-rotation disambiguation,
     not GT-specific -- see `_pca_axes` docstring). Reported as the Section-A "global
     orientation error" diagnostic only -- NOT fed into the leg step below, because
     `optimise_hierarchical.py --init_joint_rot_from` has no global-orientation override (it
     always zero-inits `global_rot`) and `synth_clean` is canonically aligned by construction
     (true global orientation is always identity), so identity is both what the fitter will
     actually use and the numerically correct choice here.
  2. Translation: closed-form target-centroid alignment (see `process_specimen`).
  3. Leg direction vectors + joint rotation recovery:
     `simple_leg_heuristic.estimate_leg_rotations` (single rigid-rod-per-leg, see that
     module's docstring for the full method and why it replaced geom_leg_init).
  4. Head/gaster/wing/antenna joints are left at zero rotation (rest pose) -- the protocol
     explicitly allows this for segments the heuristic cannot reliably estimate ("do not
     attempt accurate pretarsus or antenna tip localization").
  5. Logging: per-specimen predicted joint_rot/global_rot + a human-readable summary of which
     legs got a nonzero coxa rotation vs fell back to rest (too few assigned points).

Deterministic given the same input mesh (no RNG anywhere in this file or in geom_leg_init).

Output: <out_dir>/cheap_init.npz with the SAME keys/shapes as `make_synth_corpus.py`'s
ground_truth.npz (`joint_rot` (N,54,3), `names` (N,)), so it is a drop-in for
`optimise_hierarchical.py --init_joint_rot_from`. Also writes `<out_dir>/cheap_init_global.npz`
(global_rot_aa, trans per specimen -- NOT consumed by the fitter, which has no global-orient
override; kept for the initialisation-quality (Section A) audit) and one
`<out_dir>/cheap_init_log/<specimen>.txt` per specimen.
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.transforms import matrix_to_axis_angle  # noqa: E402

import simple_leg_heuristic as slh  # noqa: E402

DEFAULT_MODEL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"


def load_model(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    return dd


def _pca_axes(pts, ref_axes=None):
    """Orthonormal (3,3) principal axes (columns, descending variance) of `pts` (P,3).

    Sign of the first two axes is fixed by positive correlation with `ref_axes` (the
    template's own axes) when given -- a standard closest-rotation disambiguation (choose
    the sign minimizing the resulting rotation's angle to the identity/reference frame),
    not a GT-specific shortcut: any PCA-registration pipeline needs *some* sign rule, and
    using the reference frame as the tiebreak is the common default. The third axis is
    always `cross(axis0, axis1)` to guarantee a proper (det=+1) rotation.
    """
    c = pts.mean(0)
    centered = pts - c
    cov = centered.T @ centered / max(len(centered), 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    axes = v[:, order]
    if ref_axes is not None:
        for i in range(2):
            if np.dot(axes[:, i], ref_axes[:, i]) < 0:
                axes[:, i] *= -1
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    return axes, c


def estimate_global_pose(target_verts_np, template_axes, template_centroid, root_rest_np):
    """PCA global orientation + closed-form translation. Returns (R (3,3), trans (3,)) numpy.

    trans is solved so the rigidly-rotated template's MEAN vertex lands on the target
    centroid: target_centroid = R @ (template_centroid - root_rest) + root_rest + trans,
    matching the convention `geom_leg_init.analytic_coxa_anchors` and `SMAL3DFitter.forward`
    use (rotation pivots about the root's own rest position, trans is a uniform shift on
    top). See module docstring step 2.
    """
    specimen_axes, specimen_centroid = _pca_axes(target_verts_np, ref_axes=template_axes)
    R = specimen_axes @ template_axes.T
    trans = specimen_centroid - R @ (template_centroid - root_rest_np) - root_rest_np
    return R, trans


def process_specimen(mesh_path, dd, jnames, rest_J, template_axes, template_centroid, root_rest_np):
    verts, _, _ = load_obj(mesh_path, load_textures=False)
    target_np = verts.numpy().astype(np.float64)

    R, trans_np = estimate_global_pose(target_np, template_axes, template_centroid, root_rest_np)
    global_rot_aa_np = _rotmat_to_aa(R)  # reported as the Section-A diagnostic only, see below

    # `optimise_hierarchical.py --init_joint_rot_from` has no global-orientation override --
    # it always zero-inits global_rot (SMAL3DFitter.__init__ only accepts init_joint_rot).
    # synth_clean is canonically aligned by construction (make_synth_corpus.py uses
    # global_rot_scale=0.0), so identity is also the numerically correct choice here, and it
    # is the ONE the fitter will actually start from regardless of what we estimate. Feeding
    # the (noisy, up to ~20deg on this corpus) PCA rotation estimate into the leg chain-IK
    # below would corrupt joint_rot for a reason that has nothing to do with leg-articulation
    # estimation quality -- so the IK step uses identity here, and the PCA estimate is kept
    # only as the separately-reported "global orientation error" diagnostic (protocol 3.A).
    R_for_legs = np.eye(3)
    trans_for_legs_np = target_np.mean(0) - root_rest_np

    global_rot_aa = torch.as_tensor(_rotmat_to_aa(R_for_legs), dtype=torch.float32)
    trans = torch.as_tensor(trans_for_legs_np, dtype=torch.float32)
    target_pts = torch.as_tensor(target_np, dtype=torch.float32)

    joint_rot = slh.estimate_leg_rotations(global_rot_aa, trans, rest_J, jnames, target_pts)

    chains = slh.leg_chains(jnames)
    anchors, R_root = slh.analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
    assigned = slh.assign_points_to_legs(target_pts, anchors)
    summary_lines = [
        f"specimen: {os.path.basename(mesh_path)}",
        f"global_rot_aa (deg, PCA diagnostic only, NOT used for legs): "
        f"{np.degrees(global_rot_aa_np).round(2).tolist()}",
        f"trans (used for legs): {trans_for_legs_np.round(4).tolist()}",
        "per-leg point counts and whether a nonzero coxa rotation was estimated:",
    ]
    for key, chain_idx in chains.items():
        estimated = assigned[key].shape[0] >= slh.MIN_PTS
        summary_lines.append(f"  {key}: {assigned[key].shape[0]} pts assigned, estimated={estimated}")

    return joint_rot.numpy(), global_rot_aa_np, trans_for_legs_np, "\n".join(summary_lines)


def _rotmat_to_aa(R_np):
    R_t = torch.as_tensor(R_np, dtype=torch.float32).unsqueeze(0)
    return matrix_to_axis_angle(R_t)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", default="diagnostics/moonshot/synth_clean")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_ceiling_20260820")
    args = ap.parse_args()

    dd = load_model(args.model)
    jnames = list(dd["J_names"])
    Jr = np.asarray(dd["J_regressor"])
    vt = np.asarray(dd["v_template"])
    rest_J = torch.as_tensor(Jr @ vt, dtype=torch.float32)
    root_rest_np = (Jr @ vt)[0]

    template_axes, template_centroid = _pca_axes(vt.astype(np.float64))

    files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    print(f"[cheap_init] {len(files)} specimens in {args.mesh_dir}", flush=True)

    log_dir = os.path.join(args.out_dir, "cheap_init_log")
    os.makedirs(log_dir, exist_ok=True)

    all_joint_rot, all_global_rot, all_trans = [], [], []
    for f, name in zip(files, names):
        jr, grot, trans, summary = process_specimen(
            f, dd, jnames, rest_J, template_axes, template_centroid, root_rest_np
        )
        all_joint_rot.append(jr)
        all_global_rot.append(grot)
        all_trans.append(trans)
        with open(os.path.join(log_dir, f"{name}.txt"), "w") as fh:
            fh.write(summary + "\n")
        print(f"[cheap_init] {name} done", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        os.path.join(args.out_dir, "cheap_init.npz"),
        joint_rot=np.stack(all_joint_rot).astype(np.float32),
        names=np.array(names),
    )
    np.savez(
        os.path.join(args.out_dir, "cheap_init_global.npz"),
        global_rot_aa=np.stack(all_global_rot).astype(np.float32),
        trans=np.stack(all_trans).astype(np.float32),
        names=np.array(names),
    )
    with open(os.path.join(args.out_dir, "cheap_init_provenance.json"), "w") as fh:
        json.dump(
            {
                "mesh_dir": args.mesh_dir,
                "model": args.model,
                "n_specimens": len(names),
                "command": " ".join(sys.argv),
            },
            fh,
            indent=2,
        )
    print(f"[cheap_init] wrote {args.out_dir}/cheap_init.npz ({len(names)} specimens)", flush=True)


if __name__ == "__main__":
    main()
