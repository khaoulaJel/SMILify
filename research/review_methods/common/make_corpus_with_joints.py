"""Regenerate the C3/C11 training corpus (`synth_b2_train_corrb06.npz`) WITH the parameters and
forward-kinematics joint positions that the original file did not store.

Why regenerate instead of re-using: skeleton-conditioned methods (M01, M05, M06) need per-specimen
joint positions as input/targets, and the original corpus kept only `verts` + `joint_rot`.

Same generator, same arguments, same seeds as `make_b2_training_corpus_20260825.py`, so the
specimens are identical. This is VERIFIED, not assumed: `--check_against` compares every
regenerated vertex array to the original corpus and refuses to write if the max abs difference
exceeds `--tol` (CPU/GPU float32 differences are ~1e-6; a seed or argument mismatch gives O(0.1)).

Joints are the FK rotation pivots `smal_model.J_transformed + trans` -- the same definition JAB uses
(`diagnostics/joint_alignment_benchmark/tools/model_joints.py`), so synthetic and real joint numbers
mean the same thing.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

from fitter_3d.pointcloud2smil.sample_smil_model import (  # noqa: E402
    generate_correlated_chain_parameters,
    generate_random_parameters,
)
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402

PARAMS = ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales", "betas_trans"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--pose_scale", type=float, default=0.25)
    ap.add_argument("--shape_scale", type=float, default=1.0)
    ap.add_argument("--scale_scale", type=float, default=0.10)
    ap.add_argument("--sampler", choices=["iid", "correlated"], default="correlated")
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--check_against", default="diagnostics/moonshot/synth_b2_train_corrb06.npz",
                    help="original corpus to verify against; '' disables (new corpora only)")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", default="/hpcwork/nao48500/review_methods/corpus_b2_with_joints.npz")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = np.load(os.path.join(REPO, args.check_against))["verts"] if args.check_against else None

    store = {k: [] for k in PARAMS + ["verts", "joints"]}
    n_done, seed, worst = 0, args.seed, 0.0
    while n_done < args.n:
        b = min(args.batch, args.n - n_done)
        fitter = SMAL3DFitter(batch_size=b, device=dev, shape_family=-1)
        kw = dict(seed=seed, random_dist="normal", shape_scale=args.shape_scale,
                  pose_scale=args.pose_scale, trans_scale=0.0, scale_scale=args.scale_scale,
                  global_rot_scale=0.0)
        if args.sampler == "correlated":
            generate_correlated_chain_parameters(fitter, rho=args.rho, **kw)
        else:
            generate_random_parameters(fitter, **kw)
        with torch.no_grad():
            verts = fitter()
            joints = fitter.smal_model.J_transformed + fitter.trans.unsqueeze(1)
        v = verts.cpu().numpy()
        if ref is not None:
            err = float(np.abs(v - ref[n_done:n_done + b]).max())
            worst = max(worst, err)
            if err > args.tol:
                raise SystemExit(f"batch at {n_done}: regenerated verts differ from "
                                 f"{args.check_against} by {err:.3e} > tol {args.tol} -- refusing")
        store["verts"].append(v)
        store["joints"].append(joints.cpu().numpy())
        for k in PARAMS:
            store[k].append(getattr(fitter, k).detach().cpu().numpy())
        n_done += b
        seed += 1
        print(f"[corpus] {n_done}/{args.n}  max|dv| vs original so far = {worst:.2e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {k: np.concatenate(v, 0) for k, v in store.items()}
    np.savez(args.out, **out, sampler=args.sampler, rho=args.rho, seed=args.seed,
             verified_against=args.check_against, verified_max_abs_diff=worst)
    print(f"[corpus] wrote {args.out}  joints {out['joints'].shape}  verified max|dv|={worst:.2e}")


if __name__ == "__main__":
    main()
