"""B2 training-corpus generator for the correspondence network (B1).

Lean version of `diagnostics/moonshot/make_synth_corpus.py` for training-scale corpora: skips
per-specimen .obj export (irrelevant for training, and prohibitively slow/heavy at N=thousands),
saves only what `CorrespondenceDataset` (`train_correspondence_net_B2_20260825.py`) needs:
`verts` (N,V,3), `joint_rot` (N,54,3), `names`. Uses A1's corrected sampler
(`generate_correlated_chain_parameters`, rho=0.6) by default -- the sampler this corpus exists to
validate use of, not the old i.i.d. one (`--sampler iid` available for the C1 ablation).

Held-out test set for anything needing GROUND TRUTH stays `synth_clean` (12 specimens) throughout
-- this corpus is for TRAINING ONLY, never used as an eval reference, so its size is not
constrained by finding #8's "no larger corpus exists for a GT eval reference" concern (that
finding is about corpora used as ground truth to VALIDATE against; this one only ever supplies
generator input, and the generator's own parameters are already fully known here by construction,
so there's no held-out-accuracy question to beg).
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from fitter_3d.pointcloud2smil.sample_smil_model import (  # noqa: E402
    generate_correlated_chain_parameters,
    generate_random_parameters,
)
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--pose_scale", type=float, default=0.25)
    ap.add_argument("--shape_scale", type=float, default=1.0)
    ap.add_argument("--scale_scale", type=float, default=0.10)
    ap.add_argument("--sampler", choices=["iid", "correlated"], default="correlated")
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=500, help="generation batch size (memory-bound, not a training hyperparam)")
    ap.add_argument("--out", default="diagnostics/moonshot/synth_b2_train_corrb06")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_verts, all_joint_rot, names = [], [], []
    n_done = 0
    seed = args.seed
    while n_done < args.n:
        b = min(args.batch, args.n - n_done)
        fitter = SMAL3DFitter(batch_size=b, device=dev, shape_family=-1)
        kwargs = dict(
            seed=seed, random_dist="normal", shape_scale=args.shape_scale,
            pose_scale=args.pose_scale, trans_scale=0.0, scale_scale=args.scale_scale,
            global_rot_scale=0.0,
        )
        if args.sampler == "correlated":
            generate_correlated_chain_parameters(fitter, rho=args.rho, **kwargs)
        else:
            generate_random_parameters(fitter, **kwargs)
        with torch.no_grad():
            verts = fitter()  # (b, V, 3)
        all_verts.append(verts.cpu().numpy())
        all_joint_rot.append(fitter.joint_rot.detach().cpu().numpy())
        names.extend([f"b2train_{n_done + i:05d}" for i in range(b)])
        n_done += b
        seed += 1
        print(f"[b2corpus] {n_done}/{args.n}")

    np.savez(
        out_path + ".npz",
        verts=np.concatenate(all_verts, axis=0),
        joint_rot=np.concatenate(all_joint_rot, axis=0),
        names=np.array(names),
        sampler=args.sampler,
        rho=args.rho if args.sampler == "correlated" else -1.0,
    )
    print(f"[b2corpus] wrote {out_path}.npz  n={n_done}  sampler={args.sampler} rho={args.rho}")


if __name__ == "__main__":
    main()
