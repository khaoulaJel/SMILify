"""Y1 step 1 -- generate the pose ladder: identical shape, pose magnitude varied.

Bar in PREREGISTRATION_Y1_pose_causality.md, fixed before this ran.

`generate_synth_large.py::sample_batch` seeds once and draws `betas` BEFORE `joint_rot`, so calling
it with the same seed and different `pose_scale` yields IDENTICAL betas and pose that is identically
directed and merely rescaled. That is the one-factor ladder the experiment needs, and it is asserted
here (check 5.2) rather than assumed.

`sample_batch` returns only (verts, joint_rot_matrix, joint_rot_aa), but it SETS betas and
log_beta_scales on the smil object, so they are read back off the object afterwards rather than
duplicating the sampler (which could silently drift from it).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "moonshot"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from generate_synth_large import sample_batch  # noqa: E402

POSE_SCALES = [0.05, 0.15, 0.25, 0.50, 0.75]


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default="diagnostics/pose_causality/corpora")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from fitter_3d.trainer import SMAL3DFitter
    smil = SMAL3DFitter(batch_size=args.n, device=device)
    faces = np.asarray(smil.faces.cpu().numpy() if torch.is_tensor(smil.faces) else smil.faces)
    faces = faces.reshape(-1, 3).astype(np.int64)

    out_root = os.path.join(REPO, args.out_root)
    os.makedirs(out_root, exist_ok=True)

    betas_ref, rot_ref, meta = None, None, {}
    for ps in POSE_SCALES:
        verts, _, rot_aa = sample_batch(smil, args.n, pose_scale=ps, seed=args.seed)
        betas = smil.betas.detach().cpu().numpy().copy()
        lbs = smil.log_beta_scales.detach().cpu().numpy().copy()

        # ---- check 5.2: one-factor ladder --------------------------------------------------
        if betas_ref is None:
            betas_ref, rot_ref, ps_ref = betas, rot_aa, ps
        else:
            if not np.array_equal(betas, betas_ref):
                raise SystemExit(f"VOID: betas differ at pose_scale={ps}; ladder is confounded")
            expected = rot_ref * (ps / ps_ref)
            if not np.allclose(rot_aa, expected, atol=1e-5):
                raise SystemExit(f"VOID: joint_rot at {ps} is not {ps/ps_ref:.3f}x the reference")

        # ---- check 5.3: pose magnitude ------------------------------------------------------
        deg = float(np.degrees(np.linalg.norm(rot_aa, axis=2)).mean())

        d = os.path.join(out_root, f"ps{ps:.2f}")
        os.makedirs(d, exist_ok=True)
        for i in range(args.n):
            write_obj(os.path.join(d, f"synth_{i:03d}.obj"), verts[i], faces)
        np.savez_compressed(os.path.join(d, "ground_truth.npz"),
                            verts=verts, joint_rot=rot_aa, betas=betas, log_beta_scales=lbs,
                            pose_scale=ps)
        meta[f"{ps:.2f}"] = {"dir": d, "mean_deg_per_joint": deg, "n": args.n}
        print(f"pose_scale {ps:.2f}: {args.n} meshes, mean {deg:.2f} deg/joint -> {d}", flush=True)

    degs = [meta[f"{p:.2f}"]["mean_deg_per_joint"] for p in POSE_SCALES]
    strictly_increasing = all(b > a for a, b in zip(degs, degs[1:]))
    print(f"\n[CHECK 5.2] betas byte-identical across the ladder, joint_rot exactly rescaled: PASS")
    print(f"[CHECK 5.3] mean deg/joint strictly increasing: {strictly_increasing}  {degs}")
    if not strictly_increasing:
        raise SystemExit("VOID: the ladder does not vary its own factor")

    meta["n_betas"] = int(betas_ref.shape[1])
    with open(os.path.join(out_root, "ladder.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nn_betas = {meta['n_betas']} (instrument check 5.1 requires gen/spread <= 0.10 by k=this)")
    print(f"wrote {out_root}/ladder.json")


if __name__ == "__main__":
    main()
