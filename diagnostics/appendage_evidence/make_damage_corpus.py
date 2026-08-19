"""Phase 3 damage/missing-geometry corpus: same specimens as `synth_clean`, distal-leg vertices
removed from the TARGET mesh before fitting (per instruction: masking applied to the fitting
INPUT, not applied after the fact to a fit's output).

Ground truth is left UNCHANGED (literally copied from `synth_clean/ground_truth.npz`) -- the
question this corpus answers is "given a damaged specimen, how well does each measurement recover
the length the (intact) animal actually had", which needs the true, undamaged length as the
target, exactly as a real damaged museum specimen's true leg length is unknown but not zero.

Dropout vertex selection: `candidates.dropout_vertices`, frozen from the template (same style as
the C2 endpoint rule) -- the same vertex-index set is removed for every specimen at a given
`frac`, so drop30/drop60 mean the same physical thing across the whole corpus.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
import candidates as cand  # noqa: E402
from fitter_3d.pointcloud2smil.sample_smil_model import export_mesh_to_obj  # noqa: E402


def build_damaged_mesh(verts, faces, drop_idx):
    """verts: (V,3) np, faces: (F,3) np (this specimen's own topology, template's face list --
    unchanged across specimens). Returns (verts', faces') with `drop_idx` vertices removed and
    faces renumbered onto the remaining contiguous index set."""
    V = verts.shape[0]
    keep_mask = np.ones(V, dtype=bool)
    keep_mask[drop_idx] = False
    keep_idx = np.where(keep_mask)[0]
    remap = -np.ones(V, dtype=np.int64)
    remap[keep_idx] = np.arange(keep_idx.size)

    face_keep = keep_mask[faces].all(axis=1)
    new_faces = remap[faces[face_keep]]
    return verts[keep_idx], new_faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, required=True, help="fraction of each leg's distal patch to drop")
    ap.add_argument("--src_corpus", default="synth_clean")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    M = ms.load_model()
    cache = cand.build_geodesic_cache(M)
    drop_by_leg = cand.dropout_vertices(M, cache, args.frac)
    all_drop = np.concatenate(list(drop_by_leg.values()))
    print(f"[damage] frac={args.frac}: dropping {all_drop.size} vertices/leg-set "
          f"({', '.join(f'{k}:{v.size}' for k, v in drop_by_leg.items())})")

    src = os.path.join(MOON, args.src_corpus)
    gt = np.load(os.path.join(src, "ground_truth.npz"))
    gtv, names = gt["verts"], [str(x) for x in gt["names"]]
    faces = np.asarray(M["dd"]["f"])

    outdir = os.path.join(MOON, args.out)
    os.makedirs(outdir, exist_ok=True)
    for i, name in enumerate(names):
        vd, fd = build_damaged_mesh(gtv[i], faces, all_drop)
        export_mesh_to_obj(
            torch.tensor(vd, dtype=torch.float32),
            torch.tensor(fd, dtype=torch.int64)[None],
            os.path.join(outdir, f"{name}.obj"),
        )

    # ground truth is UNCHANGED -- copy verbatim, this corpus tests recovery of the TRUE
    # (undamaged) length from damaged input, not a redefinition of truth.
    np.savez(
        os.path.join(outdir, "ground_truth.npz"),
        verts=gtv,
        names=gt["names"],
        betas=gt["betas"],
        joint_rot=gt["joint_rot"],
        log_beta_scales=gt["log_beta_scales"],
        noise=gt["noise"],
        extent=gt["extent"],
    )
    print(f"[damage] wrote {len(names)} damaged targets + unchanged ground_truth.npz -> {outdir}")


if __name__ == "__main__":
    main()
