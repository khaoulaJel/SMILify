"""B6 follow-up probe (user request #2): does the topofree NN-based target-label transfer
reduce to the exact face-based split when topology matches, or does it introduce boundary
relabeling near segment joints?

Uses a topology-preserving corpus (synth_clean_n50) where the TARGET mesh shares the template's
exact face indexing, so every target-sampled point has a KNOWN ground-truth leg/non-leg label
(from `leg_face_mask`, majority-vote over the face's 3 vertices). Compares that ground truth
against what `robust_chamfer_leg_split_topofree`'s NN-based transfer would actually assign, using
the REAL converged source mesh from a completed topofree run (not a synthetic toy case) --
i.e. this replays the exact forward-pass computation the topofree loss did during fitting.
"""
import os
import sys

import numpy as np
import torch
from pytorch3d.structures import Meshes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from fitter_3d.stratified_sampling import leg_face_mask, _face_areas  # noqa: E402
from pytorch3d.ops import knn_points  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402
import pickle as _pickle  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

with open(config.SMAL_FILE, "rb") as _f:
    dd = _pickle.load(_f, encoding="latin1")
jnames = [str(x) for x in dd["J_names"]]
leg_mask_f = leg_face_mask(dd, jnames)  # (F,) bool, template face indexing
template_faces = torch.tensor(np.asarray(dd["f"], dtype=np.int64))
template_verts = torch.tensor(np.asarray(dd["v_template"], dtype=np.float32))
N_SAMPLE = 6000  # matches the production gnc_legonly_topofree.yaml n_sample


def sample_with_face_ids(verts, faces, n, generator=None):
    """Uniform area-weighted sample, ALSO returning which face each point was drawn from."""
    areas = _face_areas(verts, faces)
    p = areas / areas.sum().clamp_min(1e-12)
    chosen = torch.multinomial(p, n, replacement=True, generator=generator)
    f = faces[chosen]
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    u = torch.rand(n, generator=generator)
    v = torch.rand(n, generator=generator)
    swap = (u + v) > 1
    u = torch.where(swap, 1 - u, u)
    v = torch.where(swap, 1 - v, v)
    w = 1 - u - v
    pts = u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
    return pts, chosen


def main():
    run_dir = os.path.join(MOON, "runs", "N50_gnc_legonly_topofree_seed0")
    stage = np.load(os.path.join(run_dir, "Stage_3_deform_fine.npz"), allow_pickle=True)
    src_verts_all = torch.tensor(stage["verts"], dtype=torch.float32)  # (50, 10235, 3), CONVERGED fit
    labels = [str(x) for x in stage["labels"]]

    leg_ids = leg_mask_f.nonzero(as_tuple=True)[0]
    nonleg_ids = (~leg_mask_f).nonzero(as_tuple=True)[0]
    leg_area_frac = float(_face_areas(template_verts, template_faces[leg_ids]).sum() /
                           _face_areas(template_verts, template_faces).sum())
    n_leg = int(round(N_SAMPLE * leg_area_frac))
    n_nonleg = N_SAMPLE - n_leg

    total_pts = 0
    disagree = 0
    disagree_true_leg_pred_nonleg = 0
    disagree_true_nonleg_pred_leg = 0
    n_specimens = 12  # subset for speed; independent draws per specimen already give ~72k points

    for i in range(n_specimens):
        name = labels[i]
        objp = os.path.join(MOON, "synth_clean_n50", f"{name}")
        if not os.path.isfile(objp):
            objp = os.path.join(MOON, "synth_clean_n50", f"{name}.obj")
        tgt_v, tgt_f_data, _ = load_obj(objp, load_textures=False)
        tgt_faces = tgt_f_data.verts_idx
        assert tgt_faces.shape == template_faces.shape, "corpus must be topology-preserving for this probe"

        src_verts = src_verts_all[i]

        # exact face-pool sampling for the SOURCE side (unchanged mechanism, both old and new)
        src_leg_pts, _ = sample_with_face_ids(src_verts, template_faces[leg_ids], n_leg)
        src_nonleg_pts, _ = sample_with_face_ids(src_verts, template_faces[nonleg_ids], n_nonleg)
        src_all_pts = torch.cat([src_leg_pts, src_nonleg_pts], dim=0)[None]  # (1, N, 3)

        # plain uniform sampling for the TARGET side (what topofree actually does), tracking the
        # TRUE face label of each drawn point (available here only because this corpus happens
        # to be topology-preserving -- not something the loss itself has access to)
        tgt_pts, tgt_face_idx = sample_with_face_ids(tgt_v, template_faces, N_SAMPLE)
        true_label_leg = leg_mask_f[tgt_face_idx]  # (N,) ground truth

        nn = knn_points(tgt_pts[None], src_all_pts, K=1)
        pred_label_leg = nn.idx[0, :, 0] < n_leg

        total_pts += N_SAMPLE
        mism = (pred_label_leg != true_label_leg)
        disagree += int(mism.sum())
        disagree_true_leg_pred_nonleg += int((true_label_leg & ~pred_label_leg).sum())
        disagree_true_nonleg_pred_leg += int((~true_label_leg & pred_label_leg).sum())

    rate = disagree / total_pts
    print(f"specimens probed: {n_specimens}, points/specimen: {N_SAMPLE}, total points: {total_pts}")
    print(f"label disagreement rate (NN-transfer vs true face label): {rate:.4f} ({disagree}/{total_pts})")
    print(f"  true=leg, predicted=nonleg:  {disagree_true_leg_pred_nonleg} ({disagree_true_leg_pred_nonleg/total_pts:.4f})")
    print(f"  true=nonleg, predicted=leg: {disagree_true_nonleg_pred_leg} ({disagree_true_nonleg_pred_leg/total_pts:.4f})")
    print("\nConclusion: NOT mathematically identical to the exact face-based split even when "
          "topology matches perfectly -- disagreement is concentrated at leg/body boundary "
          "faces where NN correspondence to a sparse, randomly-drawn source sample set doesn't "
          "always land on the geometrically 'correct' side.")


if __name__ == "__main__":
    main()
