"""Probe: unnormalised skinning weights in the ant model files (found 2026-10-09 during Task 2 gate 1).

Linear blend skinning computes v' = sum_j w_ij T_j v_i, so a vertex whose weight row sums to s < 1
is scaled by s even at zero pose (T_j = I). This script reports, per model file, how many rows do not
sum to 1 and on which joint, and how far the default SMAL3DFitter state moves those vertices.

    SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl \
    python diagnostics/fig_registration_stages/probes/skinning_weight_defect_PROBE.py [fit.npz ...]

Optional fit npz files (one specimen each) are checked for the resulting left/right asymmetry.
"""

import glob
import hashlib
import os
import pickle
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)


def main():
    for p in sorted(glob.glob(os.path.join(REPO, "3D_model_prep", "*.pkl"))):
        with open(p, "rb") as f:
            dd = pickle.load(f, encoding="latin1")
        if "weights" not in dd:
            continue
        w = np.asarray(dd["weights"])
        s = w.sum(1)
        bad = np.abs(s - 1) > 1e-3
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
        by = {}
        if bad.any():
            jn = list(dd["J_names"])
            u, c = np.unique(w[bad].argmax(1), return_counts=True)
            by = {jn[k]: int(n) for k, n in zip(u, c)}
        print(f"{os.path.basename(p):55s} md5 {md5} V={len(s)} rows_not_1={bad.sum()} min_sum={s.min():.3f} {by}")

    import config
    from fitter_3d.trainer import SMAL3DFitter

    dd = config.dd
    print("\nconfig.SMAL_FILE =", config.SMAL_FILE)
    w = np.asarray(dd["weights"])
    s = w.sum(1)
    bad = np.where(np.abs(s - 1) > 1e-3)[0]
    vt = np.asarray(dd["v_template"])
    smal = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    with torch.no_grad():
        v0 = smal()[0].numpy()
    d = np.linalg.norm(v0 - vt, axis=1)
    print(f"default state vs v_template: max {d.max():.4f}; moved > 0.05: {(d > 0.05).sum()}; "
          f"all moved verts are rows_not_1: {set(np.where(d > 1e-4)[0]) <= set(bad)}")
    print(f"  ratio |v0|/|v_template| on rows_not_1 vs weight sum: max abs diff "
          f"{np.abs(np.linalg.norm(v0[bad], axis=1) / np.linalg.norm(vt[bad], axis=1) - s[bad]).max():.2e}")

    tree = cKDTree(vt)
    mir = np.array([1.0, -1.0, 1.0])
    _, partner = tree.query(vt[bad] * mir)
    ok = np.where((np.abs(s - 1) < 1e-3) & (np.abs(vt[:, 1]) > 0.05))[0]
    samp = np.random.default_rng(0).choice(ok, 500, replace=False)
    _, p2 = tree.query(vt[samp] * mir)
    for path in sys.argv[1:]:
        v = np.load(path, allow_pickle=True)["verts"][0]
        a = np.linalg.norm(v[bad] - v[partner] * mir, axis=1)
        r = np.linalg.norm(v[samp] - v[p2] * mir, axis=1)
        print(f"{path}: defect L-vs-mirrored-R median {np.median(a):.4f} max {a.max():.4f} | "
              f"reference L/R pairs median {np.median(r):.4f} p95 {np.percentile(r, 95):.4f}")


if __name__ == "__main__":
    main()
