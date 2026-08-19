"""Shared scoring: identical for C0/C1/C2, and identical to `calib_features.py`'s own formulas
(copied verbatim from its `main()`, not reimplemented) so no candidate can look better merely
because it was scored a different way.

`score_run(run, corpus)` runs the FULL fitted-vertex array and the FULL ground-truth vertex array
(same per-specimen normalisation `calib_features.py` uses -- centre + divide by target .obj's own
max|coord|) through `candidates.measure_all` ONCE EACH, so every candidate is read off the same
two vertex arrays under the same normalisation, and never off a separately-defined "true length".
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
import candidates as cand  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402


def load_fit_and_truth(run, corpus, stage="Stage_3_deform_fine.npz"):
    d = np.load(os.path.join(MOON, "runs", run, stage))
    labels = [str(x) for x in d["labels"]]
    fitted = d["verts"].astype(np.float64)

    gt = np.load(os.path.join(MOON, corpus, "ground_truth.npz"))
    gtv_all, names = gt["verts"], [str(x) for x in gt["names"]]
    truth = []
    for lab in labels:
        stem = lab[:-4] if lab.endswith(".obj") else lab
        ov, _, _ = load_obj(os.path.join(MOON, corpus, f"{stem}.obj"), load_textures=False)
        ov = ov.numpy()
        c = ov.mean(0)
        truth.append((gtv_all[names.index(stem)] - c) / np.abs(ov - c).max())
    return fitted, np.stack(truth), labels


def score_columns(F, G):
    """F, G: {col: (n,) array}, SAME keys. Formula copied verbatim from calib_features.py's
    main() (R/snr/bias loop) -- not reimplemented, so scoring is identical to the project's
    existing reliability numbers."""
    out = {}
    for c in F:
        f, g = F[c], G[c]
        R = float(np.corrcoef(f, g)[0, 1]) if f.std() > 1e-12 and g.std() > 1e-12 else np.nan
        snr = float(g.std() / max((f - g).std(), 1e-12))
        bias = float(np.median(np.abs((f - g) / np.maximum(np.abs(g), 1e-12))))
        out[c] = dict(R=R, snr=snr, bias=bias)
    return out


def score_run(run, corpus, M=None, bones=None, geodesic_cache=None):
    M = M or ms.load_model()
    bones = bones if bones is not None else ms.bone_table(M)
    geodesic_cache = geodesic_cache or cand.build_geodesic_cache(M)

    fitted, truth, labels = load_fit_and_truth(run, corpus)
    F = cand.measure_all(fitted, M, bones, geodesic_cache)
    G = cand.measure_all(truth, M, bones, geodesic_cache)
    assert set(F) == set(G), "fit/truth column mismatch -- candidates.py must be symmetric"
    scores = score_columns(F, G)
    return scores, F, G, labels


def summarize_by_candidate(scores):
    """Median R per candidate family (c0/c1_chain_distal/c1_chain_full/c2_geo_distal/
    c2_geo_full), the same block-median idea `calib_features.py` uses per anatomical block."""
    families = {}
    for c in scores:
        if c.startswith("c0_"):
            fam = "C0_baseline"
        elif c.startswith("c1_chain_distal"):
            fam = "C1_chain_distal"
        elif c.startswith("c1_chain_full"):
            fam = "C1_chain_full"
        elif c.startswith("c2_geo_distal"):
            fam = "C2_geodesic_distal"
        elif c.startswith("c2_geo_full"):
            fam = "C2_geodesic_full"
        else:
            fam = "other"
        families.setdefault(fam, []).append(c)
    out = {}
    for fam, cols in families.items():
        Rs = [scores[c]["R"] for c in cols]
        snrs = [scores[c]["snr"] for c in cols]
        biases = [scores[c]["bias"] for c in cols]
        out[fam] = dict(
            n=len(cols),
            median_R=float(np.nanmedian(Rs)),
            median_snr=float(np.nanmedian(snrs)),
            median_bias=float(np.nanmedian(biases)),
        )
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="SYN_clean_w5")
    ap.add_argument("--corpus", default="synth_clean")
    args = ap.parse_args()

    scores, F, G, labels = score_run(args.run, args.corpus)
    fam = summarize_by_candidate(scores)
    print(f"=== {args.run} / {args.corpus}  (n={len(labels)} specimens) ===")
    print(f"{'family':<22}{'n cols':>7}{'median R':>10}{'median SNR':>12}{'median |bias|':>15}")
    for k in ("C0_baseline", "C1_chain_distal", "C1_chain_full", "C2_geodesic_distal", "C2_geodesic_full"):
        if k in fam:
            v = fam[k]
            print(f"{k:<22}{v['n']:>7}{v['median_R']:>10.3f}{v['median_snr']:>12.2f}{100*v['median_bias']:>14.1f}%")
