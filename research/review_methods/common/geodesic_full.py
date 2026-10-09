"""Full-surface geodesic correspondence error on the template (Kim, Lipman & Funkhouser 2011 protocol).

The existing `diagnostics/correspondence_accuracy/geodesic.py` is deliberately WITHIN-LEG only, so it
cannot score cross-leg errors -- the dominant real-scan failure (Q3a A2). The template is a single
connected component (verified: 1 component, 10,235 vertices), so full-surface geodesic distance is
defined for every pair. Same graph construction as geodesic.py (edge graph weighted by Euclidean
edge length, Dijkstra), all pairs, cached as float32.

Error is reported normalised by sqrt(template surface area) (= 1.165), as in the protocol, and as
a cumulative curve: fraction of points with normalised error <= t.

Validation (run as __main__): on the leg x leg blocks, this matrix must equal geodesic.py's cached
within-leg tables (same graph, same algorithm), up to float32 rounding.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "correspondence_accuracy"))
CACHE = "/hpcwork/nao48500/review_methods/geodesic_full_f32.npy"


def template_area(V, F):
    return float(np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum() / 2)


def load(dd=None):
    """Return (D (V,V) float32 memmap of geodesic distances, sqrt_area)."""
    import anatomy_proxy as ap
    dd = dd or ap.load_dd()
    V = np.asarray(dd["v_template"], float)
    F = np.asarray(dd["f"]).astype(np.int64)
    if not os.path.isfile(CACHE):
        from geodesic import build_edge_graph
        from scipy.sparse.csgraph import dijkstra
        G = build_edge_graph(V, F)
        D = np.empty((len(V), len(V)), np.float32)
        for s in range(0, len(V), 512):
            D[s:s + 512] = dijkstra(G, directed=False, indices=np.arange(s, min(s + 512, len(V))))
        assert np.isfinite(D).all(), "template graph is disconnected"
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.save(CACHE, D)
    return np.load(CACHE, mmap_mode="r"), float(np.sqrt(template_area(V, F)))


def normalised_error(D, sqrt_area, true_idx, pred_idx):
    return np.asarray(D[np.asarray(true_idx), np.asarray(pred_idx)], float) / sqrt_area


def cumulative_curve(err, thresholds=np.linspace(0, 0.25, 26)):
    err = np.asarray(err)
    return thresholds, np.array([(err <= t).mean() for t in thresholds])


if __name__ == "__main__":
    import anatomy_proxy as ap
    from labels import vertex_labels, LEGS
    from geodesic import load_or_build_leg_tables
    dd = ap.load_dd()
    D, sa = load(dd)
    print(f"cached {CACHE}: {D.shape}, sqrt(area) = {sa:.4f}, max geodesic {float(np.max(D)):.4f}")
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    vl = vertex_labels(names, np.asarray(dd["weights"]).argmax(1))
    tabs = load_or_build_leg_tables(np.asarray(dd["v_template"], float), np.asarray(dd["f"]).astype(np.int64), vl)
    worst = 0.0
    for leg in LEGS:
        vidx, dist = tabs[leg]
        worst = max(worst, float(np.abs(np.asarray(D[np.ix_(vidx, vidx)], float) - dist).max()))
    print(f"validation vs geodesic.py within-leg tables: max |diff| = {worst:.2e}")
    assert worst < 1e-5, "full geodesic disagrees with geodesic.py"
