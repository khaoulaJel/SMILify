"""Per-leg geodesic distance tables on the TEMPLATE mesh, cached to disk.

WHY: the categorical confusion matrix (confusion.py) says whether a scan point's matched
segment equalled its true segment -- a binary hit/miss. It cannot distinguish "matched to the
immediately adjacent segment" from "matched to a segment three hops away", i.e. it cannot see
partial credit. Geodesic (mesh-surface) distance between the TRUE vertex and the MATCHED vertex
answers that: if split_segments moves errors from "far" to "adjacent but still wrong", that is
real progress the categorical accuracy number alone cannot show.

SCOPE: restricted to WITHIN-LEG pairs (both true and matched vertex on the SAME leg), because
that is exactly the population `within_leg_segment_confusion`/`within_leg_segment_geodesic`
already isolate (points whose leg-level match was already correct) -- a cross-leg swap is not
"close in a few mm" in any useful sense, it is a different structure, so it is out of scope
here by design, not by oversight.

METHOD: build the template's edge graph (one entry per undirected mesh edge, weighted by
Euclidean edge length -- the standard graph-distance approximation to true geodesic distance,
exact enough at this vertex density for a "how many segments away" read), then run
scipy.sparse.csgraph.dijkstra ONCE per leg with `indices` = every vertex belonging to that leg
(all six segments), against the FULL template graph (not a leg-only induced subgraph, so a
shortest path is never artificially forced to stay on the leg if it genuinely wouldn't need to
-- though at this scale it always does; only the leg x leg submatrix is kept). Six legs, each
600-1150 vertices: cheap, seconds not minutes. Cached to `out/geodesic_tables.npz` (per leg,
the sorted global vertex-index array plus its dense distance submatrix) so repeat audits don't
recompute it.
"""

import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "out", "geodesic_tables.npz")


def build_edge_graph(verts, faces):
    """Sparse (n_verts, n_verts) graph, one entry per undirected mesh edge, weighted by
    Euclidean edge length. `faces` is the TEMPLATE's own face array (all callers here use
    template geometry only -- geodesic distance is a property of the template mesh, not of any
    individual fitted/target instance).
    """
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)  # one row per undirected edge, dedupes the two faces sharing it
    d = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    n = len(verts)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    data = np.concatenate([d, d])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def build_leg_tables(template_verts, template_faces, vlabels):
    """{leg_name: (vertex_ids (sorted global template indices), dist (len x len) submatrix)}."""
    from labels import LEGS

    graph = build_edge_graph(template_verts, template_faces)
    leg_id = vlabels["leg_id"]
    tables = {}
    for leg_name in LEGS:
        vidx = np.sort(np.where(leg_id == leg_name)[0])
        dist_full = dijkstra(graph, indices=vidx, directed=False)  # (len(vidx), n_verts)
        tables[leg_name] = (vidx, dist_full[:, vidx])
    return tables


def load_or_build_leg_tables(template_verts, template_faces, vlabels, cache_path=CACHE_PATH):
    from labels import LEGS

    if os.path.isfile(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        if int(d["n_verts"]) == len(template_verts):
            return {leg: (d[f"{leg}_vidx"], d[f"{leg}_dist"]) for leg in LEGS}
    tables = build_leg_tables(template_verts, template_faces, vlabels)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = dict(n_verts=len(template_verts))
    for leg, (vidx, dist) in tables.items():
        payload[f"{leg}_vidx"] = vidx
        payload[f"{leg}_dist"] = dist
    np.savez(cache_path, **payload)
    return tables


class LegGeodesicLookup:
    """dist(leg_name, true_vertex_idx_array, matched_vertex_idx_array) -> (N,) distances."""

    def __init__(self, tables):
        self.tables = {}
        for leg, (vidx, dist) in tables.items():
            local = {v: i for i, v in enumerate(vidx)}
            self.tables[leg] = (local, dist)

    def dist(self, leg_names, true_vidx, matched_vidx):
        out = np.full(len(true_vidx), np.nan, dtype=np.float64)
        for leg_name in np.unique(leg_names):
            m = leg_names == leg_name
            local, dist = self.tables[leg_name]
            ti = np.array([local[v] for v in true_vidx[m]])
            mi = np.array([local[v] for v in matched_vidx[m]])
            out[m] = dist[ti, mi]
        return out
