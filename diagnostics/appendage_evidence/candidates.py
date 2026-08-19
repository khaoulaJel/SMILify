"""Phase 2 -- candidate leg-length measurement definitions, scored identically to the baseline.

C0  joint-to-joint bone length (UNCHANGED). `seg_l_k_ti` (tibia: ti-joint to ta-joint) and
    `seg_l_k_ta` (tarsus: ta-joint to pt-joint), read straight off `measure.bone_table` /
    `measure.bone_lengths` -- the exact functions that produced the R=0.136 number. Not
    reimplemented; imported and filtered by `calib_features.block_of`.

C1  joint-CHAIN path length: sum of consecutive bone lengths along a chain, still built from
    `Jr @ verts` joint positions (same upstream data as C0), just combining more of them.
    - `chain_distal`  = seg_ti + seg_ta   (ti -> ta -> pt: same anatomical span as C0's two
      columns combined into one number -- the direct, like-for-like comparison to C0).
    - `chain_full`    = seg_co + seg_tr + seg_fe + seg_ti + seg_ta  (co -> pt: whole-leg length,
      a different anatomical quantity, included because Phase 1 found proximal joints are far
      better localised -- if error is scattered rather than systematic, folding more segments in
      could average some of it out. NOT a direct substitute for "distal leg length"; reported
      separately, never conflated with chain_distal in Phase 3's headline comparison.)

C2  geodesic / mesh-edge centerline length: shortest path over the mesh's own triangle-edge graph
    (`scipy.sparse.csgraph.dijkstra`, edge weight = current Euclidean edge length), walked across
    the leg's own dominantly-skinned surface patch. Two variants matching C1's spans:
    - `geo_distal` : endpoints anchored at the ti-joint and the pt-joint, path confined to the
      {ti, ta, pt} patch (confirmed a single connected component for every leg, 109-174 verts).
    - `geo_full`   : endpoints anchored at the co-joint and the pt-joint, path confined to the
      {co, tr, fe, ti, ta, pt} patch (confirmed connected, ~1500 verts for leg 1).

ENDPOINT SELECTION, stated precisely because it is the one free choice in C2 and the plan requires
it be justified and frozen. For each (leg, side, span), the proximal and distal endpoint VERTEX
INDICES are chosen ONCE from the TEMPLATE rest-pose geometry (`M['v_template']`, `J0 = Jr @
v_template`) and then reused, unchanged, for every specimen and for BOTH the fitted and the
ground-truth vertex arrays:
    proximal_idx = argmin_{v in patch} || V0[v] - J0[proximal_joint] ||   (patch vertex nearest,
                    in the template, to the joint marking the start of the span)
    distal_idx   = argmin_{v in patch} || V0[v] - J0[pt_joint] ||        (patch vertex nearest to
                    the pretarsus joint -- the true anatomical leg tip, for every span)
This is deterministic and identical across fit/ground-truth by construction: because correspondence
in this synthetic ceiling test is by vertex INDEX (fitted vertex i is supposed to land on ground
truth vertex i), freezing the two integer indices from the template is exactly "the same rule
applied to both" -- there is no per-specimen re-selection that could drift between fit and truth.
Mesh TOPOLOGY (which vertices are edge-adjacent) is also read once from the template faces and
reused for every specimen; only edge WEIGHTS (current Euclidean length) vary per specimen, which is
what makes it a geodesic on that specimen's actual geometry rather than the template's.
"""

import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from calib_features import block_of  # noqa: E402 -- reused to select the exact C0 columns

LEGS = (1, 2, 3)
SIDES = ("r", "l")
# spans: (label, proximal_joint_segment, chain_segments) -- chain_segments are the `seg_*` parent
# tags in the order they appear walking distal-ward, matching `measure.bone_table`'s naming
# (`seg_X` = the bone whose PROXIMAL joint is X).
SPANS = {
    "distal": ("ti", ("ti", "ta")),
    "full": ("co", ("co", "tr", "fe", "ti", "ta")),
}


# ---------------------------------------------------------------------------
# C0 -- baseline, unchanged
# ---------------------------------------------------------------------------
def measure_c0(J, bones):
    """Existing joint-to-joint tibia/tarsus bone lengths. `J` is (n, n_joints, 3)."""
    L = ms.bone_lengths(J, bones)
    out = {}
    for k, b in enumerate(bones):
        if block_of(b["name"]) == "leg_distal":
            out[f"c0_{b['name']}"] = L[:, k]
    return out


# ---------------------------------------------------------------------------
# C1 -- joint-chain path length
# ---------------------------------------------------------------------------
def measure_c1(J, bones, M):
    """Sum of consecutive `seg_*` bone lengths along each span, per leg/side."""
    jn = M["jnames"]
    L = ms.bone_lengths(J, bones)
    by_name = {b["name"]: k for k, b in enumerate(bones)}
    out = {}
    for span, (_, segs) in SPANS.items():
        for leg in LEGS:
            for side in SIDES:
                names = [f"seg_l_{leg}_{seg}_{side}" for seg in segs]
                idxs = [by_name[n] for n in names if n in by_name]
                if len(idxs) != len(names):
                    continue  # a segment was dropped upstream (e.g. drop_wing/drop_zero) -- skip
                out[f"c1_chain_{span}_l_{leg}_{side}"] = L[:, idxs].sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# C2 -- geodesic / mesh-edge centerline length
# ---------------------------------------------------------------------------
def _patch_and_graph(M, leg, side, joint_segs):
    """One leg/side/span's patch vertex indices + local edge list, from TEMPLATE topology only."""
    jn, dom = M["jnames"], M["dominant"]
    f = np.asarray(M["dd"]["f"])
    joint_ids = [j for j, n in enumerate(jn) if n in {f"l_{leg}_{s}_{side}" for s in joint_segs}]
    idx = np.where(np.isin(dom, joint_ids))[0]
    idxset = set(idx.tolist())
    pos = {v: i for i, v in enumerate(idx.tolist())}
    edges = set()
    for tri in f:
        tri = [int(v) for v in tri]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            if a in idxset and b in idxset:
                edges.add((pos[a], pos[b]))
    return idx, np.array(sorted(edges), dtype=np.int64)


def _endpoint(V0, J0, patch_global_idx, joint_id):
    d = np.linalg.norm(V0[patch_global_idx] - J0[joint_id], axis=1)
    return int(np.argmin(d))  # LOCAL index into patch_global_idx


def build_geodesic_cache(M):
    """Precompute, ONCE from the template, everything C2 needs per leg/side/span: the patch
    vertex indices, its local edge list, and the frozen proximal/distal endpoint local-indices.
    Reused unchanged for every specimen and for both fitted and ground-truth vertex arrays."""
    jn = M["jnames"]
    V0 = M["v_template"]
    J0 = M["Jr"] @ V0
    jid = {n: j for j, n in enumerate(jn)}
    cache = {}
    for span, (prox_seg, segs) in SPANS.items():
        joint_segs = sorted(set(segs) | {"pt"})  # every span's patch always includes the tip
        for leg in LEGS:
            for side in SIDES:
                patch_idx, edges = _patch_and_graph(M, leg, side, joint_segs)
                prox_j = jid[f"l_{leg}_{prox_seg}_{side}"]
                dist_j = jid[f"l_{leg}_pt_{side}"]
                prox_local = _endpoint(V0, J0, patch_idx, prox_j)
                dist_local = _endpoint(V0, J0, patch_idx, dist_j)
                cache[(span, leg, side)] = dict(
                    patch_idx=patch_idx, edges=edges, prox_local=prox_local, dist_local=dist_local
                )
    return cache


def _geodesic_batch(verts_batch, patch_idx, edges, prox_local, dist_local):
    """verts_batch: (n, V, 3). Returns (n,) shortest-path length, current-geometry edge weights."""
    n = verts_batch.shape[0]
    P = verts_batch[:, patch_idx, :]  # (n, n_patch, 3)
    n_patch = patch_idx.size
    out = np.full(n, np.nan)
    for i in range(n):
        w = np.linalg.norm(P[i, edges[:, 0]] - P[i, edges[:, 1]], axis=1)
        G = coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n_patch, n_patch))
        d = dijkstra(G, directed=False, indices=prox_local)
        out[i] = d[dist_local]
    return out


def dropout_vertices(M, cache, frac):
    """Global vertex indices to remove to simulate a damaged/missing distal leg, for every leg
    and side, at drop fraction `frac` (of the {ti,ta,pt} 'distal' patch's own vertex count).

    RULE, frozen from the template exactly like C2's endpoints (reused, not a new mechanism):
    rank each 'distal'-span patch vertex by its TEMPLATE geodesic distance from that span's own
    proximal anchor (`cache[('distal', leg, side)]`'s `prox_local`, already computed for C2), and
    drop the top `frac` fraction by that ranking -- i.e. the geometrically most-distal vertices,
    which is what "missing/damaged tibia-tarsus-pretarsus tip" means anatomically. Computed once
    on the template (pose-independent topology + rest-pose edge lengths), then the SAME vertex
    index set is dropped for every specimen -- this is a fixed corpus-construction choice, not a
    per-specimen re-selection, so drop30/drop60 mean the same physical thing for every specimen.
    """
    V0 = M["v_template"]
    out = {}
    for leg in LEGS:
        for side in SIDES:
            c = cache[("distal", leg, side)]
            patch_idx, edges, prox_local = c["patch_idx"], c["edges"], c["prox_local"]
            n_patch = patch_idx.size
            w = np.linalg.norm(V0[patch_idx[edges[:, 0]]] - V0[patch_idx[edges[:, 1]]], axis=1)
            G = coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n_patch, n_patch))
            d = dijkstra(G, directed=False, indices=prox_local)
            order = np.argsort(-d)  # most distal (largest geodesic distance) first
            n_drop = int(round(frac * n_patch))
            drop_local = order[:n_drop]
            out[(leg, side)] = patch_idx[drop_local]
    return out


def measure_c2(verts, cache):
    """verts: (n, V, 3), the full mesh (fitted or ground truth) -- SAME array C0/C1 are read from."""
    out = {}
    for (span, leg, side), c in cache.items():
        out[f"c2_geo_{span}_l_{leg}_{side}"] = _geodesic_batch(
            verts, c["patch_idx"], c["edges"], c["prox_local"], c["dist_local"]
        )
    return out


# ---------------------------------------------------------------------------
# entry point: every candidate, from the SAME verts array, in one call
# ---------------------------------------------------------------------------
def measure_all(verts, M, bones, geodesic_cache):
    """verts: (n, V, 3). Returns {column_name: (n,) array}, covering C0+C1+C2. Called identically
    on fitted verts and on ground-truth verts by the Phase 3 scorer -- this function never sees
    which one it was given, so it cannot introduce a fit-vs-truth asymmetry by construction."""
    J = ms.joints(verts, M["Jr"])
    out = {}
    out.update(measure_c0(J, bones))
    out.update(measure_c1(J, bones, M))
    out.update(measure_c2(verts, geodesic_cache))
    return out


if __name__ == "__main__":
    # smoke test: shapes and finiteness only, no scoring here (Phase 3's job)
    M = ms.load_model()
    bones = ms.bone_table(M)
    cache = build_geodesic_cache(M)
    print(f"geodesic cache: {len(cache)} (span, leg, side) entries")
    for key, c in list(cache.items())[:3]:
        print(f"  {key}: patch={c['patch_idx'].size} verts, {c['edges'].shape[0]} edges, "
              f"prox_local={c['prox_local']}, dist_local={c['dist_local']}")
    n_demo = 3
    verts_demo = np.tile(M["v_template"][None], (n_demo, 1, 1))
    cols = measure_all(verts_demo, M, bones, cache)
    print(f"\n{len(cols)} candidate columns; on the template repeated {n_demo}x (all rows equal):")
    for k, v in sorted(cols.items()):
        assert np.all(np.isfinite(v)), f"{k} has non-finite values"
        print(f"  {k:<28} {v[0]:.5f}")
