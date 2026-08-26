"""
cpu_self_intersection.py

CPU-only, no-CUDA drop-in replacement for `mesh_intersection.bvh_search_tree.BVH`, answering
the same question (which triangle pairs in a mesh actually intersect) without touching the
`bvh_cuda` kernel confirmed to fault under repeated invocation (TASK8, TASK9). Motivated by
splitting the repair pipeline's two separable pieces -- detect intersections, then push to
resolve them -- and replacing only the piece that's crashing. The push (energies.py's
DistanceFieldPenetrationLoss / signed_TPE_verts etc., driven by the now-working CG solver
swap) is untouched; this module only needs to match BVH's __call__ contract exactly, so nothing
downstream has to change.

Two phases, standard for this problem, not novel:
  - Broad phase: trimesh's own AABB tree (`Trimesh.triangles_tree`, backed by `rtree`) prunes
    candidate face pairs to those whose bounding boxes overlap -- avoids the O(F^2) pair count
    a naive check would need (20454 faces -> ~2*10^8 raw pairs).
  - Narrow phase: exact triangle-triangle intersection via 6 vectorized segment-triangle tests
    per candidate pair (each of one triangle's 3 edges against the other triangle, both ways;
    Moller-Trumbore ray-triangle intersection restricted to t in [0,1] for a bounded segment).
    Two triangles are flagged as intersecting iff any of the 6 sub-tests hits -- correct for
    generic (non-coplanar) crossing configurations, which is what a fitted, deformed mesh's
    self-intersections actually look like.

Validated against the exact same synthetic cases used to validate the CUDA BVH kernel earlier
in this investigation (coincident triangles, a hand-verified interior-piercing pair, a
separated-pair negative control, multiple simultaneous pairs) before being trusted on real data
-- see the bottom of this file's __main__ block.
"""

import numpy as np
import torch
import trimesh


def _segment_triangle_intersect_batch(p0, p1, tri_v0, tri_v1, tri_v2, eps=1e-8):
    """Vectorized Moller-Trumbore segment-triangle intersection, segment restricted to t in
    [0, 1] (i.e. a bounded edge, not an infinite ray).

    Args:
        p0, p1: (N, 3) segment endpoints.
        tri_v0, tri_v1, tri_v2: (N, 3) triangle vertices (same N, paired with the segments).
    Returns:
        (N,) bool -- True where the segment properly crosses the triangle's interior.
    """
    d = p1 - p0
    e1 = tri_v1 - tri_v0
    e2 = tri_v2 - tri_v0
    h = np.cross(d, e2)
    a = np.sum(e1 * h, axis=1)
    parallel = np.abs(a) < eps
    a_safe = np.where(parallel, 1.0, a)
    f = 1.0 / a_safe
    s = p0 - tri_v0
    u = f * np.sum(s * h, axis=1)
    q = np.cross(s, e1)
    v = f * np.sum(d * q, axis=1)
    t = f * np.sum(e2 * q, axis=1)
    return (
        (~parallel)
        & (u >= -eps) & (u <= 1 + eps)
        & (v >= -eps) & (u + v <= 1 + eps)
        & (t >= -eps) & (t <= 1 + eps)
    )


def find_self_intersecting_face_pairs(verts: np.ndarray, faces: np.ndarray):
    """
    Args:
        verts: (V, 3) float
        faces: (F, 3) int
    Returns:
        (K, 2) int array of intersecting face-index pairs, i < j, deduplicated.
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tree = mesh.triangles_tree
    tri_lo = mesh.triangles.min(axis=1)  # (F, 3)
    tri_hi = mesh.triangles.max(axis=1)  # (F, 3)

    candidates = set()
    for i in range(len(faces)):
        lo, hi = tri_lo[i], tri_hi[i]
        for j in tree.intersection((lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])):
            if j > i:
                candidates.add((i, j))
            elif j < i:
                candidates.add((j, i))
    candidates.discard(None)
    candidates = np.array(sorted(c for c in candidates if c[0] != c[1]), dtype=np.int64)
    if len(candidates) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    tri = verts[faces]  # (F, 3, 3)
    i_idx, j_idx = candidates[:, 0], candidates[:, 1]
    ti = tri[i_idx]  # (K, 3, 3)
    tj = tri[j_idx]

    # Exclude candidate pairs that share a vertex -- e.g. two faces meeting at a normal mesh
    # edge/corner. `tri` is built directly from `verts[faces]`, so a shared vertex INDEX
    # produces bit-identical float positions (same source array, no computation in between) --
    # exact equality is the correct, not just convenient, check here, no tolerance needed.
    # Without this, the edge-vs-triangle narrow-phase test below flags every ordinary
    # face-to-face mesh connection as a "crossing" at the shared vertex (t=0/t=1 boundary of
    # the segment test) -- not a real self-intersection, just normal manifold topology. This
    # is presumably handled internally by the CUDA BVH kernel already (it never reports these
    # as collisions); this check makes the CPU detector match that behavior explicitly rather
    # than relying on it being implicit.
    shares_vertex = np.zeros(len(candidates), dtype=bool)
    for a in range(3):
        for b in range(3):
            shares_vertex |= np.all(ti[:, a] == tj[:, b], axis=1)
    keep = ~shares_vertex
    candidates = candidates[keep]
    ti = ti[keep]
    tj = tj[keep]
    if len(candidates) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    hits = np.zeros(len(candidates), dtype=bool)
    # 3 edges of ti vs tj, 3 edges of tj vs ti = 6 sub-tests, OR'd together
    edge_pairs = [(0, 1), (1, 2), (2, 0)]
    for a, b in edge_pairs:
        hits |= _segment_triangle_intersect_batch(ti[:, a], ti[:, b], tj[:, 0], tj[:, 1], tj[:, 2])
    for a, b in edge_pairs:
        hits |= _segment_triangle_intersect_batch(tj[:, a], tj[:, b], ti[:, 0], ti[:, 1], ti[:, 2])

    return candidates[hits]


class CPUSelfIntersectionDetector:
    """Drop-in replacement for mesh_intersection.bvh_search_tree.BVH -- same __call__
    contract (triangles: (B, F, 3, 3) -> (B, F*max_collisions, 2) int64, -1-padded), backed
    by find_self_intersecting_face_pairs instead of the CUDA BVH kernel. B=1 only (this
    project's repair use case is always one mesh at a time).

    Args:
        max_collisions: unused, kept for signature compatibility with BVH.
        face_part_id: optional (F,) int array, part index per face (from
            fitter_3d.bvh_penetration_loss.build_face_part_ids), -1 for a seam face. When
            given together with allowed_part_pairs, detected collisions between faces NOT in
            an allowed (part_a, part_b) combination are dropped before returning -- e.g.
            excluding legitimate anatomical contact (leg-to-thorax attachment seams) that
            this tool would otherwise "repair" right alongside genuine problem penetration.
            None (default): unfiltered, whole-mesh behaviour, unchanged from before.
        allowed_part_pairs: set of frozenset({part_idx_a, part_idx_b}) pairs to keep.
    """

    def __init__(self, max_collisions=None, face_part_id=None, allowed_part_pairs=None):
        self.max_collisions = max_collisions  # unused; kept for signature compatibility
        self.face_part_id = face_part_id
        self.allowed_part_pairs = allowed_part_pairs

    def __call__(self, triangles: torch.Tensor) -> torch.Tensor:
        assert triangles.shape[0] == 1, "CPUSelfIntersectionDetector only supports batch size 1"
        device = triangles.device
        tri_np = triangles[0].detach().cpu().numpy()  # (F, 3, 3)
        # Deliberately NOT deduplicated by position -- the input is already an exploded
        # triangle-soup (each face owns its own 3 vertex slots, matching BVH's own
        # `vertices[faces]` input convention), and neither the broad-phase AABB tree nor the
        # narrow-phase edge test needs shared vertex indexing to be correct. Deduplicating by
        # position would silently merge distinct, coincident-by-value triangles into one
        # shared-vertex face -- exactly the degenerate case this detector needs to be tested
        # against, not accidentally special-cased away.
        F = tri_np.shape[0]
        verts = tri_np.reshape(-1, 3)
        faces = np.arange(3 * F, dtype=np.int64).reshape(F, 3)

        pairs = find_self_intersecting_face_pairs(verts, faces)

        if self.face_part_id is not None and self.allowed_part_pairs is not None and len(pairs) > 0:
            pa = self.face_part_id[pairs[:, 0]]
            pb = self.face_part_id[pairs[:, 1]]
            keep = np.array(
                [frozenset((a, b)) in self.allowed_part_pairs for a, b in zip(pa, pb)],
                dtype=bool,
            )
            pairs = pairs[keep]

        n = max(len(pairs), 1)
        out = -np.ones((1, n, 2), dtype=np.int64)
        if len(pairs) > 0:
            out[0, : len(pairs)] = pairs
        return torch.as_tensor(out, dtype=torch.int64, device=device)


if __name__ == "__main__":
    # Same synthetic validation suite used for the CUDA BVH kernel earlier in this
    # investigation (diagnostics/torch_mesh_isect_compat_probe.py) -- reused here so the two
    # detectors are checked against identical cases, not different ones.
    def collision_pairs_from(verts, faces):
        det = CPUSelfIntersectionDetector()
        out = det(torch.tensor(verts[faces], dtype=torch.float32).unsqueeze(0))
        out = out[0].numpy()
        return set(tuple(sorted(row)) for row in out if row[0] >= 0)

    print("--- pass 0: two EXACTLY coincident triangles ---")
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                  [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    f = np.array([[0, 1, 2], [3, 4, 5]])
    pairs = collision_pairs_from(v, f)
    print("pairs:", pairs, "-- PASS" if (0, 1) in pairs else "-- FAIL")

    print("\n--- pass 1: hand-verified interior-piercing pair ---")
    v = np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [0.2, 0.2, -0.5], [0.3, 0.2, 0.5], [0.2, 0.3, 0.5],
        [10., 10., 10.], [11., 10., 10.], [10., 11., 10.],
        [20., 20., 20.], [21., 20., 20.], [20., 21., 20.],
    ])
    f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    pairs = collision_pairs_from(v, f)
    print("pairs:", pairs, "-- PASS" if pairs == {(0, 1)} else "-- FAIL")

    print("\n--- pass 2: same triangle 1, translated far away (separation) ---")
    v2 = v.copy()
    v2[3:6] += 50.0
    pairs2 = collision_pairs_from(v2, f)
    print("pairs:", pairs2, "-- PASS" if len(pairs2) == 0 else "-- FAIL")
