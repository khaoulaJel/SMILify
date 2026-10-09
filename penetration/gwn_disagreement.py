"""
Generalized, read-only GWN-vs-proximity disagreement instrument.

Generalizes the validated 3-specimen/50-specimen gaster<->legs-only diagnostic
(diagnostics/gwn_diagnostic.py, diagnostics/gwn_diagnostic_full.py -- Step 6 of
scripts/penetration_joint_study/FINDINGS.md, Pearson r=0.947 between disagreement
asymmetry and count-worsening) to ALL non-adjacent part pairs, both directions,
so it can run as a standalone eval instrument on any Stage_3-style fit rather
than only the two specific gaster/legs runs the original probe compared.

What it measures: `fitter_3d/penetration_loss.py`'s proximity-based inside/outside
test (nearest-triangle-CENTROID matching, an explicitly acknowledged approximation)
is compared, per query vertex, against a TRUE generalized winding-number
inside/outside signal (Jacobson et al. 2013) computed on a capped (closed) copy of
the target part's surface. Vertices are only counted if the winding-number margin
is "trustworthy" (|w - 0.5| exceeds K_MARGIN times that vertex's own local mean
incident edge length -- edge lengths vary ~30x across an ant mesh, so a fixed
absolute margin would be meaningless near antenna/mandible tips).

Manifoldness gate: NOT re-run here. `diagnostics/manifoldness_check.py` established
(and `diagnostics/moonshot/refute_gwn_validity_PROBE.py` cross-checked) that this is
a structural invariant of the SMIL model, not a per-run risk: pose/shape/deform_verts
only ever move vertex POSITIONS, never face-vertex incidence, so a fit produced by
this pipeline cannot become non-manifold regardless of deformation. Re-checking it
on every call would be pure overhead for a fact that cannot change. If this module
is ever pointed at meshes from a different pipeline (e.g. raw scan reconstructions),
re-run the manifoldness gate first -- see refute_gwn_validity_PROBE.py for why scan
meshes are a real risk where SMIL fits are not.

CPU-only, read-only. Does not touch any training loss or weight.
"""

from collections import defaultdict

import numpy as np
import torch

K_MARGIN_DEFAULT = 2.5
TAU_FRACTION_DEFAULT = 0.03
MAX_DEPTH_FRACTION_DEFAULT = 0.08


def find_boundary_loops(faces_subset):
    """Boundary edge loops of a face subset (a part cut from the full mesh)."""
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces_subset):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (a, b) if a < b else (b, a)
            edge_faces[key].append(fi)
    boundary_edges = [k for k, v in edge_faces.items() if len(v) == 1]
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)
    visited = set()
    loops = []
    for a, b in boundary_edges:
        e0 = frozenset((a, b))
        if e0 in visited:
            continue
        loop = [a, b]
        visited.add(e0)
        prev, curr = a, b
        while True:
            nxt = None
            for n in adj[curr]:
                if n == prev:
                    continue
                ee = frozenset((curr, n))
                if ee not in visited:
                    nxt = n
                    break
            if nxt is None:
                break
            visited.add(frozenset((curr, nxt)))
            loop.append(nxt)
            prev, curr = curr, nxt
            if curr == a:
                break
        if loop[0] == loop[-1]:
            loop = loop[:-1]
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def cap_part_mesh(verts, faces_subset):
    """Fan-triangulate every boundary loop from its centroid, oriented outward,
    so a single anatomical part (open where it attached to its neighbors) becomes
    a closed surface -- required for winding_number() to give a valid signal.
    This capping is a real geometric approximation local to the cap itself."""
    part_vert_ids = np.unique(faces_subset)
    part_centroid = verts[part_vert_ids].mean(axis=0)
    loops = find_boundary_loops(faces_subset)
    all_verts = verts.copy()
    new_faces = [tuple(f) for f in faces_subset]
    for loop in loops:
        pts = verts[loop]
        loop_centroid = pts.mean(axis=0)
        cidx = all_verts.shape[0]
        all_verts = np.vstack([all_verts, loop_centroid[None, :]])
        outward_ref = loop_centroid - part_centroid
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            tri_normal = np.cross(verts[b] - verts[a], loop_centroid - verts[a])
            if np.dot(tri_normal, outward_ref) < 0:
                new_faces.append((b, a, cidx))
            else:
                new_faces.append((a, b, cidx))
    return all_verts, np.array(new_faces, dtype=np.int64)


def per_vertex_edge_length(verts, faces):
    acc = defaultdict(list)
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            length = np.linalg.norm(verts[a] - verts[b])
            acc[a].append(length)
            acc[b].append(length)
    out = np.zeros(verts.shape[0])
    for v, lens in acc.items():
        out[v] = np.mean(lens)
    return out


def _disagreement_rate_one_direction(
    verts, idx_query, surface_faces_orig, capped_V, capped_F, vert_edge_len, tau, max_depth,
    winding_number_fn, k_margin,
):
    """One (query part -> surface part) direction. Returns
    (disagreement_rate or None, n_trustworthy, n_query)."""
    from fitter_3d.penetration_loss import _directional_penalty  # local import: torch/fitter_3d dep

    verts_t = torch.tensor(verts, dtype=torch.float32).unsqueeze(0)
    tau_t = torch.tensor([tau])
    max_depth_t = torch.tensor([max_depth])
    idx_q_t = torch.tensor(idx_query, dtype=torch.long)
    faces_surf_t = torch.tensor(surface_faces_orig, dtype=torch.long)
    _, diag = _directional_penalty(verts_t, idx_q_t, faces_surf_t, tau_t, max_depth_t)
    proximity_penetrating = diag["penetrating_mask"][0].numpy()

    query_pts = verts[idx_query]
    w = winding_number_fn(query_pts, capped_V, capped_F)
    margin = np.abs(w - 0.5)
    true_inside = w > 0.5
    local_edge = vert_edge_len[idx_query]
    trustworthy = margin > (k_margin * local_edge)

    n_trust = int(trustworthy.sum())
    if n_trust == 0:
        return None, 0, len(idx_query)
    disagreement = (proximity_penetrating != true_inside) & trustworthy
    return float(disagreement.sum()) / n_trust, n_trust, len(idx_query)


def compute_specimen_gwn_disagreement(
    verts,
    faces,
    part_vertex_indices,
    part_faces,
    non_adjacent_pairs,
    winding_number_fn,
    k_margin=K_MARGIN_DEFAULT,
    tau_fraction=TAU_FRACTION_DEFAULT,
    max_depth_fraction=MAX_DEPTH_FRACTION_DEFAULT,
):
    """GWN-vs-proximity disagreement rate for every (part_a, part_b) pair in
    non_adjacent_pairs, both directions, for ONE specimen's fitted vertices.

    Returns a list of row dicts: part_a, part_b, direction ("A_into_B"/"B_into_A"),
    disagreement_rate (None if no trustworthy query vertices), n_trustworthy, n_query,
    trustworthy_frac.

    Capped meshes are computed once per PART (not per pair) and cached, since the
    same part surface is reused across every pair it appears in.
    """
    bbox_diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    tau = tau_fraction * bbox_diag
    max_depth = max_depth_fraction * bbox_diag
    vert_edge_len = per_vertex_edge_length(verts, faces)

    cap_cache = {}

    def capped(part_name):
        if part_name not in cap_cache:
            cap_cache[part_name] = cap_part_mesh(verts, part_faces[part_name])
        return cap_cache[part_name]

    rows = []
    for part_a, part_b in non_adjacent_pairs:
        capped_V_b, capped_F_b = capped(part_b)
        capped_V_a, capped_F_a = capped(part_a)

        rate_AtoB, ntrust_AtoB, n_AtoB = _disagreement_rate_one_direction(
            verts, part_vertex_indices[part_a], part_faces[part_b], capped_V_b, capped_F_b,
            vert_edge_len, tau, max_depth, winding_number_fn, k_margin,
        )
        rate_BtoA, ntrust_BtoA, n_BtoA = _disagreement_rate_one_direction(
            verts, part_vertex_indices[part_b], part_faces[part_a], capped_V_a, capped_F_a,
            vert_edge_len, tau, max_depth, winding_number_fn, k_margin,
        )
        rows.append({
            "part_a": part_a, "part_b": part_b, "direction": "A_into_B",
            "disagreement_rate": rate_AtoB, "n_trustworthy": ntrust_AtoB, "n_query": n_AtoB,
            "trustworthy_frac": (ntrust_AtoB / n_AtoB) if n_AtoB else 0.0,
        })
        rows.append({
            "part_a": part_a, "part_b": part_b, "direction": "B_into_A",
            "disagreement_rate": rate_BtoA, "n_trustworthy": ntrust_BtoA, "n_query": n_BtoA,
            "trustworthy_frac": (ntrust_BtoA / n_BtoA) if n_BtoA else 0.0,
        })
    return rows
