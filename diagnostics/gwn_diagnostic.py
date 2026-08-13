"""
Diagnostic: does the CURRENT proximity-based penetration loss's implied inside/outside
verdict agree with a TRUE generalized-winding-number inside/outside signal, on the
directional-asymmetry pair (gaster <-> legs), gentle arm, 3-specimen set?

Methodology notes (read before trusting the numbers):
  - winding_number() (reused from diagnostics/moonshot/joint_placement_common.py, pure
    numpy, Jacobson et al. 2013) needs a CLOSED mesh. A single anatomical part cut from
    the full mesh has open boundary loops wherever it attached to neighboring parts, so
    each part is capped (fan-triangulation from each boundary loop's centroid) before
    being used as the "true surface" -- this is a real geometric approximation at the
    cap itself (query vertices very close to the joint/cap could be biased), not exact.
  - Manifoldness of the FULL fitted mesh was already confirmed clean (see manifoldness_check.py
    output) -- this diagnostic caps PART subsets specifically for closure, a separate step.
  - "Trustworthy" band is edge-length-normalized, not a fixed absolute margin: a query
    vertex's winding-number margin |w - 0.5| must exceed k * (that vertex's own local
    mean incident edge length on the FULL mesh) to count as trustworthy. k=2.5 (midpoint
    of the suggested 2-3x range).
"""
import sys
sys.path.insert(0, "/p/home/jusers/jellal1/jureca/SMILify")
sys.path.insert(0, "/p/home/jusers/jellal1/jureca/SMILify/diagnostics/moonshot")

import numpy as np
import torch
from collections import defaultdict

from joint_placement_common import winding_number
from fitter_3d.trainer import SMAL3DFitter
from fitter_3d.part_groups import get_part_vertex_indices, PART_GROUPS_COARSE
from fitter_3d.penetration_loss import _build_part_faces, _directional_penalty

K_MARGIN = 2.5
SPECIMENS = [
    "Acanthostichus_aff.brevicornis_CASENT0744328_processed.obj",
    "Acromyrmex_coronatus_CASENT0744365_processed.obj",
    "Solenopsis_invicta_CASENT0744384_processed.obj",
]


def find_boundary_loops(faces_subset):
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
    """Closes a part's open boundary loops with a fan triangulation from each loop's
    centroid, oriented outward (away from the part's own vertex centroid) so the
    capped mesh is consistently oriented for winding_number."""
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
    return all_verts, np.array(new_faces, dtype=np.int64), len(loops)


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


def main():
    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].cpu().numpy()
    part_vertex_indices = get_part_vertex_indices(PART_GROUPS_COARSE)
    part_faces = _build_part_faces(faces, part_vertex_indices)

    idx_gaster = part_vertex_indices["gaster"]
    idx_legs = part_vertex_indices["legs"]
    faces_gaster = part_faces["gaster"]
    faces_legs = part_faces["legs"]

    print("Capping part meshes (gaster, legs)...")
    # capped ONCE per specimen (verts differ per specimen), so build inside the specimen loop

    gentle = np.load("fit3d_results_all_gentle/Stage_3_deform_fine.npz", allow_pickle=True)
    labels = list(gentle["labels"])

    overall_rows = []

    for spec in SPECIMENS:
        i = labels.index(spec)
        verts = gentle["verts"][i]  # (V, 3)
        bbox_diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
        tau = 0.03 * bbox_diag
        max_depth = 0.08 * bbox_diag

        vert_edge_len = per_vertex_edge_length(verts, faces)

        gaster_capped_V, gaster_capped_F, n_loops_gaster = cap_part_mesh(verts, faces_gaster)
        legs_capped_V, legs_capped_F, n_loops_legs = cap_part_mesh(verts, faces_legs)
        # sanity: capped mesh should itself be closed now
        def boundary_count(V, F):
            ef = defaultdict(int)
            for f in F:
                for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                    k = (a, b) if a < b else (b, a)
                    ef[k] += 1
            return sum(1 for c in ef.values() if c != 2)

        print(f"\n=== {spec} ===")
        print(f"  gaster capped: {n_loops_gaster} loop(s) filled, remaining bad edges: "
              f"{boundary_count(gaster_capped_V, gaster_capped_F)}")
        print(f"  legs capped:   {n_loops_legs} loop(s) filled, remaining bad edges: "
              f"{boundary_count(legs_capped_V, legs_capped_F)}")

        verts_t = torch.tensor(verts, dtype=torch.float32).unsqueeze(0)  # (1, V, 3)
        tau_t = torch.tensor([tau])
        max_depth_t = torch.tensor([max_depth])

        for direction, idx_query, surface_faces_orig, capped_V, capped_F in [
            ("A_into_B (gaster verts vs legs surface)", idx_gaster, faces_legs, legs_capped_V, legs_capped_F),
            ("B_into_A (legs verts vs gaster surface)", idx_legs, faces_gaster, gaster_capped_V, gaster_capped_F),
        ]:
            idx_q_t = torch.tensor(idx_query, dtype=torch.long)
            faces_surf_t = torch.tensor(surface_faces_orig, dtype=torch.long)
            _, diag = _directional_penalty(verts_t, idx_q_t, faces_surf_t, tau_t, max_depth_t)
            proximity_penetrating = diag["penetrating_mask"][0].numpy()  # (Nq,) bool, current loss's verdict

            query_pts = verts[idx_query]  # (Nq, 3)
            w = winding_number(query_pts, capped_V, capped_F)
            margin = np.abs(w - 0.5)
            true_inside = w > 0.5

            local_edge = vert_edge_len[idx_query]
            trustworthy = margin > (K_MARGIN * local_edge)

            n_total = len(idx_query)
            n_trustworthy = int(trustworthy.sum())
            disagreement = proximity_penetrating != true_inside
            n_disagree_trustworthy = int((disagreement & trustworthy).sum())
            n_prox_pen_trustworthy = int((proximity_penetrating & trustworthy).sum())

            print(f"  [{direction}]")
            print(f"    n_query={n_total}  trustworthy={n_trustworthy} ({100*n_trustworthy/n_total:.1f}%)  "
                  f"ambiguous={n_total-n_trustworthy} ({100*(n_total-n_trustworthy)/n_total:.1f}%)")
            print(f"    proximity-flagged-penetrating: {int(proximity_penetrating.sum())} "
                  f"(of which trustworthy: {n_prox_pen_trustworthy})")
            if n_trustworthy > 0:
                print(f"    disagreement among TRUSTWORTHY: {n_disagree_trustworthy} "
                      f"({100*n_disagree_trustworthy/n_trustworthy:.2f}% of trustworthy)")
            else:
                print("    disagreement among TRUSTWORTHY: n/a (zero trustworthy points)")

            overall_rows.append({
                "specimen": spec, "direction": direction, "n_total": n_total,
                "n_trustworthy": n_trustworthy, "n_disagree_trustworthy": n_disagree_trustworthy,
                "n_prox_pen": int(proximity_penetrating.sum()),
            })

    print("\n=== OVERALL (pooled across specimens/directions) ===")
    tot_n = sum(r["n_total"] for r in overall_rows)
    tot_trust = sum(r["n_trustworthy"] for r in overall_rows)
    tot_disagree = sum(r["n_disagree_trustworthy"] for r in overall_rows)
    print(f"trustworthy fraction: {tot_trust}/{tot_n} = {100*tot_trust/tot_n:.1f}%")
    if tot_trust > 0:
        print(f"disagreement fraction (of trustworthy): {tot_disagree}/{tot_trust} = {100*tot_disagree/tot_trust:.2f}%")


if __name__ == "__main__":
    main()
