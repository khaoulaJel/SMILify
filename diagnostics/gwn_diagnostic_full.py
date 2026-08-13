"""
Extension of gwn_diagnostic.py to all 50 bench50_clean specimens: for each specimen,
compute GWN-disagreement rate for A_into_B (gaster->legs) and B_into_A (legs->gaster),
gentle arm, and correlate the disagreement ASYMMETRY (A_into_B rate - B_into_A rate)
against whether that specimen's actual count for each direction got worse/better
(baseline -> gentle), using the already-saved per-pair CSVs from step 1 (no new GPU time).

Pilot (n=3) finding to replicate or refute: disagreement rate was 3-8x higher in the
worsening direction (A_into_B) than the improving direction (B_into_A), for the 2
specimens that showed the count-worsening pattern, absent in the 1 that didn't.
"""
import sys
sys.path.insert(0, "/p/home/jusers/jellal1/jureca/SMILify")
sys.path.insert(0, "/p/home/jusers/jellal1/jureca/SMILify/diagnostics/moonshot")

import numpy as np
import pandas as pd
import torch
from collections import defaultdict

from joint_placement_common import winding_number
from fitter_3d.trainer import SMAL3DFitter
from fitter_3d.part_groups import get_part_vertex_indices, PART_GROUPS_COARSE
from fitter_3d.penetration_loss import _build_part_faces, _directional_penalty

K_MARGIN = 2.5


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


def disagreement_rate(verts, faces, idx_query, surface_faces_orig, capped_V, capped_F, vert_edge_len, tau, max_depth):
    verts_t = torch.tensor(verts, dtype=torch.float32).unsqueeze(0)
    tau_t = torch.tensor([tau])
    max_depth_t = torch.tensor([max_depth])
    idx_q_t = torch.tensor(idx_query, dtype=torch.long)
    faces_surf_t = torch.tensor(surface_faces_orig, dtype=torch.long)
    _, diag = _directional_penalty(verts_t, idx_q_t, faces_surf_t, tau_t, max_depth_t)
    proximity_penetrating = diag["penetrating_mask"][0].numpy()

    query_pts = verts[idx_query]
    w = winding_number(query_pts, capped_V, capped_F)
    margin = np.abs(w - 0.5)
    true_inside = w > 0.5
    local_edge = vert_edge_len[idx_query]
    trustworthy = margin > (K_MARGIN * local_edge)

    n_trust = int(trustworthy.sum())
    if n_trust == 0:
        return None, 0, 0
    disagreement = (proximity_penetrating != true_inside) & trustworthy
    return float(disagreement.sum()) / n_trust, n_trust, len(idx_query)


def main():
    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].cpu().numpy()
    part_vertex_indices = get_part_vertex_indices(PART_GROUPS_COARSE)
    part_faces = _build_part_faces(faces, part_vertex_indices)
    idx_gaster = part_vertex_indices["gaster"]
    idx_legs = part_vertex_indices["legs"]
    faces_gaster = part_faces["gaster"]
    faces_legs = part_faces["legs"]

    gentle = np.load("fit3d_results_all_gentle/Stage_3_deform_fine.npz", allow_pickle=True)
    labels = list(gentle["labels"])
    specimens = labels  # all 50

    # per-specimen, per-direction count deltas from the already-saved per-pair CSVs (step 1)
    pp_base = pd.read_csv("fit3d_results_all_baseline/Stage_3_deform_fine_per_pair_penetration.csv")
    pp_gentle = pd.read_csv("fit3d_results_all_gentle/Stage_3_deform_fine_per_pair_penetration.csv")

    def get_count(df, specimen, direction):
        row = df[(df["specimen"] == specimen) & (df["part_a"] == "gaster") & (df["part_b"] == "legs")
                  & (df["direction"] == direction)]
        return float(row["num_penetrating"].values[0]) if len(row) else np.nan

    rows = []
    for si, spec in enumerate(specimens):
        i = labels.index(spec)
        verts = gentle["verts"][i]
        bbox_diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
        tau = 0.03 * bbox_diag
        max_depth = 0.08 * bbox_diag
        vert_edge_len = per_vertex_edge_length(verts, faces)

        gaster_capped_V, gaster_capped_F = cap_part_mesh(verts, faces_gaster)
        legs_capped_V, legs_capped_F = cap_part_mesh(verts, faces_legs)

        rate_AtoB, ntrust_AtoB, n_AtoB = disagreement_rate(
            verts, faces, idx_gaster, faces_legs, legs_capped_V, legs_capped_F, vert_edge_len, tau, max_depth)
        rate_BtoA, ntrust_BtoA, n_BtoA = disagreement_rate(
            verts, faces, idx_legs, faces_gaster, gaster_capped_V, gaster_capped_F, vert_edge_len, tau, max_depth)

        base_AtoB = get_count(pp_base, spec, "A_into_B")
        gentle_AtoB = get_count(pp_gentle, spec, "A_into_B")
        base_BtoA = get_count(pp_base, spec, "B_into_A")
        gentle_BtoA = get_count(pp_gentle, spec, "B_into_A")

        row = {
            "specimen": spec,
            "rate_AtoB": rate_AtoB, "rate_BtoA": rate_BtoA,
            "asymmetry": (rate_AtoB - rate_BtoA) if (rate_AtoB is not None and rate_BtoA is not None) else np.nan,
            "delta_AtoB": gentle_AtoB - base_AtoB,
            "delta_BtoA": gentle_BtoA - base_BtoA,
            "worsen_AtoB": (gentle_AtoB - base_AtoB) > 0,
            "improve_BtoA": (gentle_BtoA - base_BtoA) < 0,
        }
        rows.append(row)
        print(f"[{si+1}/{len(specimens)}] {spec}: rate_AtoB={rate_AtoB} rate_BtoA={rate_BtoA} "
              f"delta_AtoB={row['delta_AtoB']:+.0f} delta_BtoA={row['delta_BtoA']:+.0f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("/tmp/claude-34909/-p-home-jusers-jellal1-jureca-SMILify/ee8ca7f8-5407-4536-8927-01bd0da6433b/scratchpad/gwn_full_results.csv", index=False)

    print("\n=== SUMMARY across 50 specimens ===")
    valid = df.dropna(subset=["asymmetry"])
    print(f"n specimens with valid asymmetry: {len(valid)}/{len(df)}")

    # correlation: asymmetry vs delta_AtoB (does higher A_into_B disagreement track worse A_into_B outcome?)
    from scipy.stats import spearmanr, pearsonr
    r_p, p_p = pearsonr(valid["asymmetry"], valid["delta_AtoB"])
    r_s, p_s = spearmanr(valid["asymmetry"], valid["delta_AtoB"])
    print(f"asymmetry vs delta_AtoB: Pearson r={r_p:.3f} (p={p_p:.4f})  Spearman rho={r_s:.3f} (p={p_s:.4f})")

    # split by worsen_AtoB
    worsen = valid[valid["worsen_AtoB"]]
    not_worsen = valid[~valid["worsen_AtoB"]]
    print(f"\nspecimens where A_into_B WORSENED (n={len(worsen)}): mean asymmetry = {worsen['asymmetry'].mean():.4f}")
    print(f"specimens where A_into_B did NOT worsen (n={len(not_worsen)}): mean asymmetry = {not_worsen['asymmetry'].mean():.4f}")
    print(f"mean rate_AtoB in worsened group: {worsen['rate_AtoB'].mean():.4f}  vs not-worsened: {not_worsen['rate_AtoB'].mean():.4f}")
    print(f"mean rate_BtoA in worsened group: {worsen['rate_BtoA'].mean():.4f}  vs not-worsened: {not_worsen['rate_BtoA'].mean():.4f}")

    from scipy.stats import mannwhitneyu
    if len(worsen) > 0 and len(not_worsen) > 0:
        u, p = mannwhitneyu(worsen["asymmetry"], not_worsen["asymmetry"], alternative="two-sided")
        print(f"Mann-Whitney U test (asymmetry, worsened vs not): U={u:.1f} p={p:.4f}")


if __name__ == "__main__":
    main()
