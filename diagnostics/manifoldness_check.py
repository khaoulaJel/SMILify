"""
Pure-numpy reimplementation of custom_processing/prepare_antscan_data_for_mesh_fitting_enhanced.py's
mesh_integrity_report() and report_mesh_degeneracy() -- those take a bpy.types.Object (Blender's
bmesh API), and bpy is not installed in the pytorch3d conda env used for the fitter. Semantics
matched to their docstrings exactly:
  - boundary_edges: edges with exactly 1 linked face
  - non_manifold_edges: edges with 3+ linked faces
  - non_manifold_verts: vertex whose surrounding faces don't form a single connected fan
    (bowtie configuration) -- proxy: build a graph of faces touching v, connect two faces if
    they share an edge incident to v; >1 connected component => non-manifold vertex.
  - is_watertight: boundary_edges == 0 and non_manifold_edges == 0 and non_manifold_verts == 0
  - signed_volume: divergence-theorem formula, sum over faces of v0.(v1 x v2) / 6
  - degeneracy: zero-length edges, zero-area faces, duplicate verts (rounded-position collision)
"""
import numpy as np
from collections import defaultdict


def mesh_integrity_report_np(verts: np.ndarray, faces: np.ndarray) -> dict:
    edges = defaultdict(list)  # (v_lo, v_hi) -> [face_idx, ...]
    for fi, f in enumerate(faces):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (a, b) if a < b else (b, a)
            edges[key].append(fi)

    boundary_edges = sum(1 for fl in edges.values() if len(fl) == 1)
    non_manifold_edges = sum(1 for fl in edges.values() if len(fl) >= 3)

    # faces touching each vertex, and which edges (incident to that vertex) link them
    vert_faces = defaultdict(set)
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces[v].add(fi)
    vert_edge_faces = defaultdict(lambda: defaultdict(list))  # v -> edge_key -> [face_idx]
    for key, fl in edges.items():
        for v in key:
            vert_edge_faces[v][key] = fl

    non_manifold_verts = 0
    for v, finc in vert_faces.items():
        if len(finc) <= 1:
            continue
        # union-find over faces incident to v, linked via edges incident to v
        parent = {fi: fi for fi in finc}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for key, fl in vert_edge_faces[v].items():
            if len(fl) == 2:
                union(fl[0], fl[1])
        n_components = len({find(fi) for fi in finc})
        if n_components > 1:
            non_manifold_verts += 1

    is_watertight = boundary_edges == 0 and non_manifold_edges == 0 and non_manifold_verts == 0

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    signed_volume = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)

    return {
        "vertex_count": len(verts),
        "face_count": len(faces),
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "non_manifold_verts": non_manifold_verts,
        "is_watertight": is_watertight,
        "signed_volume": round(signed_volume, 4),
    }


def report_mesh_degeneracy_np(verts: np.ndarray, faces: np.ndarray, zero_len_eps=1e-8,
                                zero_area_eps=1e-9, dup_vert_decimals=7) -> dict:
    edge_set = set()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_set.add((a, b) if a < b else (b, a))
    edges = np.array(list(edge_set))
    edge_lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    n_zero_length_edges = int(np.sum(edge_lengths < zero_len_eps))

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    n_zero_area_faces = int(np.sum(face_areas < zero_area_eps))

    rounded = np.round(verts, dup_vert_decimals)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    n_duplicate_verts = int(np.sum(counts[counts > 1] - 1))

    percentiles = {p: float(np.percentile(edge_lengths, p)) for p in (0, 0.1, 1, 5, 50)}

    return {
        "n_verts": len(verts),
        "n_edges": len(edges),
        "n_faces": len(faces),
        "n_zero_length_edges": n_zero_length_edges,
        "n_zero_area_faces": n_zero_area_faces,
        "n_duplicate_verts": n_duplicate_verts,
        "edge_length_percentiles": percentiles,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/p/home/jusers/jellal1/jureca/SMILify")
    from fitter_3d.trainer import SMAL3DFitter

    SPECIMENS = [
        "Acanthostichus_aff.brevicornis_CASENT0744328_processed.obj",
        "Acromyrmex_coronatus_CASENT0744365_processed.obj",
        "Solenopsis_invicta_CASENT0744384_processed.obj",
    ]

    print("=== Rest-pose template (sanity baseline) ===")
    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    v_template = fitter.smal_model.v_template.detach().cpu().numpy()
    faces = fitter.faces[0].cpu().numpy()
    r = mesh_integrity_report_np(v_template, faces)
    d = report_mesh_degeneracy_np(v_template, faces)
    print(r)
    print(d)
    print()

    for arm, path in [("baseline", "fit3d_results_all_baseline"), ("gentle", "fit3d_results_all_gentle")]:
        data = np.load(f"{path}/Stage_3_deform_fine.npz", allow_pickle=True)
        labels = list(data["labels"])
        for spec in SPECIMENS:
            i = labels.index(spec)
            verts = data["verts"][i]
            print(f"=== {arm} / {spec} (fitted Stage_3) ===")
            r = mesh_integrity_report_np(verts, faces)
            d = report_mesh_degeneracy_np(verts, faces)
            print(r)
            print(d)
            print()
