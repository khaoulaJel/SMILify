"""D2b -- at the FINAL fitted state, is the optimizer's own correspondence machinery putting
distal-leg target points on the WRONG LEG (cross-leg confusion), or on the RIGHT leg but the
WRONG SEGMENT (within-leg ambiguity)? Uses the fitter's own `TargetPartition` class
(`fitter_3d.trainer_hierarchical`), not a reimplementation, run once at the specimen's converged
fitted-vertex state -- this is a snapshot audit (end of optimization), not a full trajectory
replay, but it is the most informative single point: if convergence itself has settled into a
corrupted assignment, that is the failure that matters for the final measurement.

TRUE LABELS come from the SAME area-weighted sampling used in `probe_d2a_sample_starvation.py`
(reused, not reimplemented): sampling the specimen's own target mesh gives, for free, which FACE
(hence which true leg+segment, by majority dominant-weight vote) each target point actually
belongs to -- this is exact in the synthetic ceiling test because target and ground truth share
correspondence by construction.

TWO METRICS, matching the two distinct questions in the task:
  cross_leg_confusion   for target points whose TRUE leg is K: fraction the fitter's own
                        TargetPartition (leg-granularity, matches every run so far --
                        split_distal was never used) assigns to leg != K.
  within_leg_seg_mismatch  for target points whose true leg IS correctly identified, and whose
                        true segment is ti/ta/pt: of the fitted vertices in that (correctly
                        identified) leg group, which one is nearest (the same knn direction
                        `_partitioned_chamfer` uses for the target->source term)? Fraction where
                        that nearest fitted vertex's OWN dominant segment != the point's true
                        segment.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.ops import knn_points  # noqa: E402
from fitter_3d.trainer_hierarchical import TargetPartition, vertex_groups  # noqa: E402
from probe_d2a_sample_starvation import face_areas, face_group_by_segment  # noqa: E402

N_SAMPLE = 8000
DEVICE = "cpu"


def true_face_labels(M):
    """(leg_of_face, side_of_face, seg_of_face) per TEMPLATE face, '' where not a leg face."""
    fgroup_seg = face_group_by_segment(M)  # e.g. 'l2_ti_r' or 'other'
    leg_of, seg_of = [], []
    for g in fgroup_seg:
        if g == "other":
            leg_of.append(None)
            seg_of.append(None)
        else:
            bits = g.split("_")  # ['l2','ti','r']
            leg_of.append(bits[0] + "_" + bits[2])  # 'l2_r'
            seg_of.append(bits[1])  # 'ti'
    return np.array(leg_of, dtype=object), np.array(seg_of, dtype=object)


def sample_with_true_labels(verts, faces, leg_of_face, seg_of_face, n_sample, rng):
    areas = face_areas(verts, faces)
    p = areas / areas.sum()
    face_idx = rng.choice(len(p), size=n_sample, p=p)
    v0, v1, v2 = faces[face_idx, 0], faces[face_idx, 1], faces[face_idx, 2]
    b = rng.dirichlet([1, 1, 1], size=n_sample)  # barycentric coords, same scheme as pytorch3d
    pts = b[:, 0:1] * verts[v0] + b[:, 1:2] * verts[v1] + b[:, 2:3] * verts[v2]
    return pts, leg_of_face[face_idx], seg_of_face[face_idx]


def audit_run(run, corpus, M, vg, group_names, leg_of_face, seg_of_face, faces, rng, n_specimens=None):
    d = np.load(os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz"))
    labels = [str(x) for x in d["labels"]]
    fitted = d["verts"].astype(np.float64)
    dom = M["dominant"]  # per-fitted-vertex dominant joint idx, for the within-leg check
    jn = M["jnames"]

    def vertex_seg(j):
        n = jn[j]
        if not n.startswith("l_"):
            return None
        return n.split("_")[2]

    vertex_true_seg = np.array([vertex_seg(j) for j in dom], dtype=object)

    vg_t = torch.as_tensor(vg, device=DEVICE)
    name2i = {n: i for i, n in enumerate(group_names)}

    rows = []
    for i, lab in enumerate(labels[: n_specimens or len(labels)]):
        stem = lab[:-4] if lab.endswith(".obj") else lab
        ov, of, _ = load_obj(os.path.join(MOON, corpus, f"{stem}.obj"), load_textures=False)
        verts_t, faces_t = ov.numpy(), of.verts_idx.numpy()
        pts, true_leg, true_seg = sample_with_true_labels(verts_t, faces_t, leg_of_face, seg_of_face, N_SAMPLE, rng)
        is_leg_pt = true_leg != None  # noqa: E711
        pts_t = torch.tensor(pts[is_leg_pt], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        true_leg_pt = true_leg[is_leg_pt]
        true_seg_pt = true_seg[is_leg_pt]

        fitted_i = torch.tensor(fitted[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        part = TargetPartition(vg_t, len(group_names), DEVICE)
        part.update(fitted_i, pts_t)
        assigned_group = part.assign[0].cpu().numpy()  # (P,) group id per point
        assigned_name = np.array(group_names)[assigned_group]

        # metric 1: cross-leg confusion (assigned leg != true leg), split by true segment
        cross_leg = assigned_name != true_leg_pt
        row = dict(run=run, specimen=stem)
        for seg in ("co", "tr", "fe", "ti", "ta", "pt"):
            m = true_seg_pt == seg
            row[f"cross_leg_{seg}"] = float(cross_leg[m].mean()) if m.any() else float("nan")
            row[f"n_{seg}"] = int(m.sum())

        # metric 2: within-leg segment mismatch, RESTRICTED to points whose leg was assigned
        # correctly (isolates segment-level ambiguity from leg-level confusion)
        correct_leg_mask = ~cross_leg
        for seg in ("ti", "ta", "pt"):
            m = (true_seg_pt == seg) & correct_leg_mask
            if not m.any():
                row[f"within_leg_mismatch_{seg}"] = float("nan")
                continue
            sub_pts = pts_t[0][m]
            leg_name = true_leg_pt[m][0]  # same leg for all points in this seg's true-leg group...
            # (not quite -- true_leg_pt[m] can mix legs 1/2/3; handle per-leg below)
            mismatches, total = 0, 0
            for leg_name_u in np.unique(true_leg_pt[m]):
                mm = (true_leg_pt == leg_name_u) & m
                gid = name2i[leg_name_u]
                leg_vertex_mask = vg == gid
                leg_vert_idx = np.where(leg_vertex_mask)[0]
                src = torch.tensor(fitted[i, leg_vert_idx], dtype=torch.float32).unsqueeze(0)
                tgt = pts_t[0][mm].unsqueeze(0)
                nn_idx = knn_points(tgt, src, K=1).idx[0, :, 0].cpu().numpy()
                matched_global_vidx = leg_vert_idx[nn_idx]
                matched_seg = vertex_true_seg[matched_global_vidx]
                mismatches += int((matched_seg != seg).sum())
                total += mm.sum()
            row[f"within_leg_mismatch_{seg}"] = mismatches / max(total, 1)
        rows.append(row)
        print(
            f"{run:<24}{stem:<12}"
            + " ".join(f"{seg}:xleg={row[f'cross_leg_{seg}']:.2f}" for seg in ("ti", "ta", "pt"))
            + "  "
            + " ".join(f"{seg}:mismatch={row[f'within_leg_mismatch_{seg}']:.2f}" for seg in ("ti", "ta", "pt"))
        )
    return rows


def main():
    M = ms.load_model()
    weights = M["weights"]
    jnames = M["jnames"]
    faces = np.asarray(M["dd"]["f"])
    vg, group_names = vertex_groups(weights, jnames, split_distal=False)
    leg_of_face, seg_of_face = true_face_labels(M)
    rng = np.random.default_rng(1)

    conditions = [
        ("SYN_clean_pose0_w5", "synth_clean_pose0"),
        ("SYN_clean_pose05_w5", "synth_clean_pose05"),
        ("SYN_clean_pose10_w5", "synth_clean_pose10"),
        ("SYN_clean_pose15_w5", "synth_clean_pose15"),
        ("SYN_clean_pose20_w5", "synth_clean_pose20"),
        ("SYN_clean_w5", "synth_clean"),
        ("SYN_clean_pose35_w5", "synth_clean_pose35"),
        ("SYN_clean_drop30_w5", "synth_clean_drop30"),
        ("SYN_clean_drop60_w5", "synth_clean_drop60"),
    ]

    import json
    all_rows = []
    for run, corpus in conditions:
        p = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            print(f"[skip] {run} not found")
            continue
        rows = audit_run(run, corpus, M, vg, group_names, leg_of_face, seg_of_face, faces, rng)
        all_rows.extend(rows)

    print(f"\n=== summary: mean cross-leg confusion across specimens, per condition ===")
    print(f"{'run':<24}{'ti':>8}{'ta':>8}{'pt':>8}")
    by_run = {}
    for r in all_rows:
        by_run.setdefault(r["run"], []).append(r)
    for run in [c[0] for c in conditions if c[0] in by_run]:
        rs = by_run[run]
        vals = [np.nanmean([r[f"cross_leg_{seg}"] for r in rs]) for seg in ("ti", "ta", "pt")]
        print(f"{run:<24}{vals[0]:>8.3f}{vals[1]:>8.3f}{vals[2]:>8.3f}")

    OUT = os.path.join(HERE, "out")
    os.makedirs(OUT, exist_ok=True)
    json.dump(all_rows, open(os.path.join(OUT, "d2b_correspondence_audit.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/d2b_correspondence_audit.json")


if __name__ == "__main__":
    main()
