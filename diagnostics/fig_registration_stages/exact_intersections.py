"""Exact self-intersection for the stage CSVs: intersecting triangle pairs between non-adjacent body parts.

Why: the columns pen_hard_triples / pen_soft / pen_unique_verts are PROXIMITY counts (a template vertex within a
distance threshold of a non-adjacent part, fitter_3d/penetration_loss.py). They flag near-contact: for the figure
specimen they read 18, 46, 3, 6 at the four panels although the surfaces do not cross at any stage. This script adds
an exact measure: the number of triangle pairs (one triangle from each of two anatomically non-adjacent parts,
fitter_3d.part_groups) that intersect, by FCL triangle-triangle tests under a bounding-volume hierarchy
(python-fcl). No distance threshold. This is the "triangle collisions" count of the body-fitting literature
(SMPLify-X, Pavlakos et al. 2019). Limits: the six legs are one part, so leg-leg crossings are not counted; a surface
lying wholly inside another without crossing counts 0 (column gwn_gaster_legs_inside covers containment).

Self-check on start: two unit icospheres, separated -> 0, overlapping -> equals an independent brute-force count.

Adds columns `intersecting_tri_pairs` and `intersecting_tri_pairs_gaster_legs` to stages_all20.csv and stages.csv
(existing columns untouched) and `median20_intersecting_tri_pairs` to stages.csv. No fitting, no rendering.
Usage: python diagnostics/fig_registration_stages/exact_intersections.py --hier_dir <...> --moon_dir <...>
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
os.chdir(REPO)

import fcl  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from fitter_3d.part_groups import get_non_adjacent_pairs, get_part_vertex_indices  # noqa: E402
from fitter_3d.penetration_loss import _build_part_faces  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402

MAX_CONTACTS = 200_000


def count_pair(va, fa, vb, fb):
    def bvh(v, f):
        b = fcl.BVHModel()
        b.beginModel(len(v), len(f))
        b.addSubModel(np.asarray(v, float), np.asarray(f, np.int32))
        b.endModel()
        return fcl.CollisionObject(b, fcl.Transform())
    res = fcl.CollisionResult()
    fcl.collide(bvh(va, fa), bvh(vb, fb), fcl.CollisionRequest(num_max_contacts=MAX_CONTACTS, enable_contact=True), res)
    assert len(res.contacts) < MAX_CONTACTS
    return len({(c.b1, c.b2) for c in res.contacts})


def _seg_tri(p, q, tri):
    d, e1, e2 = q - p, tri[1] - tri[0], tri[2] - tri[0]
    h = np.cross(d, e2)
    a = e1 @ h
    if abs(a) < 1e-14:
        return False
    s = p - tri[0]
    u = (s @ h) / a
    if u < 0 or u > 1:
        return False
    qv = np.cross(s, e1)
    v = (d @ qv) / a
    return v >= 0 and u + v <= 1 and 0 <= (e2 @ qv) / a <= 1


def self_check():
    s1 = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    far, near = s1.copy(), s1.copy()
    far.apply_translation([2.5, 0.013, 0.007])
    near.apply_translation([1.5, 0.013, 0.007])
    assert count_pair(s1.vertices, s1.faces, far.vertices, far.faces) == 0
    A, Bt = s1.triangles, near.triangles
    brute = sum(any(_seg_tri(a[k], a[(k + 1) % 3], b) for k in range(3)) or any(_seg_tri(b[k], b[(k + 1) % 3], a) for k in range(3))
                for a in A for b in Bt if np.linalg.norm(a.mean(0) - b.mean(0)) < 1.0)
    n = count_pair(s1.vertices, s1.faces, near.vertices, near.faces)
    assert n == brute and n > 0, (n, brute)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hier_dir", default="/hpcwork/nao48500/fig_registration_stages/D1_s0_cpu1_merged_hier")
    ap.add_argument("--moon_dir", default="/hpcwork/nao48500/fig_registration_stages/D1_s0_cpu1_merged")
    a = ap.parse_args()
    print(f"[self-check] overlapping spheres: FCL = brute force = {self_check()} triangle pairs; separated = 0")
    files = {s: os.path.join(a.hier_dir, f"{s}.npz") for s in ("H0_body", "H1_legs", "H2_joint")}
    files.update({s: os.path.join(a.moon_dir, f"{s}.npz") for s in ("Stage_2_deform_coarse", "Stage_3_deform_fine")})
    d3 = np.load(files["Stage_3_deform_fine"], allow_pickle=True)
    labels = [str(x).replace(".obj", "").zfill(2) for x in d3["labels"]]
    faces = (d3["faces"][0] if d3["faces"].ndim == 3 else d3["faces"]).astype(int)
    pf = _build_part_faces(faces, get_part_vertex_indices())
    pairs = get_non_adjacent_pairs()
    with torch.no_grad():
        v0 = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)()
    V = {"init": np.repeat(v0.cpu().numpy().astype(float).reshape(1, -1, 3), len(labels), 0)}
    V.update({s: np.load(f, allow_pickle=True)["verts"].astype(float) for s, f in files.items()})

    def score(v):
        tot = gl = 0
        for pa, pb in pairs:
            if len(pf.get(pa, [])) and len(pf.get(pb, [])):
                n = count_pair(v, pf[pa], v, pf[pb])
                tot += n
                gl += n if {pa, pb} == {"gaster", "legs"} else 0
        return tot, gl

    rec = {}
    for s, vv in V.items():
        for i, sp in enumerate(labels):
            rec[(sp, s)] = rec[(labels[0], s)] if (s == "init" and i > 0) else score(vv[i])
    for name in ("stages_all20.csv", "stages.csv"):                 # append columns as text; existing cells are not re-parsed
        p = os.path.join(HERE, name)
        with open(p, newline="") as fh:
            rows = list(csv.reader(fh))
        head, body = rows[0], rows[1:]
        assert "intersecting_tri_pairs" not in head, f"{name} already carries the exact columns"
        i_sp, i_st = head.index("specimen"), head.index("stage")
        new = ["intersecting_tri_pairs", "intersecting_tri_pairs_gaster_legs"] + (["median20_intersecting_tri_pairs"] if name == "stages.csv" else [])
        med = {s: float(np.median([rec[(sp, s)][0] for sp in labels])) for s in V}
        for r in body:
            k = (str(r[i_sp]).zfill(2), r[i_st])
            r += [str(rec[k][0]), str(rec[k][1])] + ([str(med[r[i_st]])] if name == "stages.csv" else [])
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(head + new)
            w.writerows(body)
        print(f"wrote {name}" + (": " + ", ".join(f"{r[i_st]}={r[len(head)]}" for r in body) if name == "stages.csv" else ""))
    n_any = {s: sum(rec[(sp, s)][0] > 0 for sp in labels) for s in V}
    print("specimens (of 20) with any intersecting triangle pair, per stage:", n_any)


if __name__ == "__main__":
    main()
