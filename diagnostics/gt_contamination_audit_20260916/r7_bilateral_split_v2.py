"""R7 v2 -- can attachment-based lateralisation split the merged SDF appendage regions?  [CORRECTED]

RE-SCORING COPY, 2026-09-16.  Two defects in the original
(`diagnostics/groundtruth/r7_bilateral_split.py`), both fixed here:

  1. FRAME.  It read `landmarks[k]["original"]` raw (Blender Z-up).  VOID per AUDIT_20260916.md.
     Landmarks now come through `gt_frame_v2`: `obj = (x, z, -y)` plus the on-surface assert.
  2. MIDLINE SOURCE.  It carried the expert joints into the .obj frame with
     `annotation/joint_to_obj_transform.json`, a FIT-DERIVED similarity.  Replaced by the JAB
     registration (`joint_alignment_benchmark/data/gt_joints_fitframe.json`), which is solved
     mesh-to-mesh at residual <= 7e-8, already handles the four MIRRORED scenes and their
     `_r`/`_l` swap, excludes Dolichoderus, and consults no model.

Neither the scored question nor the method changes: appendage vertices are labelled by the
laterality of the BODY vertex they attach to along the mesh's own edge graph (geodesic multi-source
Dijkstra), and the test is whether the right and left apex of a pair receive DIFFERENT labels.
That question is invariant to an `_r`/`_l` relabelling, as is the head-width straddle gate.
The SMAL fit is no longer loaded at all -- nothing in this version uses it.

Original header follows.
--------------------------------------------------------------------------------------------
R7 -- can the model's bilateral geometry split the merged SDF appendage regions?

R6 found the SDF field separates body from appendage on real scans but merges left and right: the
two mandibular apexes land in the SAME connected component on 9 of 12 specimens, because the
structures physically touch in an ethanol scan. The blocker is laterality, not segmentation.

WHY A NAIVE MIDLINE SPLIT IS WRONG, and is not what this does. Assigning "x < midline -> left" fails
exactly where it matters: Aenictus and Eciton hold their mandibles crossed, so the RIGHT mandible's
TIP sits on the LEFT of the body. V11b measured that directly. A position-based split would confuse
precisely the specimens the anterior thread cares about.

WHAT THIS DOES INSTEAD -- attachment, not position. An appendage's ARTICULATION is always on its own
side, however far the tip crosses. So for every appendage vertex we walk the mesh's own edge graph
back to the nearest BODY vertex (multi-source Dijkstra from the thick side), and take the laterality
of that attachment point. The tip inherits the side of the structure it grows from.

The midline comes from the FITTED model, restricted to the body parts (b_h, b_t, b_a_*). R2/R4
established those are the parts the fit places correctly -- head_width_l 0.3%, clypeal_ant_mid 6.3%
-- while the appendages are the misplaced ones. So the plane is taken from the trustworthy half of
the fit and used to adjudicate the untrustworthy half.

SCORED against the four expert appendage landmarks: after splitting, do the right and left mandible
apexes receive DIFFERENT laterality labels? R6's baseline is 2/12 specimens correct.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import gt_frame_v2 as GF  # noqa: E402

MESHDIR = "/hpcwork/nao48500/worker_alt_data"
RNG = np.random.default_rng(0)
THRESH, N_RAYS, CONE_DEG = 0.0307, 25, 120.0
PAIRS = [("mandibular_apex_r", "mandibular_apex_l"), ("scape_apex_r", "scape_apex_l")]


def cone_dirs(normal, n_rays, half_angle):
    axis = -normal / (np.linalg.norm(normal) + 1e-12)
    u = RNG.random(n_rays)
    cos_t = 1.0 - u * (1.0 - np.cos(half_angle))
    sin_t = np.sqrt(np.clip(1 - cos_t ** 2, 0, 1))
    phi = RNG.random(n_rays) * 2 * np.pi
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(axis, e1)
    return (cos_t[:, None] * axis
            + sin_t[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))


def sdf_field(mesh, points, normals, batch=20000):
    inter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    half = np.deg2rad(CONE_DEG) / 2.0
    n = len(points)
    dirs = np.empty((n, N_RAYS, 3))
    for i in range(n):
        dirs[i] = cone_dirs(normals[i], N_RAYS, half)
    eps = 1e-5 * float(mesh.scale)
    origins = np.repeat(points + eps * (-normals), N_RAYS, axis=0)
    d_flat = dirs.reshape(-1, 3)
    lengths = np.full(len(origins), np.nan)
    for s in range(0, len(origins), batch):
        e = min(s + batch, len(origins))
        loc, ir, _ = inter.intersects_location(origins[s:e], d_flat[s:e], multiple_hits=False)
        if len(ir):
            lengths[s + ir] = np.linalg.norm(loc - origins[s:e][ir], axis=1)
    L = lengths.reshape(n, N_RAYS)
    out = np.full(n, np.nan)
    for i in range(n):
        v = L[i][np.isfinite(L[i])]
        if v.size < 3:
            continue
        med, sd = np.median(v), v.std()
        keep = v[np.abs(v - med) <= sd] if sd > 0 else v
        out[i] = keep.mean() if keep.size else med
    return out


def main():
    ann = GF.load_landmarks()              # FRAME-CORRECTED + on-surface assert, gt_expert
    GTJ = GF.load_expert_joints_obj()      # JAB-registered expert joints, .obj frame
    sids = sorted(set(ann) & set(GTJ))
    assert len(sids) >= 10, sids

    # ARM B midline: the R8 v2 body-symmetry plane. R8 v2 re-scored in the corrected frame now
    # straddles on 10/11, MORE specimens than the expert-joint-pair midline used as arm A, so the
    # attachment method is given the better of the two available midlines as well.
    r8p = os.path.join(HERE, "r8_results_v2.json")
    R8 = {}
    if os.path.exists(r8p):
        for r in json.load(open(r8p)):
            if "plane_centre" in r:
                R8[r["specimen"]] = (np.asarray(r["normal"], float),
                                     np.asarray(r["plane_centre"], float), float(r["plane_offset"]))
    print(f"[arm B] R8 v2 symmetry planes available for {len(R8)}/{len(sids)} specimens")

    print(f"\n{len(sids)} specimens (gt_expert, frame-corrected landmarks, JAB-registered joints)\n")
    print(f"{'specimen':34s}{'body v':>8}{'app v':>8}{'t(s)':>7}   A: mandibles / scapes"
          f"      B: mandibles / scapes")
    rows = []
    for si, sid in enumerate(sids):
        t0 = time.time()
        m = trimesh.load(os.path.join(MESHDIR, f"{sid}_processed.obj"), process=False)
        Vs = np.asarray(m.vertices)
        o = Vs.mean(0); sc = np.abs(Vs - o).max()

        # --- MIDLINE FROM THE EXPERT JOINTS, not from the fit.
        # This experiment is about whether ATTACHMENT-BASED lateralisation works, not about midline
        # estimation, so the midline is human ground truth. If the method works with a correct
        # midline, midline estimation is a distinct and easier sub-problem; if it fails even here,
        # the method itself is wrong.
        # The joints come from the JAB registration and are ALREADY in the .obj frame -- mesh-to-mesh,
        # residual <= 7e-8, mirrored scenes and their _r/_l swap handled, no fit consulted.
        GT = GTJ[sid]
        pr_ = [(GT[n], GT[n[:-2] + "_l"]) for n in GT
               if n.endswith("_r") and n[:-2] + "_l" in GT]
        if len(pr_) < 3:
            print(f"{sid[:33]:34s}  insufficient bilateral joints -- skipped"); continue
        lat = np.mean([a - b for a, b in pr_], axis=0)
        lat /= np.linalg.norm(lat) + 1e-12
        origin_obj = np.mean([(a + b) / 2 for a, b in pr_], axis=0)
        sideA = lambda P: (np.atleast_2d(P) - origin_obj) @ lat   # scan (.obj) frame
        sideB = None
        if sid in R8:
            nB, cB, dB = R8[sid]
            sideB = lambda P, nB=nB, cB=cB, dB=dB: (np.atleast_2d(P) - cB) @ nB - dB

        # --- SDF appendage mask, then attachment-based laterality
        s = sdf_field(m, Vs, np.asarray(m.vertex_normals))
        scale = float(np.linalg.norm(m.bounding_box.extents))
        thin = np.isfinite(s) & (s / scale < THRESH)
        body_v = np.where(~thin)[0]
        app_v = np.where(thin)[0]
        if len(body_v) < 10 or len(app_v) < 10:
            print(f"{sid[:33]:34s}  degenerate split -- skipped"); continue

        e = np.vstack([m.edges_unique, m.edges_unique[:, ::-1]])
        w = np.linalg.norm(Vs[e[:, 0]] - Vs[e[:, 1]], axis=1)
        G = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(Vs), len(Vs))).tocsr()
        # Multi-source Dijkstra ALONG THE MESH from every body vertex. `sources` gives, per
        # vertex, WHICH body vertex it was reached from -- the attachment point. This must be
        # geodesic, not Euclidean: a crossed mandible's tip is Euclidean-nearest to the OPPOSITE
        # side of the head, which is exactly the error this experiment exists to avoid.
        dist_g, _, sources = dijkstra(G, directed=False, indices=body_v, min_only=True,
                                      return_predecessors=True)
        reach = np.isfinite(dist_g) & (sources >= 0)
        tree = cKDTree(Vs)
        idx = {k: int(tree.query(ann[sid][k])[1]) for k in ann[sid]}

        def evaluate(side):
            """Label every reachable vertex by the side of its ATTACHMENT, then score the pairs."""
            if side is None:
                return None
            lab = np.zeros(len(Vs))
            lab[reach] = np.sign(side(Vs[sources[reach]]).ravel())
            # SANITY: the two head-width landmarks must straddle the plane. If they do not, the
            # midline is wrong and this specimen's result is meaningless, not negative.
            hw = None
            if "head_width_r" in idx and "head_width_l" in idx:
                hw = bool(np.sign(float(side(ann[sid]["head_width_r"]))) !=
                          np.sign(float(side(ann[sid]["head_width_l"]))))
            res = {}
            for r, l in PAIRS:
                if r not in idx or l not in idx:
                    continue
                res[r.split("_")[0]] = bool(lab[idx[r]] != lab[idx[l]] and lab[idx[r]] != 0)
            return dict(midline_ok=hw, result=res)

        A, B = evaluate(sideA), evaluate(sideB)
        rows.append(dict(specimen=sid, n_body=len(body_v), n_app=len(app_v),
                         n_unreachable=int((~reach).sum()),
                         midline_ok=A["midline_ok"], result=A["result"],   # arm A, back-compatible
                         armA=A, armB=B, secs=time.time() - t0))

        def fmt(X):
            if X is None:
                return "n/a"
            t = " ".join(f"{k}:{'OK' if v else 'merged'}" for k, v in X["result"].items())
            return t + ("" if X["midline_ok"] else "  [MIDLINE INVALID]")
        print(f"{sid[:33]:34s}{len(body_v):>8}{len(app_v):>8}{time.time()-t0:>7.1f}   "
              f"A: {fmt(A):44s} B: {fmt(B)}", flush=True)

    summary = {}
    for arm, label in (("armA", "A: expert bilateral JOINT pairs"),
                       ("armB", "B: R8 v2 body-symmetry plane")):
        sub = [r for r in rows if r[arm] is not None]
        valid = [r for r in sub if r[arm]["midline_ok"]]
        print(f"\n  MIDLINE {label}")
        print(f"    valid midline (head-width landmarks straddle it): {len(valid)}/{len(sub)}")
        if not valid:
            print("    NO valid midline -- the laterality question is NOT TESTED under this arm.")
            continue
        a = sum(1 for r in valid if r[arm]["result"].get("mandibular", False))
        b = sum(1 for r in valid if r[arm]["result"].get("scape", False))
        print(f"    mandibles correctly lateralised: {a}/{len(valid)}"
              f"   (R6 v2 baseline, SDF components alone: 2/11)")
        print(f"    scapes correctly lateralised:    {b}/{len(valid)}   (R6 v2 baseline: 10/11)")
        summary[arm] = dict(n_valid=len(valid), mandibular=a, scape=b)
    ok = any(v["mandibular"] >= 0.75 * v["n_valid"] for v in summary.values() if v["n_valid"])
    print("\n=> ATTACHMENT-BASED SPLITTING WORKS. Laterality can be recovered geometrically,"
          if ok else
          "\n=> attachment-based splitting does NOT recover laterality reliably. The merge is")
    print("   so the named part label needed by R3 is obtainable without learning."
          if ok else "   deeper than surface connectivity -- a learned assignment is warranted.")
    json.dump(dict(rows=rows, summary=summary),
              open(os.path.join(HERE, "r7_results_v2.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
