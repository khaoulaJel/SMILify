"""R6 v2 -- does the SDF part prior recover named part INSTANCES on REAL scans?  [FRAME-CORRECTED]

RE-SCORING COPY, 2026-09-16.  The original `diagnostics/groundtruth/r6_sdf_instances_real.py` read
`landmarks[k]["original"]` raw, i.e. in the annotation scene's Blender Z-up frame, and is VOID per
AUDIT_20260916.md.  This copy takes the landmarks through `gt_frame_v2`, which applies the canonical
`obj = (x, z, -y)` correction and ASSERTS that each specimen's landmarks then lie on the scan
surface, and restricts the set to the 11 usable `annotation/gt_expert/` specimens (Dolichoderus
excluded: annotated on the Discothyrea mesh).  The original is left untouched as the record.

Nothing else is changed: same SDF, same threshold, same components, same scoring rule.


Fabian's `diagnostics/moonshot/sdf_prior_PROBE.py` (2026-08-07) established two things, on two
DIFFERENT datasets:

  * the SDF FIELD is reproducible on REAL ethanol scans -- r 0.77-0.94, 87-95% two-way agreement,
    <1.3 s per specimen, on meshes with 621-5521 components and no watertightness;
  * part INSTANCE recovery is near-perfect -- 5 of 6 legs pure, head appendages (mandibles and
    antennae, left and right) at purity 0.98-1.00 -- but that was measured on the TEMPLATE, which
    is watertight and single-component.

Instance recovery on real fragmented scans was never tested, and it is the one thing standing
between that probe and the automatic part labels R3 needs. R3 showed that supplying "which part
should reach this landmark" moves the named parts from 34.7% of a body length off to 0.2%. If SDF
supplies it automatically, that loop closes with code already in the repo.

METHOD, deliberately the probe's own: CGAL-style inward ray-cone SDF (trimesh + embree), threshold
at the probe's global appendage cut, connected components of the thin side. Each component is then
scored against the 12 annotated specimens' expert landmarks: does the component containing the
human mandible apex also contain the antennal landmarks (a merge), or are they separate instances?

This is a REAL-SCAN test with no template, no fit, and no training.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import gt_frame_v2 as GF  # noqa: E402

MESHDIR = "/hpcwork/nao48500/worker_alt_data"
RNG = np.random.default_rng(0)
THRESH = 0.0307            # the probe's body-vs-appendage cut, in units of bbox diagonal
N_RAYS, CONE_DEG = 25, 120.0
# landmarks that sit on an appendage, and the structure each belongs to
APPENDAGE = {"mandibular_apex_r": "mandible_r", "mandibular_apex_l": "mandible_l",
             "scape_apex_r": "antenna_r", "scape_apex_l": "antenna_l"}


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


def sdf_field(mesh, points, normals, n_rays=N_RAYS, cone_deg=CONE_DEG, batch=20000):
    inter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    half = np.deg2rad(cone_deg) / 2.0
    n = len(points)
    dirs = np.empty((n, n_rays, 3))
    for i in range(n):
        dirs[i] = cone_dirs(normals[i], n_rays, half)
    eps = 1e-5 * float(mesh.scale)
    origins = np.repeat(points + eps * (-normals), n_rays, axis=0)
    d_flat = dirs.reshape(-1, 3)
    lengths = np.full(len(origins), np.nan)
    for s in range(0, len(origins), batch):
        e = min(s + batch, len(origins))
        loc, idx_ray, _ = inter.intersects_location(origins[s:e], d_flat[s:e], multiple_hits=False)
        if len(idx_ray):
            lengths[s + idx_ray] = np.linalg.norm(loc - origins[s:e][idx_ray], axis=1)
    L = lengths.reshape(n, n_rays)
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
    ann = GF.load_landmarks()          # FRAME-CORRECTED + on-surface assert, gt_expert, n = 11
    print(f"\n{len(ann)} annotated specimens (gt_expert, frame-corrected)\n")
    print(f"{'specimen':34s}{'comps':>7}{'thin%':>7}{'t(s)':>7}   appendage landmarks -> component")
    rows = []
    for sid, L in sorted(ann.items()):
        t0 = time.time()
        m = trimesh.load(os.path.join(MESHDIR, f"{sid}_processed.obj"), process=False)
        s = sdf_field(m, m.vertices, m.vertex_normals)
        scale = float(np.linalg.norm(m.bounding_box.extents))
        thin = np.isfinite(s) & (s / scale < THRESH)
        # connected components restricted to the thin side
        keep_f = thin[m.faces].all(axis=1)
        sub = m.submesh([np.where(keep_f)[0]], append=True) if keep_f.any() else None
        comps = sub.split(only_watertight=False) if sub is not None else []
        comps = [c for c in comps if len(c.vertices) >= 30]
        assign, dist = {}, {}
        for k, struct in APPENDAGE.items():
            if k not in L:
                continue
            best, bd = -1, np.inf
            for ci, c in enumerate(comps):
                d = float(np.min(np.linalg.norm(c.vertices - L[k], axis=1)))
                if d < bd:
                    best, bd = ci, d
            assign[struct] = best
            dist[struct] = bd / scale * 100
        # a MERGE is two different structures landing in the same component
        vals = [v for v in assign.values() if v >= 0]
        merged = len(vals) - len(set(vals))
        far = sum(1 for v in dist.values() if v > 5)
        rows.append(dict(specimen=sid, n_comp=len(comps), thin_frac=float(thin.mean()),
                         assign=assign, dist=dist, merged=merged, far=far,
                         secs=time.time() - t0))
        a = " ".join(f"{k.split('_')[0][:4]}{k[-1]}->c{v}" for k, v in assign.items())
        print(f"{sid[:33]:34s}{len(comps):>7}{100*thin.mean():>6.0f}%{time.time()-t0:>7.1f}   "
              f"{a}   {'MERGED' if merged else 'distinct'}"
              f"{'  (some >5% away)' if far else ''}", flush=True)

    nm = sum(1 for r in rows if r["merged"] == 0)
    nf = sum(1 for r in rows if r["far"] == 0)
    pair = lambda a, b: sum(1 for r in rows if a in r["assign"] and b in r["assign"]
                            and r["assign"][a] != r["assign"][b])
    print(f"\n  mandible_r vs mandible_l in DIFFERENT components: {pair('mandible_r','mandible_l')}"
          f"/{len(rows)}")
    print(f"  antenna_r  vs antenna_l  in DIFFERENT components: {pair('antenna_r','antenna_l')}"
          f"/{len(rows)}")
    print(f"  mandible_r vs antenna_r  in DIFFERENT components: {pair('mandible_r','antenna_r')}"
          f"/{len(rows)}")
    print(f"\n  specimens where all four appendage landmarks land in DISTINCT components: "
          f"{nm}/{len(rows)}")
    print(f"  specimens where every landmark is within 5% of its component: {nf}/{len(rows)}")
    print(f"  median components on the thin side: {np.median([r['n_comp'] for r in rows]):.0f}")
    print(f"  median time: {np.median([r['secs'] for r in rows]):.1f}s")
    ok = nm >= 0.75 * len(rows) and nf >= 0.75 * len(rows)
    print("\n=> SDF RECOVERS APPENDAGE INSTANCES ON REAL SCANS. The part label R3 needs can be"
          if ok else
          "\n=> SDF does NOT cleanly recover appendage instances on real scans. Template-level")
    print("   produced automatically, with code already in the repo."
          if ok else "   purity does not transfer to fragmented ethanol scans.")
    json.dump(rows, open(os.path.join(HERE, "r6_results_v2.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
