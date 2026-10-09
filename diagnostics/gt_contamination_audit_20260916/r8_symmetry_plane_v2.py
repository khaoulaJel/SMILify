"""R8 v2 -- can the RAW SCAN supply a bilateral coordinate system?  [FRAME-CORRECTED]

RE-SCORING COPY, 2026-09-16.  The original `diagnostics/groundtruth/r8_symmetry_plane.py` scored its
straddle test against `landmarks[k]["original"]` read RAW (Blender Z-up) and is VOID per
AUDIT_20260916.md.  This copy loads the landmarks through `gt_frame_v2` (correction
`obj = (x, z, -y)`, on-surface assert), on the 11 usable `annotation/gt_expert/` specimens.

ADDED, and not in the original: a ROLL SWEEP of the reflection residual about the body's long axis,
computed on the SAME SDF body mask the plane is fitted to.  `r8_roll_degeneracy_PROBE.py` ran that
sweep on the FULL mesh and found the residual varies a median 53.5% of its best value, i.e. roll is
NOT degenerate there.  Running it on the body mask makes the probe and this experiment directly
comparable, so the "rotationally degenerate about the long axis" claim can be checked rather than
assumed.  The pass bar is restated for n = 11 at the original's 75% rate.


R7 failed three times, all at the same step: the midline. Fitted body centroid, fitted bilateral
part pairs, and expert bilateral JOINT pairs each produced a plane that put BOTH head-width
landmarks on the same side, on 11 of 12 specimens. The cause was diagnosed: these are pinned museum
specimens, and bilateral pairs disagree about the lateral axis by 12-27 degrees because the legs are
splayed asymmetrically. Every one of those constructions infers the midline FROM APPENDAGES.

R8 does not. It infers the plane from the approximately bilaterally symmetric BODY SURFACE itself:
reflect the body mass across a candidate plane and measure how well it overlaps itself.

DELIBERATELY MINIMAL. No hand-tuned weighting toward the "thick" region, no per-part masks, no
robust kernels. Just: SDF body mask -> plane optimisation -> validate. If it fails, the diagnosis
comes first and any weighting scheme second -- otherwise this becomes another sophisticated-looking
method that is really testing a hidden assumption, which is exactly how R7 went wrong.

VALIDATION, three independent checks, none of which uses the answer being sought:
  1. STRADDLE   the two head-width landmarks must fall on opposite sides. They are annotated
                independently of anything here.
  2. RESIDUAL   mean distance from the reflected body surface to the original, in % of body scale.
  3. STABILITY  angle between planes fitted to three random 50% subsamples of the body mask.

HARD STOPPING RULE, fixed before the run: if the straddle test does not pass on at least 9 of 12
specimens, the R7 branch closes. No further midline variants, SDF thresholds or symmetry
heuristics will be tried.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import gt_frame_v2 as GF  # noqa: E402

MESHDIR = "/hpcwork/nao48500/worker_alt_data"
RNG = np.random.default_rng(0)
THRESH, N_RAYS, CONE_DEG = 0.0307, 25, 120.0
N_SAMPLE = 4000
PASS_BAR = 9          # of 11: the original's 75% rate (9/12), ceil(0.75 * 11) = 9
ROLLS = np.arange(0, 180, 15.0)


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


def unit(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def fit_plane(P, inits, scale):
    """Plane minimising the mean distance from the reflected cloud to the original."""
    tree = cKDTree(P)
    c = P.mean(0)

    def cost(x):
        nrm = unit(x[0], x[1])
        d = x[2]
        R = P - 2 * ((P - c) @ nrm - d)[:, None] * nrm
        return float(tree.query(R)[0].mean())

    best, bx = np.inf, None
    for n0 in inits:
        th = np.arccos(np.clip(n0[2], -1, 1)); ph = np.arctan2(n0[1], n0[0])
        r = minimize(cost, [th, ph, 0.0], method="Nelder-Mead",
                     options=dict(maxiter=400, xatol=1e-4, fatol=1e-6))
        if r.fun < best:
            best, bx = r.fun, r.x
    return unit(bx[0], bx[1]), float(bx[2]), best / scale * 100, c


def roll_sweep(P, scale):
    """Reflection residual as the mirror plane is rolled about the cloud's long axis (through its
    centroid). Same construction as r8_roll_degeneracy_PROBE.py, applied to the BODY MASK."""
    c = P.mean(0)
    ev, evec = np.linalg.eigh(np.cov((P - c).T))
    ax = evec[:, np.argmax(ev)]; ax /= np.linalg.norm(ax)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ax, tmp); u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    tree = cKDTree(P)
    res = []
    for t in ROLLS:
        n = np.cos(np.radians(t)) * u + np.sin(np.radians(t)) * w
        R = P - 2 * ((P - c) @ n)[:, None] * n
        res.append(float(tree.query(R)[0].mean()) / scale * 100.0)
    return np.asarray(res)


def main():
    ann = GF.load_landmarks()          # FRAME-CORRECTED + on-surface assert, gt_expert, n = 11
    print(f"\n{len(ann)} specimens   |   PASS BAR: straddle on >= {PASS_BAR}/{len(ann)}\n")
    print(f"{'specimen':34s}{'body v':>8}{'residual':>10}{'stability':>11}{'straddle':>10}"
          f"{'rollspread':>12}{'t(s)':>7}")
    rows = []
    for sid, L in sorted(ann.items()):
        t0 = time.time()
        m = trimesh.load(os.path.join(MESHDIR, f"{sid}_processed.obj"), process=False)
        Vs = np.asarray(m.vertices)
        s = sdf_field(m, Vs, np.asarray(m.vertex_normals))
        scale = float(np.linalg.norm(m.bounding_box.extents))
        body = Vs[np.isfinite(s) & (s / scale >= THRESH)]
        if len(body) < 500:
            print(f"{sid[:33]:34s}  body mask too small -- skipped"); continue
        sub = body[RNG.choice(len(body), min(N_SAMPLE, len(body)), replace=False)]
        # inits: the three PCA axes of the body mass, no anatomical assumption
        ev = np.linalg.eigh(np.cov((sub - sub.mean(0)).T))[1].T
        nrm, d, resid, c = fit_plane(sub, ev, scale)

        side = lambda P: (np.atleast_2d(P) - c) @ nrm - d
        hr = float(side(L["head_width_r"])); hl = float(side(L["head_width_l"]))
        straddle = bool(np.sign(hr) != np.sign(hl))

        # stability across three independent 50% subsamples
        ns = []
        for _ in range(3):
            q = sub[RNG.choice(len(sub), len(sub) // 2, replace=False)]
            n2, _, _, _ = fit_plane(q, [nrm], scale)
            ns.append(n2 if n2 @ nrm > 0 else -n2)
        stab = float(np.degrees(np.arccos(np.clip(
            np.median([n2 @ nrm for n2 in ns]), -1, 1))))

        # roll degeneracy, on the same body mask (reconciliation with r8_roll_degeneracy_PROBE)
        rs = roll_sweep(sub, scale)
        spread = float((rs.max() - rs.min()) / rs.min() * 100.0)
        # angle between the optimised plane normal and the best-roll normal is NOT compared here:
        # the sweep is only asked whether roll is flat.

        rows.append(dict(specimen=sid, n_body=len(body), residual=resid, stability=stab,
                         straddle=straddle, hr=hr, hl=hl, normal=nrm.tolist(),
                         plane_centre=c.tolist(), plane_offset=float(d),
                         roll_residual_pct=rs.tolist(), roll_best=float(rs.min()),
                         roll_worst=float(rs.max()), roll_spread_pct_of_best=spread,
                         secs=time.time() - t0))
        print(f"{sid[:33]:34s}{len(body):>8}{resid:>9.2f}%{stab:>10.1f}d"
              f"{('YES' if straddle else 'no'):>10}{spread:>11.0f}%{time.time()-t0:>7.1f}",
              flush=True)

    ns_ = sum(1 for r in rows if r["straddle"])
    print(f"\n  straddle passes: {ns_}/{len(rows)}   (bar {PASS_BAR})")
    print(f"  median reflection residual: {np.median([r['residual'] for r in rows]):.2f}% of scale")
    print(f"  median plane stability under 50% subsampling: "
          f"{np.median([r['stability'] for r in rows]):.1f} degrees")
    print(f"  median roll spread on the BODY MASK: "
          f"{np.median([r['roll_spread_pct_of_best'] for r in rows]):.1f}% of the best residual"
          f"   (full mesh, r8_roll_degeneracy_PROBE: 53.5%)")
    print(f"  median best-roll residual on the body mask: "
          f"{np.median([r['roll_best'] for r in rows]):.2f}% of scale")
    ok = ns_ >= PASS_BAR
    print("\n=> THE RAW SCAN SUPPLIES A USABLE BILATERAL FRAME. R7 is worth exercising."
          if ok else
          "\n=> THE RAW SCAN DOES NOT SUPPLY A STABLE BILATERAL FRAME. Per the pre-set stopping")
    print("" if ok else
          "   rule, the R7 branch CLOSES -- no further midline variants, thresholds or symmetry\n"
          "   heuristics. Model-guided bilateral splitting is not the route to anatomical identity.")
    json.dump(rows, open(os.path.join(HERE, "r8_results_v2.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
