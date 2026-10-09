"""R8 roll-degeneracy PROBE -- the LANDMARK-FREE half of R8, re-measured.

R8's pass/fail criterion (the "straddle" test) read annotation `original` coordinates RAW and is
therefore VOID per diagnostics/gt_contamination_audit_20260916/AUDIT_20260916.md.

This probe re-measures only the part of R8 that never touches an annotation: how much the
reflective-symmetry residual of the body varies as the mirror plane is ROLLED about the specimen's
own long axis. If the residual is nearly flat in roll, reflective symmetry cannot single out the
sagittal plane -- a purely geometric statement about the scans, independent of any landmark frame.

Landmarks are used ONLY to enumerate the 12 specimen ids (a file listing), never as coordinates.

Outputs: r8_roll_degeneracy_PROBE.json
"""
import glob
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
N_SAMPLE = 4000
ROLLS = np.arange(0, 180, 15.0)  # degrees about the long axis
RNG = np.random.default_rng(0)


def reflect_residual(P, tree, normal, centre, scale):
    n = normal / np.linalg.norm(normal)
    R = P - 2 * ((P - centre) @ n)[:, None] * n
    return float(tree.query(R)[0].mean()) / scale * 100.0


def main():
    from scipy.spatial import cKDTree

    sids = sorted(json.load(open(p))["specimen_id"]
                  for p in glob.glob(os.path.join(REPO, "annotation/landmarks/*_traits.json")))
    rows = []
    for sid in sids:
        f = os.path.join(MESHDIR, f"{sid}_processed.obj")
        if not os.path.exists(f):
            print(f"{sid}: mesh missing -- skipped")
            continue
        m = trimesh.load(f, process=False)
        V = np.asarray(m.vertices, float)
        scale = float(np.linalg.norm(m.bounding_box.extents))
        sub = V[RNG.choice(len(V), min(N_SAMPLE, len(V)), replace=False)]
        c = sub.mean(0)
        X = sub - c
        evals, evecs = np.linalg.eigh(np.cov(X.T))
        long_axis = evecs[:, np.argmax(evals)]
        long_axis /= np.linalg.norm(long_axis)
        # an orthonormal pair spanning the plane perpendicular to the long axis
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(tmp @ long_axis) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(long_axis, tmp)
        u /= np.linalg.norm(u)
        w = np.cross(long_axis, u)
        tree = cKDTree(sub)
        res = [reflect_residual(sub, tree, np.cos(np.radians(t)) * u + np.sin(np.radians(t)) * w,
                               c, scale) for t in ROLLS]
        res = np.asarray(res)
        rows.append(dict(specimen=sid, scale=scale, long_axis=long_axis.tolist(),
                         rolls=ROLLS.tolist(), residual_pct=res.tolist(),
                         best=float(res.min()), worst=float(res.max()),
                         spread_pct_of_best=float((res.max() - res.min()) / res.min() * 100)))
        print(f"{sid[:38]:39s} best {res.min():5.2f}%  worst {res.max():5.2f}%  "
              f"spread {(res.max()-res.min())/res.min()*100:6.1f}% of best", flush=True)

    if rows:
        sp = np.array([r["spread_pct_of_best"] for r in rows])
        bs = np.array([r["best"] for r in rows])
        print(f"\n  n = {len(rows)} specimens, {len(ROLLS)} roll angles each")
        print(f"  median best-roll residual : {np.median(bs):.2f}% of body scale")
        print(f"  median spread across roll : {np.median(sp):.1f}% of the best residual")
    json.dump(rows, open(os.path.join(HERE, "r8_roll_degeneracy_PROBE.json"), "w"),
              indent=2, default=float)


if __name__ == "__main__":
    main()
