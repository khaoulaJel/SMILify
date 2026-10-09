"""Task 1: is each output faithful to the scan, and where does it deviate? (user request 2026-10-09)

Per specimen:
  floor    measurement noise floor: distance of an independent raw sample set to the dense raw samples
           (what t1_measure's fidelity would read for a PERFECT copy of the scan)
  maps     per-vertex distance of each output (V0, V1, W2024) to the raw scan, in the raw frame (rigid
           transform from out/measure_<s>.json), coloured on ONE fixed scale (0 .. 0.5% of diagonal,
           magenta = above the scale); and the raw outer surface coloured by which versions lost it
           (grey kept by both, red lost by V0 only, blue lost by V1 only, black lost by both)
Writes coloured PLYs to /hpcwork/nao48500/holotype_task1/maps/ (rendered by t1_render_maps.py) and
out/fidelity_<s>.json with the floor and per-version deviation percentiles above the floor.
"""
import json
import os
import sys

import numpy as np
import trimesh
from matplotlib import cm
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t1_measure import RAW, OUT, WORKER, TAU, outer_samples  # noqa: E402

MAPS = "/hpcwork/nao48500/holotype_task1/maps"
SCALE = 0.005   # colour scale top, fraction of diagonal


def main(s):
    os.makedirs(MAPS, exist_ok=True)
    meas = json.load(open(os.path.join(HERE, "out", f"measure_{s}.json")))
    raw = trimesh.load(RAW.format(s=s), process=True)
    rp, outer = outer_samples(raw)
    ro = rp[outer]
    diag = float(np.linalg.norm(ro.max(0) - ro.min(0)))
    dense, _ = trimesh.sample.sample_surface(raw, 1_000_000, seed=2)
    tree = cKDTree(dense)
    probe, _ = trimesh.sample.sample_surface(raw, 200_000, seed=9)
    floor = tree.query(probe)[0] / diag * 100
    res = dict(specimen=s, floor_p50_pct=float(np.percentile(floor, 50)), floor_p99_pct=float(np.percentile(floor, 99)))
    lost = {}
    for v, path in (("V0", OUT), ("V1", OUT), ("W2024", WORKER)):
        p = path.format(v=v, s=s)
        if not os.path.exists(p) or "transform" not in meas.get(v, {}):
            continue
        m = trimesh.load(p, process=False)
        m.apply_transform(np.array(meas[v]["transform"]))
        d = tree.query(m.vertices)[0] / diag
        col = (cm.viridis(np.clip(d / SCALE, 0, 1)) * 255).astype(np.uint8)
        col[d > SCALE] = (255, 0, 255, 255)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=col)
        m.export(f"{MAPS}/{s}_{v}_deviation.ply")
        dd = d * 100
        res[v] = dict(dev_p50_pct=float(np.percentile(dd, 50)), dev_p99_pct=float(np.percentile(dd, 99)),
                      dev_max_pct=float(dd.max()), frac_above_scale=float((d > SCALE).mean()))
        osamp, _ = trimesh.sample.sample_surface(m, 400_000, seed=4)
        lost[v] = cKDTree(osamp).query(ro)[0] > TAU * diag
    if "V0" in lost and "V1" in lost:
        cols = np.full((len(ro), 4), (170, 170, 170, 255), np.uint8)
        cols[lost["V0"] & ~lost["V1"]] = (220, 30, 30, 255)
        cols[~lost["V0"] & lost["V1"]] = (30, 80, 230, 255)
        cols[lost["V0"] & lost["V1"]] = (0, 0, 0, 255)
        trimesh.PointCloud(ro, colors=cols).export(f"{MAPS}/{s}_raw_lost.ply")
        res["lost_outer_pct"] = dict(V0_only=float(100 * (lost["V0"] & ~lost["V1"]).mean()),
                                     V1_only=float(100 * (~lost["V0"] & lost["V1"]).mean()),
                                     both=float(100 * (lost["V0"] & lost["V1"]).mean()))
    json.dump(res, open(os.path.join(HERE, "out", f"fidelity_{s}.json"), "w"), indent=1)
    print(f"[fidelity] {s}: floor p50 {res['floor_p50_pct']:.3f}% p99 {res['floor_p99_pct']:.3f}% | " +
          " | ".join(f"{v} p50 {r['dev_p50_pct']:.3f} p99 {r['dev_p99_pct']:.3f} max {r['dev_max_pct']:.3f} >scale {100*r['frac_above_scale']:.2f}%"
                     for v, r in res.items() if isinstance(r, dict) and "dev_p50_pct" in r) +
          (f" | lost V0-only {res['lost_outer_pct']['V0_only']:.2f}% V1-only {res['lost_outer_pct']['V1_only']:.2f}% both {res['lost_outer_pct']['both']:.2f}%" if "lost_outer_pct" in res else ""), flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
