"""Q3d non-leg corruption: 20% of the 24 (leg, segment) units redirected to the true posed position of
the nearest TRUNK vertex (template label not leg/antenna/mandible). Same unit definition, frame and
format as Q3b S1's s1_make_targets.py; a probe reports the measured corrupted fraction."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common")); sys.path.insert(0, os.path.join(HERE, "..", "Q3b_controlled_shift"))
import anatomy_proxy as ap
from s1_make_targets import seg_unit
OUT = "/hpcwork/nao48500/review_methods/Q3b_S1"
dd = ap.load_dd(); vlab = ap.template_vertex_labels(dd)
unit_of = np.array([seg_unit(n) for n in vlab], dtype=object)
units = sorted({u for u in unit_of if u is not None})
members = {u: np.where(np.array([x == u for x in unit_of]))[0] for u in units}
trunk = np.where(ap.coarse_vec(vlab, "appendage") == "trunk")[0]
z = np.load(os.path.join(REPO, "diagnostics/moonshot/synth_power48/ground_truth.npz"))
rng = np.random.default_rng(2007)
T = np.empty_like(z["verts"], dtype=np.float64); frac = []
for i in range(len(z["verts"])):
    V = z["verts"][i].astype(np.float64); c = V.mean(0); sc = np.abs(V - c).max(); tgt = V.copy()
    tree = cKDTree(V[trunk]); nv = 0
    for j in rng.choice(len(units), int(round(0.2 * len(units))), replace=False):
        src = members[units[j]]; _, nn = tree.query(V[src]); tgt[src] = V[trunk[nn]]; nv += len(src)
    frac.append(nv / sum(len(m) for m in members.values())); T[i] = (tgt - c) / sc
np.savez_compressed(os.path.join(OUT, "targets_nonleg0.2.npz"), names=z["names"], verts=T.astype(np.float32),
                    mask=np.ones(T.shape[:2], bool), kind="nonleg", rate=0.2)
json.dump(dict(frac_leg_vertices_corrupted=float(np.mean(frac))), open(os.path.join(HERE, "out", "nonleg_probe.json"), "w"))
print(f"non-leg 0.2: corrupted {100*np.mean(frac):.1f}% of leg-unit vertices -> {OUT}/targets_nonleg0.2.npz")
