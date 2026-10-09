"""Q2 parity probe: leg-correctness of any target file against the regime's ground truth.
Run on targets_deployed.npz (re-implementation) and targets_generator.npz (original
generate_cse_correspondence_20260827.py on the same meshes). They should agree within sampling noise.
"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import anatomy_proxy as ap
OUT = "/hpcwork/nao48500/review_methods/Q2"
dd = ap.load_dd(); vlab = ap.template_vertex_labels(dd)
legv = np.array([str(x).startswith("l_") for x in vlab])
res = {}
for regime in ("REALPOSE", "REALSCALE"):
    g = np.load(os.path.join(OUT, regime, "meshes", "ground_truth.npz"))
    for k in ("deployed", "generator", "opart"):
        t = np.load(os.path.join(OUT, regime, f"targets_{k}.npz"))
        assert list(t["names"]) == list(g["names"]), (regime, k, "specimen order differs")
        acc = []
        for i in range(len(g["verts"])):
            V = g["verts"][i].astype(float); c = V.mean(0); sc = np.abs(V - c).max()
            m = t["mask"][i] & legv
            _, nn = cKDTree((V - c) / sc).query(t["verts"][i][m])
            acc.append(float((ap.coarse_vec(vlab[nn], "leg") == ap.coarse_vec(vlab[m], "leg")).mean()))
        res[f"{regime}/{k}"] = dict(leg_correct=float(np.mean(acc)), coverage=float(t["mask"].mean()))
        print(f"{regime:<10} {k:<10} leg-correct {np.mean(acc):.3f}  coverage {t['mask'].mean():.3f}")
json.dump(res, open(os.path.join(HERE, "out", "parity_probe.json"), "w"), indent=1)
