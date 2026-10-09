"""Visual probe for the fit-independent GT registration: every expert joint drawn on its scan, in
three orthogonal views of the GT anatomical frame. Right-side joints blue, left orange, midline ink.
If the L/R swap for mirrored scenes were wrong, blue would sit on the animal's left (bottom of
the top view, where +right points up).

LIMITATION (read before trusting it for laterality): the frame's right axis is anterior x dorsal built
from the GT joints themselves, so this view checks internal consistency and scan placement, NOT
laterality. Independent laterality evidence = the production-fit swap test (5/5), weakest on
Leptogenys (35 vs 52% WL), whose frame cue is also weakest (chirality cos 0.33)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BENCH, "tools"))
from score import local_frame, read_obj
G = json.load(open(os.path.join(BENCH, "data/gt_joints_fitframe.json")))
sids = sorted(G)
fig, axs = plt.subplots(len(sids), 3, figsize=(9, 2.2 * len(sids)))
for r, sid in enumerate(sids):
    g = G[sid]; P = {n: np.asarray(v["fit"]) for n, v in g["joints"].items()}
    Fr = local_frame(P)
    V, _ = read_obj(f"/hpcwork/nao48500/worker_alt_data/{sid}_processed.obj")
    c = V.mean(0); V = (V - c) / np.abs(V - c).max()
    V = V[np.random.default_rng(0).choice(len(V), min(15000, len(V)), replace=False)]
    o = np.mean(list(P.values()), 0); to = lambda X: (np.asarray(X) - o) @ Fr.T
    Vl = to(V); names = list(P); X = to([P[n] for n in names])
    col = ["#2a78d6" if n.endswith("_r") else "#eb6834" if n.endswith("_l") else "#0b0b0b" for n in names]
    for k, (i, j, t) in enumerate([(0, 2, "top: anterior→, right↑"), (0, 1, "side: anterior→, dorsal↑"), (2, 1, "front: right→, dorsal↑")]):
        ax = axs[r, k]
        ax.scatter(Vl[:, i], Vl[:, j], s=.3, color="#d8d7d0", rasterized=True)
        ax.scatter(X[:, i], X[:, j], s=10, c=col, edgecolor="white", linewidth=.4)
        ax.set_aspect("equal"); ax.axis("off")
        if r == 0: ax.set_title(t, fontsize=8)
    axs[r, 0].text(-.05, .5, f"{sid.split('_CASENT')[0]}\n{g['tier']}{' (L/R swapped)' if g['labels_swapped'] else ''}",
                   transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=7)
fig.suptitle("Expert joints on their scans (fit-independent registration). blue = _r, orange = _l, black = midline",
             x=.01, ha="left", fontsize=9, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "gt_on_scan_PROBE.png"), dpi=160); print("ok")
