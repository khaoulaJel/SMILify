"""Task 1: plot the fidelity maps from t1_fidelity_maps.py as one figure per specimen.

Rows: lateral, dorsal (principal axes of the raw outer surface). Columns: raw outer surface coloured
by loss (grey kept by both, red lost by V0 only, blue lost by V1 only, black lost by both), then the
deviation-to-scan maps of V0, V1 and the 2024 mesh on ONE colour scale (0 .. 0.5% of the diagonal,
magenta above). Orthographic, depth-sorted so nearer points cover farther ones.
Writes /hpcwork/nao48500/holotype_task1/maps/<specimen>_fidelity.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

MAPS = "/hpcwork/nao48500/holotype_task1/maps"


def proj(P, C, ax_view, ax_x, ax_y, title, a, size=0.3):
    depth = P @ ax_view
    o = np.argsort(depth)
    a.scatter(P[o] @ ax_x, P[o] @ ax_y, c=C[o] / 255.0, s=size, linewidths=0)
    a.set_aspect("equal"); a.axis("off"); a.set_title(title, fontsize=9)


def main(s):
    cols = [("raw_lost", "raw: red lost by V0 only, blue lost by V1 only, black both")]
    cols += [(f"{v}_deviation", f"{v}: distance to scan (0-0.5% diag, magenta >)") for v in ("V0", "V1", "W2024")]
    clouds = []
    for key, title in cols:
        p = f"{MAPS}/{s}_{key}.ply"
        if not os.path.exists(p):
            continue
        g = trimesh.load(p, process=False)
        P = np.asarray(g.vertices)
        C = np.asarray(g.visual.vertex_colors if hasattr(g.visual, "vertex_colors") else g.colors)[:, :3]
        clouds.append((P, C, title))
    ref = clouds[0][0]
    c = ref.mean(0); _, ev = np.linalg.eigh(np.cov((ref - c).T)); ax = ev[:, ::-1]
    views = [("lateral", ax[:, 2], ax[:, 0], ax[:, 1]), ("dorsal", ax[:, 1], ax[:, 0], ax[:, 2])]
    fig, axs = plt.subplots(2, len(clouds), figsize=(5 * len(clouds), 6))
    for j, (P, C, title) in enumerate(clouds):
        for i, (vn, av, axx, axy) in enumerate(views):
            proj(P - c, C, av, axx, axy, f"{title}\n{vn}" if i == 0 else vn, axs[i, j])
    fig.suptitle(s, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{MAPS}/{s}_fidelity.png", dpi=110)
    print("[plot]", s)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
