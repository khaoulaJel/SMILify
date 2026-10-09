"""Minimal, dependency-free orthographic mesh renderer (matplotlib 2D PolyCollection, painter's
algorithm, Lambert shading). Used because Blender is not available on the cluster; a bpy script
for true Blender screenshots is provided separately (blender_screenshots.py)."""
import numpy as np
from matplotlib.collections import PolyCollection

LIGHT = np.array([0.35, 0.55, 0.76])


def orient_frame(M, V=None):
    """Anatomical frame from the template: returns (lat, ap, dv) unit axes with SIGNS fixed so that
    +ap points to the head and +dv points dorsally (legs are ventral)."""
    from measure import body_frame
    V = M["v_template"] if V is None else V
    bf = body_frame(M)
    lat, ap, dv = bf[0].copy(), bf[1].copy(), bf[2].copy()
    jn = M["jnames"]; J = M["Jr"] @ M["v_template"]
    head, gaster = J[jn.index("b_h")], J[jn.index("b_a_5")]
    if (head - gaster) @ ap < 0: ap = -ap
    cox = J[[jn.index(n) for n in jn if "_co_" in n]].mean(0)
    if (cox - J[jn.index("b_t")]) @ dv > 0: dv = -dv
    if np.cross(ap, dv) @ lat < 0: lat = -lat           # keep right-handed
    return lat, ap, dv


VIEWS = {  # (screen-x axis, screen-y axis, toward-viewer axis) as (index, sign) into (lat, ap, dv)
    "anterior": ((0, -1), (2, 1), (1, 1)),     # full-face: look at the head from the front
    "dorsal":   ((0, 1), (1, 1), (2, 1)),
    "lateral":  ((1, 1), (2, 1), (0, 1)),
    "lateral_neg": ((1, -1), (2, 1), (0, -1)),
}


def project(points, axes, view):
    ax3 = np.stack(axes)                        # (3,3) rows lat, ap, dv
    P = np.atleast_2d(points) @ ax3.T           # coords in (lat, ap, dv)
    (ix, sx), (iy, sy), (iz, sz) = VIEWS[view]
    return np.stack([sx * P[:, ix], sy * P[:, iy], sz * P[:, iz]], axis=1)


def draw_mesh(ax, V, F, axes, view, color=(0.80, 0.84, 0.89), face_mask=None, alpha=1.0,
              edge=False, zorder=1):
    Q = project(V, axes, view)
    if face_mask is not None:
        F = F[face_mask]
    tri = Q[F]                                   # (f,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    facing = n[:, 2] > 0
    tri, n = tri[facing], n[facing]
    shade = 0.35 + 0.65 * np.clip(n @ LIGHT, 0, 1)
    order = np.argsort(tri[:, :, 2].mean(1))
    cols = np.clip(np.outer(shade[order], color), 0, 1)
    pc = PolyCollection(tri[order, :, :2], facecolors=cols,
                        edgecolors=cols if not edge else (0, 0, 0, .08), linewidths=0.15,
                        alpha=alpha, zorder=zorder)
    ax.add_collection(pc)
    ax.set_aspect("equal"); ax.axis("off")
    lo, hi = tri[:, :, :2].reshape(-1, 2).min(0), tri[:, :, :2].reshape(-1, 2).max(0)
    pad = 0.06 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    return Q


def faces_of(F, vidx):
    s = np.zeros(F.max() + 1, bool); s[vidx] = True
    return s[F].all(1)
