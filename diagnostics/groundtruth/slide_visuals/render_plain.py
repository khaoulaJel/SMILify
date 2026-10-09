import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = "/w0/tmp/nao48500/login23-1_353989/claude-54434/-rwthfs-rz-cluster-home-nao48500-SMILify/9f42922d-7ac6-494a-aafb-70b3f65325c7/scratchpad/mesh_case"
d = np.load(os.path.join(OUT, "case.npz"))

V = d["fitted_verts"]
F = d["faces"].astype(np.int64)
if F.ndim == 3:
    F = F[0]

human_pt = np.array(d["human_landmark_fitter_frame"])
old_pt = V[int(d["old_idx"])]
floor_pt = V[int(d["floor_idx"])]

BODY = "#4A3524"


def cam_basis(elev, azim):
    er, ar = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
    world_up = np.array([0, 0, 1.0])
    right = np.cross(world_up, view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    return view, right, up


def render_mesh(ax, verts, faces, elev, azim, color=BODY):
    view, right, up = cam_basis(elev, azim)
    key = (view * 0.55 + right * -0.30 + up * 0.85)
    key /= np.linalg.norm(key)
    fill = (view * 0.7 + right * 0.5 + up * -0.15)
    fill /= np.linalg.norm(fill)
    rim = -view * 0.6 + up * 0.5
    rim /= np.linalg.norm(rim)
    lights = [(key, 0.85), (fill, 0.30), (rim, 0.18)]

    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn[nn == 0] = 1
    n = n / nn

    shade = np.zeros(len(tris))
    for ld, w in lights:
        shade += w * np.clip(n @ ld, 0, 1)
    ambient = 0.34
    b = np.clip(ambient + shade, 0.0, 1.15)

    depth = tris.mean(1) @ view
    order = np.argsort(depth)
    tris, b = tris[order], b[order]

    base = np.array(matplotlib.colors.to_rgb(color))
    cols = np.clip(base[None, :] * b[:, None], 0, 1)
    pc = Poly3DCollection(tris, facecolor=cols, alpha=1.0, edgecolor="none")
    ax.add_collection3d(pc)


def set_view(ax, center, radius, elev, azim):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


ELEV, AZIM = -20, 120
BODY_CENTER = (V.min(0) + V.max(0)) / 2
BODY_RADIUS = np.linalg.norm(V.max(0) - V.min(0)) / 2 * 0.62

cluster = np.stack([human_pt, old_pt, floor_pt])
ZOOM_CENTER = cluster.mean(0)
ZOOM_RADIUS = 0.16


def make(name, center, radius, transparent):
    fig = plt.figure(figsize=(12.8, 10.8), dpi=200)
    if not transparent:
        fig.patch.set_facecolor("white")
    else:
        fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94], projection="3d")
    if transparent:
        ax.patch.set_alpha(0.0)
    render_mesh(ax, V, F, ELEV, AZIM)
    set_view(ax, center, radius, ELEV, AZIM)
    suffix = "transparent" if transparent else "white"
    path = os.path.join(OUT, f"{name}_{suffix}.png")
    fig.savefig(path, transparent=transparent)
    print("saved", path)


make("full_view", BODY_CENTER, BODY_RADIUS, False)
make("full_view", BODY_CENTER, BODY_RADIUS, True)
make("mandible_zoom", ZOOM_CENTER, ZOOM_RADIUS, False)
make("mandible_zoom", ZOOM_CENTER, ZOOM_RADIUS, True)
