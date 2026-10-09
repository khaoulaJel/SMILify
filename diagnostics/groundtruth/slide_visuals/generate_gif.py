"""Slide 11 GIF: full ant -> zoom to mandible -> scan (grey) vs fitted template (translucent)
overlay, nearly coincident -> the model's assigned point (green) vs the true anatomical
location (orange) diverge -> labels -> the statement.

Every mesh, point and distance here is the same real data used in RESULTS_V6/V7: the fitted
surface of Aenictus_ceylonicus_CASENT0878084 against its real scan, and the real vertex the
pipeline names "mandibular_apex_r" against the real human-placed landmark.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import imageio.v2 as imageio

OUT = "/w0/tmp/nao48500/login23-1_353989/claude-54434/-rwthfs-rz-cluster-home-nao48500-SMILify/9f42922d-7ac6-494a-aafb-70b3f65325c7/scratchpad/mesh_case"
FRAME_DIR = os.path.join(OUT, "gif_frames")
os.makedirs(FRAME_DIR, exist_ok=True)

d = np.load(os.path.join(OUT, "case.npz"))
V = d["fitted_verts"]
F = d["faces"].astype(np.int64)
if F.ndim == 3:
    F = F[0]
sm = np.load(os.path.join(OUT, "scan_smooth_final.npz"))
T = sm["verts"]
FT = sm["faces"].astype(np.int64)

human_pt = np.array(d["human_landmark_fitter_frame"])   # orange -- actual anatomical location
old_pt = V[int(d["old_idx"])]                            # green  -- model's assigned point

TEMPLATE_COLOR = "#4A3524"
SCAN_COLOR = "#ACA89F"
GREEN = "#2F9E44"
ORANGE = "#D9730D"
INK = "#1B2620"
MUTED = "#5B6459"

matplotlib.rcParams["font.family"] = "Liberation Sans"

# ---------------------------------------------------------------- geometry / camera helpers
def cam_basis(elev, azim):
    er, ar = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
    world_up = np.array([0, 0, 1.0])
    right = np.cross(world_up, view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    return view, right, up


def render_mesh(ax, verts, faces, elev, azim, color, alpha=1.0, edges=False):
    view, right, up = cam_basis(elev, azim)
    key = (view * 0.55 + right * -0.30 + up * 0.85); key /= np.linalg.norm(key)
    fill = (view * 0.7 + right * 0.5 + up * -0.15); fill /= np.linalg.norm(fill)
    rim = -view * 0.6 + up * 0.5; rim /= np.linalg.norm(rim)
    lights = [(key, 0.85), (fill, 0.30), (rim, 0.18)]

    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True); nn[nn == 0] = 1
    n = n / nn
    shade = np.zeros(len(tris))
    for ld, w in lights:
        shade += w * np.clip(n @ ld, 0, 1)
    b = np.clip(0.34 + shade, 0.0, 1.15)

    depth = tris.mean(1) @ view
    order = np.argsort(depth)
    tris, b = tris[order], b[order]

    base = np.array(matplotlib.colors.to_rgb(color))
    cols = np.clip(base[None, :] * b[:, None], 0, 1)
    edge = "#00000055" if edges else "none"
    pc = Poly3DCollection(tris, facecolor=cols, alpha=alpha, edgecolor=edge, linewidth=0.28)
    ax.add_collection3d(pc)


def render_wireframe(ax, verts, faces, color, alpha=0.55, lw=0.55):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    segs = verts[e]
    lc = Line3DCollection(segs, colors=color, alpha=alpha, linewidths=lw)
    ax.add_collection3d(lc)


def set_view(ax, center, radius, elev, azim):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def project_fig(fig, ax, pt):
    x2, y2, _ = proj3d.proj_transform(pt[0], pt[1], pt[2], ax.get_proj())
    dx, dy = ax.transData.transform((x2, y2))
    return fig.transFigure.inverted().transform((dx, dy))


def crop(verts, faces, center, radius):
    """Radius crop, then keep only the component connected to the seed nearest `center` --
    otherwise a leg that merely swings through the crop sphere gets severed mid-tube and
    renders as a dark open hole (Poly3DCollection doesn't cap open meshes)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    mask = np.linalg.norm(verts - center, axis=1) < radius
    tri = faces
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    valid = mask[e[:, 0]] & mask[e[:, 1]]
    e = e[valid]
    n = len(verts)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = A + A.T
    _, labels = connected_components(A, directed=False)
    idxs = np.nonzero(mask)[0]
    seed = idxs[np.argmin(np.linalg.norm(verts[idxs] - center, axis=1))]
    keep = labels == labels[seed]
    face_mask = keep[faces].all(1)
    return faces[face_mask]


def ease(s):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(s, 0, 1))


def lerp(a, b, s):
    return a + (b - a) * s


# ---------------------------------------------------------------- fixed camera keyframes
# single fixed viewing angle throughout (the one already validated for the standalone
# full/zoom stills) -- only the framing (center, radius) dollies in, nothing rotates.
FULL_ELEV, FULL_AZIM = -20, 120
ZOOM_ELEV, ZOOM_AZIM = -20, 120
BODY_CENTER = (V.min(0) + V.max(0)) / 2
BODY_RADIUS = float(np.linalg.norm(V.max(0) - V.min(0)) / 2 * 0.62)
ZOOM_CENTER = np.stack([human_pt, old_pt]).mean(0)
ZOOM_RADIUS = 0.19
CROP_RADIUS = 0.24

F_zoom = crop(V, F, ZOOM_CENTER, CROP_RADIUS)
F_wire = crop(V, F, ZOOM_CENTER, ZOOM_RADIUS * 1.05)
FT_zoom = FT  # already cropped + display-smoothed in precompute_scan.py

FIGSIZE = (10.8, 10.8)
DPI = 150


def new_fig():
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    return fig, ax


# ---------------------------------------------------------------- stage schedule
stages = [
    ("A_full", 12),
    ("B_zoom", 22),
    ("C_scan_fadein", 10),
    ("D_hold", 8),
    ("E_green", 7),
    ("F_orange_line", 11),
    ("G_labels", 16),
    ("H_statement", 20),
    ("I_hold_end", 12),
]
frame_i = 0
frames = []

for stage_name, n in stages:
    for k in range(n):
        s = k / max(n - 1, 1)
        fig, ax = new_fig()

        if stage_name == "A_full":
            render_mesh(ax, V, F, FULL_ELEV, FULL_AZIM, TEMPLATE_COLOR)
            set_view(ax, BODY_CENTER, BODY_RADIUS, FULL_ELEV, FULL_AZIM)

        elif stage_name == "B_zoom":
            es = ease(s)
            center = lerp(BODY_CENTER, ZOOM_CENTER, es)
            radius = float(np.exp(lerp(np.log(BODY_RADIUS), np.log(ZOOM_RADIUS), es)))
            render_mesh(ax, V, F, ZOOM_ELEV, ZOOM_AZIM, TEMPLATE_COLOR)
            set_view(ax, center, radius, ZOOM_ELEV, ZOOM_AZIM)

        else:
            es = ease(s)
            elev, azim, center, radius = ZOOM_ELEV, ZOOM_AZIM, ZOOM_CENTER, ZOOM_RADIUS

            if stage_name == "C_scan_fadein":
                t_alpha = lerp(0.0, 0.65, es)
                s_alpha = lerp(0.0, 1.0, es)
            else:
                t_alpha, s_alpha = 0.65, 1.0

            render_mesh(ax, T, FT_zoom, elev, azim, SCAN_COLOR, alpha=s_alpha)
            render_wireframe(ax, V, F_wire, TEMPLATE_COLOR, alpha=t_alpha, lw=0.4)
            set_view(ax, center, radius, elev, azim)

            fig.canvas.draw()
            h_fig = project_fig(fig, ax, human_pt)
            o_fig = project_fig(fig, ax, old_pt)

            green_a = orange_a = line_a = 0.0
            if stage_name in ("E_green", "F_orange_line", "G_labels", "H_statement", "I_hold_end"):
                green_a = 1.0 if stage_name != "E_green" else ease(s)
            if stage_name in ("F_orange_line", "G_labels", "H_statement", "I_hold_end"):
                fs = ease(s) if stage_name == "F_orange_line" else 1.0
                orange_a = fs
                line_a = fs

            if green_a > 0:
                dot = mpatches.Circle(o_fig, 0.0125, transform=fig.transFigure,
                                       facecolor=GREEN, edgecolor="white", linewidth=2.0,
                                       zorder=52, alpha=green_a)
                fig.add_artist(dot)
            if orange_a > 0:
                if line_a > 0:
                    lp = o_fig + (h_fig - o_fig) * min(line_a * 1.4, 1.0)
                    ln = matplotlib.lines.Line2D([o_fig[0], lp[0]], [o_fig[1], lp[1]],
                                                  transform=fig.transFigure, color=INK,
                                                  lw=2.0, linestyle=(0, (5, 4)), zorder=51,
                                                  alpha=min(line_a * 1.4, 1.0), solid_capstyle="round")
                    fig.add_artist(ln)
                dot = mpatches.Circle(h_fig, 0.0125, transform=fig.transFigure,
                                       facecolor=ORANGE, edgecolor="white", linewidth=2.0,
                                       zorder=52, alpha=orange_a)
                fig.add_artist(dot)

            if stage_name in ("G_labels", "H_statement", "I_hold_end"):
                lab_a = ease(s) if stage_name == "G_labels" else 1.0
                fig.text(0.05, 0.93, "GEOMETRIC", fontsize=20, weight="bold", color=GREEN,
                          alpha=lab_a, family="Liberation Sans")
                fig.text(0.05, 0.895, "SURFACE ≈ MATCH", fontsize=13.5, color=GREEN,
                          alpha=lab_a, family="Liberation Sans")
                fig.text(0.05, 0.105, "ANATOMICAL", fontsize=20, weight="bold", color=ORANGE,
                          alpha=lab_a, ha="left", family="Liberation Sans")
                fig.text(0.05, 0.070, "LOCATION ≠ MATCH", fontsize=13.5, color=ORANGE,
                          alpha=lab_a, family="Liberation Sans")

            if stage_name in ("H_statement", "I_hold_end"):
                st_a = ease(s) if stage_name == "H_statement" else 1.0
                veil = mpatches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                           facecolor="white", alpha=0.62 * st_a, zorder=60)
                fig.add_artist(veil)
                fig.text(0.5, 0.56, "GEOMETRICALLY CLOSE", fontsize=30, weight="bold",
                          color=INK, alpha=st_a, ha="center", family="Liberation Sans", zorder=61)
                fig.text(0.5, 0.485, "≠ ANATOMICALLY CORRECT", fontsize=30, weight="bold",
                          color=INK, alpha=st_a, ha="center", family="Liberation Sans", zorder=61)
                fig.text(0.5, 0.41, "A good surface fit can still assign the wrong anatomy.",
                          fontsize=15, color=MUTED, alpha=st_a, ha="center",
                          family="Liberation Sans", zorder=61)

        path = os.path.join(FRAME_DIR, f"f{frame_i:04d}.png")
        fig.savefig(path)
        plt.close(fig)
        frames.append(path)
        frame_i += 1
        if frame_i % 10 == 0:
            print(frame_i, "/", sum(n for _, n in stages))

print("rendered", len(frames), "frames -- encoding gif")
imgs = [imageio.imread(p) for p in frames]
durations = [1 / 12] * len(imgs)
durations[-1] = 1.6
imageio.mimsave(os.path.join(OUT, "slide11_mandible.gif"), imgs, duration=durations, loop=0)
print("saved", os.path.join(OUT, "slide11_mandible.gif"))
