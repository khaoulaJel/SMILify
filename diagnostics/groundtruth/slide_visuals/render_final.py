import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

OUT = "/w0/tmp/nao48500/login23-1_353989/claude-54434/-rwthfs-rz-cluster-home-nao48500-SMILify/9f42922d-7ac6-494a-aafb-70b3f65325c7/scratchpad/mesh_case"
d = np.load(os.path.join(OUT, "case.npz"))
meta = json.load(open(os.path.join(OUT, "case_meta.json")))

V = d["fitted_verts"]
F = d["faces"].astype(np.int64)
if F.ndim == 3:
    F = F[0]

human_pt = np.array(d["human_landmark_fitter_frame"])
old_pt = V[int(d["old_idx"])]
floor_pt = V[int(d["floor_idx"])]

BODY = "#4A3524"        # warm dark chitin brown, lightened for legibility
GOOD = "#1F7A6C"
GOOD_SOFT = "#DCEEE9"
BAD = "#C0432A"
BAD_SOFT = "#F6E3DC"
INK = "#1B2620"
MUTED = "#5B6459"
FAINT = "#93A08D"

matplotlib.rcParams["font.family"] = "Liberation Sans"


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


def project_fig(fig, ax, pt):
    x2, y2, _ = proj3d.proj_transform(pt[0], pt[1], pt[2], ax.get_proj())
    dx, dy = ax.transData.transform((x2, y2))
    return fig.transFigure.inverted().transform((dx, dy))


ELEV, AZIM = -20, 120
BODY_CENTER = (V.min(0) + V.max(0)) / 2
BODY_RADIUS = np.linalg.norm(V.max(0) - V.min(0)) / 2 * 0.62

cluster = np.stack([human_pt, old_pt, floor_pt])
ZOOM_CENTER = cluster.mean(0)
ZOOM_RADIUS = 0.16

fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
fig.patch.set_facecolor("white")

ax_full = fig.add_axes([0.035, 0.10, 0.40, 0.80], projection="3d")
render_mesh(ax_full, V, F, ELEV, AZIM)
set_view(ax_full, BODY_CENTER, BODY_RADIUS, ELEV, AZIM)

ax_zoom = fig.add_axes([0.44, 0.10, 0.40, 0.80], projection="3d")
render_mesh(ax_zoom, V, F, ELEV, AZIM)
set_view(ax_zoom, ZOOM_CENTER, ZOOM_RADIUS, ELEV, AZIM)

fig.canvas.draw()

# ---- 2D overlay: ring on the full view marking the zoomed region ----
zc_fig = project_fig(fig, ax_full, ZOOM_CENTER)
ring = mpatches.Circle(zc_fig, 0.052, transform=fig.transFigure, fill=False,
                        edgecolor=BAD, linewidth=2.6, zorder=50)
fig.add_artist(ring)

# ---- 2D overlay: markers + connecting lines on the zoom panel ----
h_fig = project_fig(fig, ax_zoom, human_pt)
o_fig = project_fig(fig, ax_zoom, old_pt)
f_fig = project_fig(fig, ax_zoom, floor_pt)

line_good = matplotlib.lines.Line2D([h_fig[0], f_fig[0]], [h_fig[1], f_fig[1]],
                                     transform=fig.transFigure, color=GOOD, lw=3.6, zorder=51,
                                     solid_capstyle="round")
line_bad = matplotlib.lines.Line2D([h_fig[0], o_fig[0]], [h_fig[1], o_fig[1]],
                                    transform=fig.transFigure, color=BAD, lw=2.8, zorder=51,
                                    linestyle=(0, (5, 4)), solid_capstyle="round")
fig.add_artist(line_good)
fig.add_artist(line_bad)

for pt, col in [(h_fig, GOOD), (o_fig, BAD)]:
    dot = mpatches.Circle(pt, 0.0068, transform=fig.transFigure, facecolor=col,
                           edgecolor="white", linewidth=2.0, zorder=52)
    fig.add_artist(dot)

# ---- labels ----
fig.text(0.035, 0.925, "REGISTERED FIT · FULL VIEW", fontsize=13, color=FAINT,
          weight="bold", family="Liberation Sans")
fig.text(0.44, 0.925, "DETAIL · MANDIBLE, MAGNIFIED", fontsize=13, color=FAINT,
          weight="bold", family="Liberation Sans")

fig.text(h_fig[0] + 0.018, h_fig[1] + 0.028, "expert landmark", fontsize=13.5, color=GOOD,
          weight="bold", ha="left")
fig.text(h_fig[0] + 0.018, h_fig[1] + 0.006, "(true mandible apex)", fontsize=10.5, color=GOOD,
          ha="left")

fig.text(o_fig[0] + 0.018, o_fig[1] - 0.020, "vertex “mandibular_apex_r”", fontsize=13.5,
          color=BAD, weight="bold", ha="left")
fig.text(o_fig[0] + 0.018, o_fig[1] - 0.042, "the indexed landmark", fontsize=10.5, color=BAD,
          ha="left")

fig.text(zc_fig[0], zc_fig[1] - 0.075, "mandible", fontsize=11.5, color=BAD, ha="center")

# ---- number panel ----
sid_short = meta["sid"].replace("_", " ")
d_old = meta["d_old_pct_WL"]
d_floor = meta["d_floor_pct_WL"]
ratio = meta["ratio"]

ax_num = fig.add_axes([0.855, 0.10, 0.13, 0.80])
ax_num.set_axis_off()
ax_num.set_xlim(0, 1)
ax_num.set_ylim(0, 1)


def metric_box(y, h, val, label, color, soft, outline=False):
    box = mpatches.FancyBboxPatch((0.02, y), 0.96, h,
             boxstyle="round,pad=0,rounding_size=0.03",
             linewidth=(2.2 if outline else 0), edgecolor=(INK if outline else "none"),
             facecolor=("none" if outline else soft))
    ax_num.add_patch(box)
    ax_num.text(0.5, y + h * 0.62, val, ha="center", va="center", fontsize=27, weight="bold",
                color=color, family="Liberation Mono")
    ax_num.text(0.5, y + h * 0.24, label, ha="center", va="center", fontsize=10.5,
                color=color if not outline else MUTED, family="Liberation Mono")


metric_box(0.70, 0.17, f"{d_floor:.1f}%", "SURFACE\nDISTANCE", GOOD, GOOD_SOFT)
metric_box(0.44, 0.17, f"{d_old:.1f}%", "ANATOMICAL\nDISPLACEMENT", BAD, BAD_SOFT)
metric_box(0.18, 0.17, f"{ratio:.1f}×", "DISPLACEMENT\n/ SURFACE", INK, None, outline=True)

fig.text(0.855, 0.075, f"{sid_short}", fontsize=10, color=MUTED, family="Liberation Mono")
fig.text(0.855, 0.050,
         "% of Weber's length · fitted surface vs.\nhuman-placed mandible landmark",
         fontsize=9.3, color=FAINT, family="Liberation Mono", va="top")

fig.text(0.035, 0.045,
         "Same fit, two reference points for the same anatomical landmark: the nearest point on "
         "the fitted surface\nsits close to the true apex; the vertex the pipeline names "
         "“mandibular_apex” does not. Corpus median (n=12): 48.8% / 7.7% / 6.4×.",
         fontsize=10.5, color=MUTED, family="Liberation Sans")

fig.savefig(os.path.join(OUT, "slide11_real_mesh.png"), facecolor="white")
print("saved", os.path.join(OUT, "slide11_real_mesh.png"))
