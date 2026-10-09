"""Polished renders for slide 8, "THE SCORE CAN LIE" -- v2 of render_slide8.py.

Same underlying data (fit_production.npz / fit_corrected.npz / mesh_target.obj, produced by
slide8_score_can_lie.py -- real optimisation, real ant scan, verified numbers in
slide8_numeric_results.json). This script only changes PRESENTATION: camera framing, whitespace
cropping, and color palette. Text/typography is intentionally NOT baked into these images --
each panel is rendered clean so it can be composited into the actual slide layout in HTML/CSS,
which gives real control over headline/caption placement instead of fighting matplotlib fig.text.

Outputs go to fig/ as independent panels (full-body, head-zoom, production/corrected-vs-target,
animation frames + gif). The original v1 diagnostic renders (still_*.png, frame_*.png,
slide8_score_can_lie.gif in the parent dir) are left untouched per repo convention.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402

FREE_P = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
N_FRAMES = 36
OUT_DIR = os.path.join(HERE, "fig")
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DPI = 200

BG = "#0b0f16"
MESH_COLOR = "#5aa9c9"
TARGET_COLOR = "#aab2c0"
WRONG_COLOR = "#ff4d5e"
RIGHT_COLOR = "#ffcf4d"
ELEV, AZIM = 10, 95


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


def load_params(tag):
    d = np.load(os.path.join(HERE, f"fit_{tag}.npz"))
    return {k: torch.as_tensor(d[k]) for k in FREE_P}


def set_params(smal, params):
    with torch.no_grad():
        for k, v in params.items():
            getattr(smal, k).copy_(v)


def lerp(a, b, t):
    return {k: (1 - t) * a[k] + t * b[k] for k in a}


def style_ax(ax, lims, elev=ELEV, azim=AZIM):
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect([hi - lo for lo, hi in lims])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(BG)


def plot_skel(ax, j, parents, color, lw=3.2, s=26, keep=None):
    for i, p in enumerate(parents):
        if p < 0 or p >= len(j):
            continue
        if keep is not None and not (keep[i] and keep[p]):
            continue
        seg = np.stack([j[p], j[i]])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=lw, zorder=10,
                solid_capstyle="round")
    pts = j if keep is None else j[keep]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=s, zorder=11,
               edgecolor=BG, linewidth=0.4)


def crop_mesh(v, faces, lims):
    """Physically crop verts/faces to a box -- mplot3d does NOT clip to set_*lim3d."""
    keep_v = np.all([(v[:, k] >= lims[k][0]) & (v[:, k] <= lims[k][1]) for k in range(3)], axis=0)
    keep_f = keep_v[faces].all(axis=1)
    f_sub = faces[keep_f]
    used = np.unique(f_sub)
    remap = -np.ones(len(v), dtype=int)
    remap[used] = np.arange(len(used))
    return v[used], remap[f_sub]


def draw_mesh(ax, verts, faces, alpha=0.42, color=MESH_COLOR):
    tri = mtri.Triangulation(verts[:, 0], verts[:, 1], triangles=faces)
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=tri.triangles,
                     color=color, alpha=alpha, shade=True, linewidth=0, antialiased=True)


def new_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_facecolor(BG)
    return fig, ax


def autocrop(path, pad=14, bg=BG):
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    bgc = np.array(Image.new("RGB", (1, 1), bg)).reshape(3)
    mask = (np.abs(arr.astype(int) - bgc).sum(axis=-1) > 10)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, arr.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, arr.shape[0])
    im.crop((x0, y0, x1, y1)).save(path)
    return (x0, y0, x1, y1)


def main():
    dev = "cpu"
    smal = SMAL3DFitter(batch_size=1, device=dev)
    parents = smal.smal_model.parents

    P_prod = load_params("production")
    P_corr = load_params("corrected")

    set_params(smal, P_prod)
    with torch.no_grad():
        v_prod, j_prod = smal(return_joints=True)
    v_prod = v_prod[0].numpy(); j_prod = j_prod[0].numpy()

    set_params(smal, P_corr)
    with torch.no_grad():
        v_corr, j_corr = smal(return_joints=True)
    v_corr = v_corr[0].numpy(); j_corr = j_corr[0].numpy()

    faces = smal.faces[0].numpy()
    tgt = readobj(os.path.join(HERE, "mesh_target.obj"))
    rng = np.random.default_rng(0)
    tgt_sub = tgt[rng.choice(len(tgt), size=min(20000, len(tgt)), replace=False)]

    allv = np.concatenate([v_prod, v_corr, tgt_sub], axis=0)
    mn, mx = allv.min(0), allv.max(0)
    pad = (mx - mn) * 0.04
    lims = [(mn[i] - pad[i], mx[i] + pad[i]) for i in range(3)]

    # ---------- 1. surface-match stills (production & corrected vs target), clean ----------
    for tag, v in [("production", v_prod), ("corrected", v_corr)]:
        fig, ax = new_fig(11, 5)
        draw_mesh(ax, v, faces)
        ax.scatter(tgt_sub[:, 0], tgt_sub[:, 1], tgt_sub[:, 2], s=0.2, color=TARGET_COLOR,
                   alpha=0.35, linewidth=0)
        style_ax(ax, lims)
        out = os.path.join(OUT_DIR, f"panel_{tag}_vs_target.png")
        fig.savefig(out, dpi=FIG_DPI, facecolor=BG)
        plt.close(fig)
        autocrop(out)
        print("wrote", out)

    # ---------- 2. full-body panel: both skeletons over the corrected surface ----------
    fig, ax = new_fig(11, 5)
    draw_mesh(ax, v_corr, faces, alpha=0.28, color="#3d4a5c")
    ax.scatter(tgt_sub[:, 0], tgt_sub[:, 1], tgt_sub[:, 2], s=0.18, color=TARGET_COLOR,
               alpha=0.22, linewidth=0)
    plot_skel(ax, j_prod, parents, WRONG_COLOR, lw=2.4, s=18)
    plot_skel(ax, j_corr, parents, RIGHT_COLOR, lw=2.4, s=18)
    style_ax(ax, lims)
    full_path = os.path.join(OUT_DIR, "panel_full_both_skeletons.png")
    fig.savefig(full_path, dpi=FIG_DPI, facecolor=BG)
    plt.close(fig)
    autocrop(full_path)
    print("wrote", full_path)

    # ---------- 3. head-zoom panel: crop to where the two skeletons actually diverge ----------
    jdist = np.linalg.norm(j_prod - j_corr, axis=1)
    jmid = (j_prod + j_corr) / 2
    anterior = jmid[:, 0] > np.median(jmid[:, 0])  # head cluster sits at the +x end of the body
    K = 7
    candidates = np.where(anterior)[0]
    hot = candidates[np.argsort(-jdist[candidates])[:K]]
    hot_pts = np.concatenate([j_prod[hot], j_corr[hot]], axis=0)
    hn, hx = hot_pts.min(0), hot_pts.max(0)
    hpad = (hx - hn) * 0.55 + 0.02
    zoom_lims = [(hn[i] - hpad[i], hx[i] + hpad[i]) for i in range(3)]

    v_corr_z, faces_z = crop_mesh(v_corr, faces, zoom_lims)
    tgt_mask = np.all([(tgt_sub[:, k] >= zoom_lims[k][0]) & (tgt_sub[:, k] <= zoom_lims[k][1])
                        for k in range(3)], axis=0)
    joint_keep = np.all([(np.minimum(j_prod[:, k], j_corr[:, k]) <= zoom_lims[k][1]) &
                          (np.maximum(j_prod[:, k], j_corr[:, k]) >= zoom_lims[k][0])
                          for k in range(3)], axis=0)

    fig, ax = new_fig(7, 7)
    if len(v_corr_z):
        draw_mesh(ax, v_corr_z, faces_z, alpha=0.30, color="#3d4a5c")
    if tgt_mask.any():
        ax.scatter(tgt_sub[tgt_mask, 0], tgt_sub[tgt_mask, 1], tgt_sub[tgt_mask, 2],
                   s=1.4, color=TARGET_COLOR, alpha=0.35, linewidth=0)
    plot_skel(ax, j_prod, parents, WRONG_COLOR, lw=4.2, s=60, keep=joint_keep)
    plot_skel(ax, j_corr, parents, RIGHT_COLOR, lw=4.2, s=60, keep=joint_keep)
    ax.set_xlim(zoom_lims[0]); ax.set_ylim(zoom_lims[1]); ax.set_zlim(zoom_lims[2])
    ax.set_box_aspect([hi - lo for lo, hi in zoom_lims])
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_facecolor(BG)
    zoom_path = os.path.join(OUT_DIR, "panel_head_zoom.png")
    fig.savefig(zoom_path, dpi=FIG_DPI, facecolor=BG)
    plt.close(fig)
    autocrop(zoom_path)
    print("wrote", zoom_path)

    # ---------- 4. interpolation frames -> looping gif, clean (no baked text) ----------
    frames = []
    fig, ax = new_fig(11, 5)
    for i in range(N_FRAMES + 1):
        t = i / N_FRAMES
        Pt = lerp(P_prod, P_corr, t)
        set_params(smal, Pt)
        with torch.no_grad():
            v, j = smal(return_joints=True)
        v = v[0].numpy(); j = j[0].numpy()
        ax.clear()
        ax.set_facecolor(BG)
        draw_mesh(ax, v, faces)
        ax.scatter(tgt_sub[:, 0], tgt_sub[:, 1], tgt_sub[:, 2], s=0.18, color=TARGET_COLOR,
                   alpha=0.30, linewidth=0)
        col = tuple((1 - t) * np.array([255, 77, 94]) / 255 + t * np.array([255, 207, 77]) / 255)
        plot_skel(ax, j, parents, col)
        style_ax(ax, lims)
        fpath = os.path.join(OUT_DIR, f"anim_{i:03d}.png")
        fig.savefig(fpath, dpi=FIG_DPI, facecolor=BG)
        autocrop(fpath)
        frames.append(fpath)
        if i % 12 == 0:
            print(f"rendered frame {i}/{N_FRAMES}")
    plt.close(fig)

    # pad every frame onto one common canvas so the GIF doesn't jitter/error
    imgs = [Image.open(p).convert("RGB") for p in frames]
    W = max(im.width for im in imgs)
    H = max(im.height for im in imgs)
    canvas_imgs = []
    for im in imgs:
        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        canvas_imgs.append(np.array(canvas))
    seq = canvas_imgs + canvas_imgs[-8:] + canvas_imgs[::-1][1:]
    gif_path = os.path.join(OUT_DIR, "slide8_score_can_lie_v2.gif")
    imageio.mimsave(gif_path, seq, duration=0.055, loop=0)
    print("wrote", gif_path)


if __name__ == "__main__":
    main()
