"""Render stills + a looping GIF for slide 8, "THE SCORE CAN LIE".

Loads the two saved single-specimen fits (fit_production.npz / fit_corrected.npz, produced by
slide8_score_can_lie.py) plus the target scan (mesh_target.obj), and:
  1. renders production-fit-vs-target and corrected-fit-vs-target stills (near-identical surfaces,
     confirming both are good SURFACE fits despite very different anatomy underneath),
  2. linearly interpolates ALL free pose/shape params (global_rot, joint_rot, betas, trans,
     deform_verts) between the production and corrected solutions over N frames and renders the
     mesh + a skeleton overlay (joint positions from smal(return_joints=True), edges from the
     model's kintree parents array) at each step,
  3. renders a final side-by-side/overlay frame with both skeletons in different colours over the
     target scan.
Assembles the interpolation frames into a looping GIF via imageio.

Deviation from a "pure" pose-only interpolation: betas and deform_verts are interpolated too
(not held fixed), because the CORRECT arm optimized shape+pose+deform jointly (same as
R3/R9's FREE_P) -- holding shape fixed while only moving pose would not reproduce either fitted
end-state. This does mean the interpolated surface bulges slightly mid-sequence rather than
staying perfectly rigid; this is a real interpolation artifact and is documented, not hidden.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402

FREE_P = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
N_FRAMES = 30
OUT = HERE
FIG_DPI = 110


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


def load_params(tag):
    d = np.load(os.path.join(OUT, f"fit_{tag}.npz"))
    return {k: torch.as_tensor(d[k]) for k in FREE_P}


def set_params(smal, params):
    with torch.no_grad():
        for k, v in params.items():
            getattr(smal, k).copy_(v)


def lerp(a, b, t):
    return {k: (1 - t) * a[k] + t * b[k] for k in a}


def draw_frame(ax, verts, faces, joints, parents, tgt_pts, mesh_color, title, lims,
               joints2=None, joints2_color=None):
    ax.clear()
    tri = matplotlib.tri.Triangulation(verts[:, 0], verts[:, 1], triangles=faces)
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=tri.triangles,
                     color=mesh_color, alpha=0.55, shade=True, linewidth=0)
    ax.scatter(tgt_pts[:, 0], tgt_pts[:, 1], tgt_pts[:, 2], s=0.15, color="dimgray", alpha=0.25)

    def plot_skel(j, color):
        for i, p in enumerate(parents):
            if p < 0 or p >= len(j):
                continue
            seg = np.stack([j[p], j[i]])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=2.5, zorder=10)
        ax.scatter(j[:, 0], j[:, 1], j[:, 2], color=color, s=14, zorder=11)

    plot_skel(joints, "crimson" if joints2 is None else "crimson")
    if joints2 is not None:
        plot_skel(joints2, joints2_color)

    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ranges = [hi - lo for lo, hi in lims]
    ax.set_box_aspect(ranges)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)
    ax.view_init(elev=25, azim=-75)


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
    tgt = readobj(os.path.join(OUT, "mesh_target.obj"))
    rng = np.random.default_rng(0)
    tgt_sub = tgt[rng.choice(len(tgt), size=min(20000, len(tgt)), replace=False)]

    allv = np.concatenate([v_prod, v_corr, tgt_sub], axis=0)
    mn, mx = allv.min(0), allv.max(0)
    ctr = (mn + mx) / 2
    pad = (mx - mn) * 0.08
    lims = [(mn[i] - pad[i], mx[i] + pad[i]) for i in range(3)]

    # --- stills: production vs target, corrected vs target ---
    for tag, v, j in [("production", v_prod, j_prod), ("corrected", v_corr, j_corr)]:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        draw_frame(ax, v, faces, j, parents, tgt_sub, "steelblue",
                   f"{tag} fit vs. target scan\n(surface: near-identical)", lims)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"still_{tag}_vs_target.png"), dpi=FIG_DPI)
        plt.close(fig)
        print("wrote", f"still_{tag}_vs_target.png")

    # --- final comparison still: both skeletons overlaid on target ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_frame(ax, v_corr, faces, j_prod, parents, tgt_sub, "lightgray",
               "production skeleton (red) vs. anatomically-corrected skeleton (gold)\n"
               "same surface fit, same target scan, two different anatomies",
               lims, joints2=j_corr, joints2_color="gold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "still_skeleton_comparison.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("wrote still_skeleton_comparison.png")

    # --- interpolation frames: production -> corrected, all free params ---
    frames = []
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(N_FRAMES + 1):
        t = i / N_FRAMES
        Pt = lerp(P_prod, P_corr, t)
        set_params(smal, Pt)
        with torch.no_grad():
            v, j = smal(return_joints=True)
        v = v[0].numpy(); j = j[0].numpy()
        title = f"pose interpolation  t={t:.2f}\n(surface stays similar; anatomy underneath moves)"
        draw_frame(ax, v, faces, j, parents, tgt_sub, "steelblue", title, lims)
        fig.tight_layout()
        fpath = os.path.join(OUT, f"frame_{i:03d}.png")
        fig.savefig(fpath, dpi=FIG_DPI)
        frames.append(imageio.imread(fpath))
        if i % 10 == 0:
            print(f"rendered frame {i}/{N_FRAMES}")
    plt.close(fig)

    # hold on last frame + a beat on first frame for a nicer loop, then ping-pong back
    seq = frames + frames[-5:] + frames[::-1][1:]
    imageio.mimsave(os.path.join(OUT, "slide8_score_can_lie.gif"), seq, duration=0.07, loop=0)
    print("wrote slide8_score_can_lie.gif")


if __name__ == "__main__":
    main()
