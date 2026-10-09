"""F02 (clean version): how the head-width joints were added -- minimal text, real renders."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from hw_core import *
from hw_render import *

BLUE, DARK, GREY, CL, CR = "#1f5fa8", "#0d2f52", "#b3bac2", "#d35400", "#1f5fa8"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, t, ec=BLUE, tc=DARK, fs=11, lw=1.6, weight="normal", fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, weight=weight, zorder=3)


def arrow(ax, a, b, color=BLUE, lw=1.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, zorder=1))


def head_view(ax, V, Pl, Pr, M, F, fh, axes, title):
    Rk, c, idx, _ = head_frame(V, M)
    Vh, pl, pr = (V - c) @ Rk, (Pl - c) @ Rk, (Pr - c) @ Rk
    draw_mesh(ax, Vh, F, axes, "anterior", face_mask=fh)
    Q = project(np.array([pl, pr]), axes, "anterior")
    ax.plot(Q[:, 0], Q[:, 1], color=DARK, lw=2.2, zorder=20)
    for q, col in zip(Q, (CL, CR)):
        ax.scatter(q[0], q[1], s=120, color=col, edgecolor="white", lw=1.5, zorder=21)
    ax.set_title(title, fontsize=11, color=DARK, pad=4)


def main():
    d = load_all(); M = d["M"]; F = d["faces"]; V0 = M["v_template"]; R = d["R"]
    axes = orient_frame(M)
    idx_head = np.where(M["dominant"] == M["jnames"].index("b_h"))[0]
    fh = faces_of(F, idx_head)

    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.15, 0.95, 0.95], wspace=.12)

    # ---- 1 skeleton
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0, 9.7, "1   Where they sit", fontsize=13, weight="bold", color=DARK)
    box(ax, 1.6, 7.0, 3.8, 1.05, "b_h  (head)", fc="#eaf1f9")
    box(ax, 6.2, 7.12, 3.6, .8, "mandibles,\nantennae", ec=GREY, tc="#888", fs=9, lw=1.1)
    arrow(ax, (5.42, 7.52), (6.18, 7.52), color=GREY, lw=1.1)
    for n, x, c in (("b_h_l", .4, CL), ("b_h_r", 4.4, CR)):
        box(ax, x, 3.0, 3.2, 1.05, n, ec=c, tc=c, weight="bold", lw=2.4, fs=12.5)
        arrow(ax, (3.5, 7.0), (x + 1.6, 4.1), color=c, lw=1.8)
    ax.text(3.6, 5.5, "new", ha="center", fontsize=10, color=DARK, style="italic")
    ax.text(4.0, 1.9, "two new child bones of b_h\n(measurement only — they do not deform the mesh)",
            ha="center", fontsize=9.5, color="#555")

    # ---- 2 steps
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0, 9.7, "2   Placement in Blender", fontsize=13, weight="bold", color=DARK)
    steps = ["Import the fitted model",
             "Show the template shape",
             "Place a bone at the widest\npoint of the head, each side",
             "Export joint distances"]
    y = 7.9
    for k, t in enumerate(steps):
        ax.add_patch(Circle((.9, y), .45, fc=BLUE, ec="none", zorder=3))
        ax.text(.9, y, str(k + 1), ha="center", va="center", color="white", weight="bold", fontsize=12, zorder=4)
        ax.text(1.8, y, t, va="center", fontsize=11.5, color=DARK)
        if k < 3:
            arrow(ax, (.9, y - .5), (.9, y - 1.35), lw=1.3)
        y -= 1.85

    # ---- 3 template and 4 registered specimen
    axt = fig.add_subplot(gs[2:4]); axt.set_xlim(0, 10); axt.set_ylim(0, 10); axt.axis("off")
    axt.text(.4, 9.7, "3   How they follow each specimen", fontsize=13, weight="bold", color=DARK)
    axt.text(5, 1.0, r"joint $= \sum_{i=1}^{10} w_i\, v_i$     "
                     r"(10 nearest template vertices, $w_i \propto 1/d_i$)",
             ha="center", fontsize=12, color=DARK)
    axt.text(5, .15, "same vertices and weights on every specimen, so the joints move with the fitted head",
             ha="center", fontsize=9.5, color="#555")
    i20 = d["labels"].index("20")
    PL0, PR0 = R[0] @ V0, R[1] @ V0
    sub = gs[2:4].subgridspec(3, 2, height_ratios=[.16, 1, .3], wspace=.08)
    ax = fig.add_subplot(sub[1, 0])
    head_view(ax, V0, PL0, PR0, M, F, fh, axes, "template (where the bones were placed)")
    ax = fig.add_subplot(sub[1, 1])
    head_view(ax, d["obs"][i20], d["PL"][i20], d["PR"][i20], M, F, fh, axes,
              "registered specimen 20 (47 mg)")
    p = f"{SHOW}/figures/F02b_how_the_joints_were_added_clean.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    main()
