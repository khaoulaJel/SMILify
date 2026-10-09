"""Diagrams: F02 how the head-width joints were added; F14 end-to-end measurement pipeline."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os
SHOW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f"{SHOW}/figures"
BLUE, DARK, GREY, CL, CR, GREEN = "#1f5fa8", "#0d2f52", "#9aa0a6", "#d35400", "#1f5fa8", "#2e7d4f"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, t, fc="white", ec=BLUE, tc=DARK, fs=9, lw=1.4, weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, zorder=3, weight=weight)


def arr(ax, a, b, color=BLUE, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=lw, zorder=1))


def save(fig, name):
    p = f"{FIG}/{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def f02():
    fig = plt.figure(figsize=(16, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.2], wspace=.05)

    # ---- A: where they sit in the skeleton
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("A   Position in the SMIL skeleton", fontsize=11, loc="left")
    box(ax, 3.6, 8.6, 2.8, .75, "b_t  (thorax, root)", fs=9)
    box(ax, 3.6, 6.9, 2.8, .75, "b_h  (head pivot)", fs=9, fc="#eaf1f9")
    arr(ax, (5, 8.6), (5, 7.68))
    ax.text(6.55, 8.1, "… gaster, legs, wings", fontsize=7.8, color="#888")
    kids = [("ma_l", 0.3), ("ma_r", 2.25), ("an_1_l", 4.2), ("an_1_r", 6.15)]
    for n, x in kids:
        box(ax, x, 4.9, 1.75, .7, n, fs=8.4, ec=GREY, tc="#555")
        arr(ax, (5, 6.9), (x + .87, 5.62), color=GREY, lw=1)
    ax.text(5, 4.35, "existing children (trained J_regressor rows)", ha="center", fontsize=8, color="#777")
    for n, x, c in (("b_h_l", 1.9, CL), ("b_h_r", 5.9, CR)):
        box(ax, x, 2.3, 2.2, .85, n, fs=10, ec=c, tc=c, weight="bold", lw=2.2)
        arr(ax, (5, 6.9), (x + 1.1, 3.18), color=c, lw=1.6)
    ax.text(5, 1.55, "NEW: two leaf bones, children of b_h\n"
                     "inert reporters — no skinning weights,\nthey never move the mesh or change the fit",
            ha="center", fontsize=8.6, color=DARK)

    # ---- B: the placement procedure in Blender
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("B   How they were placed (SMIL Model Importer, Blender)", fontsize=11, loc="left")
    steps = [
        ("1", "Import fit", "Direct Import SMIL Model: pkl + ATTA20 npz\nPCA / clean_mesh / symmetrise / regress_joints = OFF"),
        ("2", "Show the template", "set ALL shape-key values to 0 → Basis mesh only\n(exporter reads Basis; keys are additive)"),
        ("3", "Mark the widest point", "Object Mode, snap 3D cursor to the head surface\nat the widest point of the capsule, full-face"),
        ("4", "Add the bone", "Armature Edit Mode → Add ▸ Single Bone at cursor\nname b_h_l / b_h_r, parent to b_h"),
        ("5", "Export", "Morphometry panel: load body-length CSV →\nExport Joint Distances"),
    ]
    y = 8.55
    for num, head, body in steps:
        ax.add_patch(Circle((.55, y + .38), .34, fc=BLUE, ec="none", zorder=3))
        ax.text(.55, y + .38, num, ha="center", va="center", color="white", weight="bold", fontsize=10, zorder=4)
        ax.text(1.15, y + .62, head, fontsize=9.6, weight="bold", color=DARK, va="center")
        ax.text(1.15, y + .08, body, fontsize=8.2, color="#444", va="center")
        if num != "5":
            arr(ax, (.55, y), (.55, y - .75), lw=1.1)
        y -= 1.72
    ax.text(5.1, .05, "console:  [joint distances] J_regressor: 55 trained rows preserved, "
                     "2 reporter rows computed (b_h_l, b_h_r)", ha="center", fontsize=7.6,
            family="DejaVu Sans Mono", color=GREEN)

    # ---- C: what the exporter computes, and why it follows every specimen
    ax = fig.add_subplot(gs[2]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("C   What happens on every registered specimen", fontsize=11, loc="left")
    # template patch
    rng = np.random.default_rng(3)
    pts = np.c_[rng.uniform(.5, 3.5, 22), rng.uniform(6.2, 9.0, 22)]
    bone = np.array([2.0, 7.6])
    dd = np.linalg.norm(pts - bone, axis=1); near = np.argsort(dd)[:10]
    w = 1 / dd[near]; w /= w.sum()
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color="#c9d1da", zorder=2)
    for i, wi in zip(near, w):
        ax.plot([pts[i, 0], bone[0]], [pts[i, 1], bone[1]], color=CL, lw=.7, alpha=.5)
        ax.scatter(*pts[i], s=30 + 400 * wi, color=CL, alpha=.8, edgecolor="white", lw=.6, zorder=3)
    ax.scatter(*bone, marker="*", s=260, color=CL, edgecolor="white", lw=1, zorder=4)
    ax.text(2.0, 5.75, "template (Basis):\nbone placed by hand;\n10 nearest vertices, w ∝ 1/d",
            ha="center", fontsize=8.4, color="#444", va="top")
    # arrow to specimen
    arr(ax, (4.0, 7.6), (5.9, 7.6), lw=1.8)
    ax.text(4.95, 8.0, "same vertex IDs,\nsame weights", ha="center", fontsize=8, color=BLUE)
    pts2 = pts * np.array([1.18, 1.0]) + np.array([5.2, -.25]) + rng.normal(0, .06, pts.shape)
    bone2 = (w[:, None] * pts2[near]).sum(0)
    ax.scatter(pts2[:, 0], pts2[:, 1], s=14, color="#c9d1da", zorder=2)
    for i, wi in zip(near, w):
        ax.scatter(*pts2[i], s=30 + 400 * wi, color=CL, alpha=.8, edgecolor="white", lw=.6, zorder=3)
    ax.scatter(*bone2, marker="*", s=260, color=CL, edgecolor="white", lw=1, zorder=4)
    ax.text(7.9, 5.75, "registered specimen:\nvertices moved by the fit →\njoint moves with them",
            ha="center", fontsize=8.4, color="#444", va="top")
    # equations
    box(ax, .4, 1.1, 9.2, 3.0, "", ec="#cfd8e3", fc="#f7f9fc")
    ax.text(5, 3.55, r"$\mathbf{j}_{s} = \sum_{i \in \mathcal{N}_{10}} w_i\, \mathbf{v}_{i,s}$", ha="center", fontsize=14)
    ax.text(5, 2.55, r"$\mathrm{HW}_s = \|\mathbf{j}^{\,l}_s - \mathbf{j}^{\,r}_s\| \times "
                     r"\frac{\mathrm{BL}^{\mathrm{mm}}_s}{\mathrm{BL}^{\mathrm{model}}_s}$", ha="center", fontsize=14)
    ax.text(5, 1.5, "BL = b_t–b_a_5 distance: measured in mm for the specimen,\n"
                    "and in model units on the same fit — converts model units to mm",
            ha="center", fontsize=8, color="#555")
    fig.suptitle("How the head-width joints were added to the model", fontsize=14, y=1.0)
    save(fig, "F02_how_the_joints_were_added")


def f14():
    fig, ax = plt.subplots(figsize=(16, 4.6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4.6); ax.axis("off")
    stages = [("20 Atta\nμCT meshes", "01–20.obj\n1.1–47.1 mg"),
              ("SMIL fit", "OmniAnt_25PCs\nstock fitter (master)"),
              ("Blender import", "one shape key\nper specimen"),
              ("add b_h_l / b_h_r", "placed on Basis;\n10-vertex regressor"),
              ("export", "joint distances CSV\n(model units)"),
              ("scale to mm", "× BL_mm / BL_model\nper specimen"),
              ("head width (mm)", "log–log vs mass\nslope 0.379")]
    x = .2; w = 2.0
    for k, (h, sub) in enumerate(stages):
        last = k == len(stages) - 1
        box(ax, x, 2.2, w, 1.3, h, fc=BLUE if last else "white", tc="white" if last else DARK,
            fs=10, weight="bold")
        ax.text(x + w / 2, 1.85, sub, ha="center", va="top", fontsize=8.2, color="#555")
        if not last:
            arr(ax, (x + w + .02, 2.85), (x + w + .26, 2.85), lw=1.6)
        x += w + .28
    checks = ["regressed joints reproduced in Python (0.05%)", "joints on the scan surface, 98.8% of max width",
              "same anatomical locus on all 20", "pose-invariant (0.06%)", "left/right symmetric (0.9%)",
              "tighter scaling than the 2025 reference"]
    ax.text(8, .62, "validation:  " + "   ·   ".join(checks), ha="center", fontsize=8.6, color=GREEN)
    ax.set_title("From μCT scan to a head-width measurement", fontsize=13.5, pad=4)
    save(fig, "F14_measurement_pipeline")


if __name__ == "__main__":
    f02(); f14()
