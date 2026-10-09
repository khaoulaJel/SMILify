"""Diagram figures: measurement pipeline (3), failure map (14), validated operating regime (15)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9})
BLUE, DARK, GREY, RED, GREEN, AMBER = "#1f5fa8", "#0d2f52", "#9aa0a6", "#c0392b", "#2e7d4f", "#e6a020"


def box(ax, x, y, w, h, text, fc="white", ec=BLUE, tc=DARK, fs=9.4, lw=1.4, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=3, weight=weight)


def arrow(ax, x1, y1, x2, y2, color=BLUE, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, zorder=1))


def fig_pipeline():
    """TASK 3 -- before/after: what registration buys you."""
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 6.7); ax.axis("off")

    ax.text(2.0, 6.28, "BEFORE", fontsize=12, weight="bold", color=GREY, ha="center")
    box(ax, 0.7, 4.6, 2.6, 0.72, "raw scan", ec=GREY, tc="#555")
    arrow(ax, 2.0, 4.6, 2.0, 4.0, GREY)
    box(ax, 0.7, 3.2, 2.6, 0.72, "arbitrary geometry\n(own units, own pose)", ec=GREY, tc="#555", fs=8.6)
    arrow(ax, 2.0, 3.2, 2.0, 2.6, GREY)
    box(ax, 0.7, 1.8, 2.6, 0.72, "not directly comparable\nbetween specimens",
        ec=GREY, tc="#555", fs=8.6)
    ax.text(2.0, 1.2, "measurements are\nspecimen-specific", ha="center", fontsize=8.4,
            color=RED, style="italic")

    ax.add_patch(plt.Rectangle((4.05, 1.0), 0.02, 5.0, color="#dddddd"))

    ax.text(8.2, 6.28, "AFTER", fontsize=12, weight="bold", color=BLUE, ha="center")
    steps = [("raw scan", 5.35), ("SMILify registration", 4.42),
             ("common anatomical model", 3.49), ("validated measurement locus", 2.56),
             ("head width", 1.63)]
    for i, (t, yy) in enumerate(steps):
        fc = "#eaf1f9" if i in (2, 3) else "white"
        w = "bold" if i == 4 else "normal"
        box(ax, 6.5, yy, 3.4, 0.72, t, fc=fc, weight=w)
        if i < len(steps) - 1:
            arrow(ax, 8.2, yy, 8.2, yy - 0.21)
    arrow(ax, 8.2, 1.63, 8.2, 1.30)
    box(ax, 6.5, 0.50, 3.4, 0.72, "biological scaling", fc=BLUE, ec=BLUE, tc="white", weight="bold")

    ax.annotate("every specimen expressed\nin the SAME anatomical frame",
                xy=(9.9, 3.85), xytext=(10.25, 4.75), fontsize=8.4, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax.annotate("measured at a definition,\nnot at a coordinate",
                xy=(9.9, 2.92), xytext=(10.25, 2.15), fontsize=8.4, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax.set_title("Registration turns incomparable scans into a common measurement instrument",
                 fontsize=12, pad=10)
    p = f"{FIG}/fig03_pipeline.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", p)


def fig_failure_map():
    """TASK 14 -- compress the whole investigation into one matrix."""
    rows = ["mesh topology", "leg correspondence", "proximal pose", "anterior deformation",
            "landmark identity", "absolute scale"]
    cols = ["geometry", "correspondence", "pose", "anatomy", "measurement"]
    hits = {0: [0], 1: [1], 2: [2], 3: [3], 4: [3, 4], 5: [4]}
    status = ["solved / controlled", "bounded", "bounded", "controlled",
              "unresolved", "excluded from biological analysis"]
    scol = {"solved / controlled": GREEN, "controlled": GREEN, "bounded": AMBER,
            "unresolved": RED, "excluded from biological analysis": "#6a5acd"}

    fig, ax = plt.subplots(figsize=(10.4, 4.5))
    ax.set_xlim(-0.5, len(cols) + 2.6); ax.set_ylim(-0.95, len(rows) - 0.45)
    ax.invert_yaxis(); ax.axis("off")
    for j, c in enumerate(cols):
        ax.text(j, -0.62, c, ha="center", fontsize=9.2, weight="bold", color=DARK, rotation=0)
    for i, r in enumerate(rows):
        ax.text(-0.62, i, r, ha="right", va="center", fontsize=9.4, color=DARK)
        for j in range(len(cols)):
            ax.add_patch(plt.Rectangle((j - .42, i - .34), .84, .68, fc="#f6f7f9",
                                       ec="white", lw=1.4))
        for j in hits[i]:
            ax.add_patch(plt.Rectangle((j - .42, i - .34), .84, .68, fc=BLUE, ec="white", lw=1.4))
            ax.text(j, i, "✓", ha="center", va="center", color="white", fontsize=12, weight="bold")
        ax.text(len(cols) + 0.15, i, status[i], va="center", fontsize=8.8,
                color=scol[status[i]], weight="bold")
    ax.text(len(cols) + 0.15, -0.62, "status", fontsize=9.2, weight="bold", color=DARK)
    ax.text((len(cols) - 1) / 2, -1.3, "Where each problem lives, and how far it was taken",
            ha="center", fontsize=12, weight="bold", color=DARK)
    fig.text(0.5, 0.02, "Anatomical identity is the one unresolved frontier; absolute scale is "
                         "excluded by identifiability, not by difficulty.",
             ha="center", fontsize=9, style="italic", color="#555")
    p = f"{FIG}/fig14_failure_map.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", p)


def fig_operating_regime():
    """TASK 15 -- what SMILify can currently measure. The scientific product."""
    tiers = [("PRIMARY", ["head width (HW)   —  99.5% admissible, genus signal robust",
                          "head length (HL)  —  93.9%, genus-structured failure; see caveat"],
              BLUE, "may carry a biological claim"),
             ("COVARIATE", ["Weber's length (WL)", "total body length (TBL)"], "#6b9ac4",
              "normaliser / size axis only"),
             ("CONDITIONAL", ["scape length (SL)"], AMBER, "matched pose only — 6.1% pose Δ"),
             ("EXCLUDED", ["mandible length (ML)", "petiole length (PetL)",
                           "gaster length (GL)", "hind femur (FL)"], GREY,
              "fails accuracy, reproducibility or anatomical definition")]
    fig, ax = plt.subplots(figsize=(10.2, 5.9))
    ax.set_xlim(0, 10); ax.set_ylim(0.55, 10.15); ax.axis("off")
    y = 9.1
    for name, items, col, note in tiers:
        h = 0.62 + 0.52 * len(items)
        ax.add_patch(FancyBboxPatch((0.5, y - h), 8.9, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec=col, lw=1.8, zorder=2))
        ax.add_patch(plt.Rectangle((0.5, y - h), 0.13, h, fc=col, ec="none", zorder=3))
        ax.text(0.85, y - 0.34, name, fontsize=10.4, weight="bold", color=col, va="center")
        ax.text(9.25, y - 0.34, note, fontsize=8.2, color="#666", ha="right", va="center",
                style="italic")
        for k, it in enumerate(items):
            ax.text(1.35, y - 0.86 - 0.5 * k, "• " + it, fontsize=9.6, color=DARK, va="center")
        y -= h + 0.36
    ax.text(5.0, 9.75, "What SMILify can currently measure", fontsize=13, weight="bold",
            ha="center", color=DARK)
    fig.text(0.5, 0.045,
             "Use only measurements that pass computational recovery, repeatability, "
             "anatomical-definition and biological-validity checks.",
             ha="center", fontsize=9.2, color=DARK, weight="bold")
    fig.text(0.5, -0.005,
             "HL caveat (M4-A / M4-C): its failures are concentrated in particular genera, and its "
             "genus signal does not survive removal of singleton species.",
             ha="center", fontsize=8.0, color="#777", style="italic")
    p = f"{FIG}/fig15_operating_regime.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    fig_pipeline(); fig_failure_map(); fig_operating_regime()
