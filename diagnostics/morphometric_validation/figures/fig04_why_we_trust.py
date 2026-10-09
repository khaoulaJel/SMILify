"""FIGURE 4 -- "why we trust the measurement", the instrument-validation chain in one panel.

recoverable -> repeatable -> transferable -> biologically meaningful.
Every number is mined from a completed experiment; nothing new was run.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9})
BLUE, DARK, GREY = "#1f5fa8", "#0d2f52", "#9aa0a6"

STAGES = [
    ("RECOVERABLE", "can the measurement be\nrecovered when truth is known?",
     "4.3%", "median recovery error", "M2 — 48 synthetic,\nexact ground truth"),
    ("REPEATABLE", "does the same specimen give\nthe same answer?",
     "0.88%", "mean seed CV", "M3 — 3 seeds × 48"),
    ("TRANSFERABLE", "does it survive real,\nheterogeneous data?",
     "753/757", "admissible (99.5%)", "M4-A — 757 specimens,\n172 genera"),
    ("BIOLOGICALLY\nMEANINGFUL", "does it carry\nbiological signal?",
     "0.385", "HW ∝ mass slope\n(published 0.395)", "WOLO — 20 mass-\nlabelled Atta"),
]


def main():
    fig, ax = plt.subplots(figsize=(13.0, 4.5))
    ax.set_xlim(0, 13.4); ax.set_ylim(0, 4.6); ax.axis("off")
    w, gap = 2.85, 0.52
    x = 0.25
    for i, (name, q, val, unit, src) in enumerate(STAGES):
        ax.add_patch(FancyBboxPatch((x, 0.85), w, 3.05,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec=BLUE, lw=1.7, zorder=2))
        ax.add_patch(plt.Rectangle((x, 3.52), w, 0.38, fc=BLUE, ec="none", zorder=3))
        ax.text(x + w / 2, 3.71, name, ha="center", va="center", color="white",
                fontsize=9.2, weight="bold", zorder=4)
        ax.text(x + w / 2, 3.24, q, ha="center", va="center", fontsize=8.2, color="#555")
        ax.text(x + w / 2, 2.30, val, ha="center", va="center", fontsize=25,
                color=BLUE, weight="bold")
        ax.text(x + w / 2, 1.66, unit, ha="center", va="center", fontsize=8.6, color=DARK)
        ax.text(x + w / 2, 1.14, src, ha="center", va="center", fontsize=7.4,
                color="#888", style="italic")
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.06, 2.35), (x + w + gap - 0.06, 2.35),
                                         arrowstyle="-|>", mutation_scale=15, color=BLUE, lw=1.8))
        x += w + gap

    ax.text(6.7, 4.35, "Why we trust the measurement  —  head width",
            ha="center", fontsize=13.5, weight="bold", color=DARK)
    ax.text(6.7, 0.40,
            "Each stage is a separate pre-registered experiment. A measurement that fails any one "
            "of them does not enter the biological analysis.",
            ha="center", fontsize=8.8, color="#555", style="italic")
    p = f"{FIG}/fig04_why_we_trust.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    main()
