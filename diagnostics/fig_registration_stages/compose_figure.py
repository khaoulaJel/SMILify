"""Task 2 figure (style B): compose the Blender layers into the 180-mm, 300-dpi figure + vector PDF.

Each panel = fitted template (blue) laid over the scan silhouette (grey): grey that stays visible is scan
surface the fit has not reached. One crop box for all panels (the camera stays shared). Numbers come from
stages.csv unchanged.

Outputs (this folder): fig_registration_stages.png (300 dpi), fig_registration_stages.pdf (editable text);
layers/panel_<label>.png: bare text-free transparent panels.

    python diagnostics/fig_registration_stages/compose_figure.py
"""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = os.path.join(HERE, "layers")
PANELS = [("init", "init", "init"), ("H2_joint", "pose", "pose"),
          ("Stage_2_deform_coarse", "Stage_2", "Stage 2 (coarse)"), ("Stage_3_deform_fine", "Stage_3", "Stage 3 (fine)")]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
FIT_HEX, SCAN_HEX = "#2a78d6", "#b4b3ae"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return rf"{m} $\times$ 10$^{{{int(e)}}}$"


def over(top, bottom):
    a = top[..., 3:4]
    out_a = a + bottom[..., 3:4] * (1 - a)
    rgb = (top[..., :3] * a + bottom[..., :3] * bottom[..., 3:4] * (1 - a)) / np.clip(out_a, 1e-6, None)
    return np.dstack([rgb, out_a[..., 0]])


def main():
    with open(os.path.join(HERE, "stages.csv")) as f:
        rows = {r["stage"]: r for r in csv.DictReader(f)}
    scan = plt.imread(os.path.join(LAYERS, "scan.png"))
    comps = [over(plt.imread(os.path.join(LAYERS, f"fit_{lab}.png")), scan) for _, lab, _ in PANELS]

    a = np.max([c[..., 3] for c in comps], axis=0)
    ys, xs = np.nonzero(a > 0.02)
    pad = int(0.02 * a.shape[0])
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
    comps = [c[y0:y1, x0:x1] for c in comps]
    for (_, lab, _), c in zip(PANELS, comps):
        plt.imsave(os.path.join(LAYERS, f"panel_{lab}.png"), np.clip(c, 0, 1))

    W = 180 / 25.4
    gap = 0.12                          # inches between panels (room for the arrows)
    margin = 0.04                       # inches (1 mm) so leg tips never touch the figure edge
    pw = (W - 2 * margin - 3 * gap) / 4
    ph = pw * comps[0].shape[0] / comps[0].shape[1]
    top, metrics_h, legend_h = 0.20, 0.44, 0.16
    H = top + ph + metrics_h + legend_h
    fig = plt.figure(figsize=(W, H))

    def fx(x):
        return x / W

    def fy(y):  # y measured from the top, in inches
        return 1 - y / H

    for j, ((st, lab, title), c) in enumerate(zip(PANELS, comps)):
        x = margin + j * (pw + gap)
        ax = fig.add_axes([fx(x), fy(top + ph), fx(pw), ph / H])
        ax.imshow(np.clip(c, 0, 1), interpolation="antialiased")
        ax.axis("off")
        fig.text(fx(x + 0.02), fy(0.03), "abcd"[j], ha="left", va="top", fontsize=8, fontweight="bold", color=INK)
        fig.text(fx(x + 0.17), fy(0.035), title, ha="left", va="top", fontsize=7, color=INK)
        r = rows[st]
        vals = [("chamfer", sci(float(r["chamfer_l2"]))),
                (r"F$_{0.01}$", f"{float(r['fscore@0.01']):.3f}"),
                ("penetration", f"{int(round(float(r['pen_hard_triples'])))}")]
        cx = x + pw / 2
        for k, (name, v) in enumerate(vals):
            yy = top + ph + 0.07 + k * 0.115
            fig.text(fx(cx - 0.05), fy(yy), name, ha="right", va="top", fontsize=6, color=INK2)
            fig.text(fx(cx + 0.05), fy(yy), v, ha="left", va="top", fontsize=6, color=INK)
        if j < 3:
            ay = fy(top + ph + 0.185)  # in the free band between metric blocks, clear of leg tips
            fig.add_artist(FancyArrowPatch((fx(x + pw - 0.12), ay), (fx(x + pw + gap + 0.12), ay),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6,
                                           lw=0.6, color=MUTED))

    ly = top + ph + metrics_h + 0.03
    for k, (col, txt) in enumerate(((FIT_HEX, "fitted template"), (SCAN_HEX, "scan surface not covered by the fit"))):
        lx = 0.05 + k * 1.15
        fig.add_artist(Rectangle((fx(lx), fy(ly + 0.09)), fx(0.09), 0.09 / H, transform=fig.transFigure,
                                 facecolor=col, edgecolor="none"))
        fig.text(fx(lx + 0.13), fy(ly + 0.005), txt, ha="left", va="top", fontsize=6, color=INK2)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_registration_stages.{ext}"), dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote fig_registration_stages.png/.pdf;", f"{W * 25.4:.0f} x {H * 25.4:.0f} mm")


if __name__ == "__main__":
    main()
