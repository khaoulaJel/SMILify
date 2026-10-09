import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT = "/w0/tmp/nao48500/login23-1_353989/claude-54434/-rwthfs-rz-cluster-home-nao48500-SMILify/9f42922d-7ac6-494a-aafb-70b3f65325c7/scratchpad"

# real numbers, RESULTS_R9_objective_audit_20260909.md
TOTAL_PROD = 0.000826
TOTAL_CORRECT = 0.001380
PCT_UP = (TOTAL_CORRECT / TOTAL_PROD - 1) * 100  # 67%

TERMS = [
    ("offset",    0.000240, 43),
    ("normal",    0.000145, 26),
    ("chamfer",   0.000116, 21),
    ("edge",      0.000047,  8),
    ("other 5 terms combined", 0.000006, 2),
]

INK = "#1B2620"
MUTED = "#5B6459"
FAINT = "#8B9488"
PANEL_EDGE = "#D7DED2"
BG = "#FFFFFF"
PROD = "#0E8C79"     # production -- geometrically preferred, anatomically wrong
CORRECT = "#D9730D"  # anatomically correct -- costs more, every term
PROD_SOFT = "#E1F1EC"
CORRECT_SOFT = "#FBE9D9"

matplotlib.rcParams["font.family"] = "Liberation Sans"
MONO = "Liberation Mono"

fig = plt.figure(figsize=(15.5, 9.2), dpi=200)
fig.patch.set_facecolor(BG)

# ---------------------------------------------------------------- header
fig.text(0.045, 0.955, "SMILIFY · R9 — FULL OBJECTIVE AUDIT · 12 SPECIMENS",
          fontsize=12.5, color=FAINT, weight="bold", family=MONO)
fig.text(0.045, 0.905, "GEOMETRICALLY CLOSE ≠ ANATOMICALLY CORRECT",
          fontsize=30, weight="bold", color=INK, family="Liberation Sans")
fig.text(0.045, 0.865,
          "The production fitting objective itself scores lower — “better” — at the "
          "anatomically WRONG solution than at the anatomically correct one.",
          fontsize=14, color=MUTED, family="Liberation Sans")

# ==================================================================
# PANEL A -- total objective, two bars
# ==================================================================
axA = fig.add_axes([0.05, 0.10, 0.26, 0.68])
axA.set_facecolor(BG)
axA.axis("off")

bar_w = 0.5
xs = [0, 1]
vals = [TOTAL_PROD, TOTAL_CORRECT]
cols = [PROD, CORRECT]
ymax = TOTAL_CORRECT * 1.32

for x, v, c in zip(xs, vals, cols):
    axA.add_patch(mpatches.Rectangle((x - bar_w/2, 0), bar_w, v, linewidth=0, facecolor=c))
    axA.text(x, v + ymax*0.035, f"{v:.5f}", ha="center", fontsize=15, weight="bold",
              color=c, family=MONO)

axA.text(0, -ymax*0.075, "PRODUCTION", ha="center", fontsize=12.5, weight="bold", color=INK)
axA.text(0, -ymax*0.12, "geometrically preferred", ha="center", fontsize=10, color=MUTED)
axA.text(1, -ymax*0.075, "ANATOMICALLY", ha="center", fontsize=12.5, weight="bold", color=INK)
axA.text(1, -ymax*0.12, "CORRECT", ha="center", fontsize=12.5, weight="bold", color=INK)
axA.text(1, -ymax*0.165, "every term costs more", ha="center", fontsize=10, color=MUTED)

# bracket + callout
bx0, bx1 = 0 + bar_w/2 + 0.06, 1 - bar_w/2 - 0.06
by = TOTAL_CORRECT * 1.06
axA.plot([bx0, bx0, bx1, bx1], [TOTAL_PROD*1.05, by, by, TOTAL_CORRECT*1.02],
          color=INK, lw=1.4, solid_capstyle="round")
axA.text((bx0+bx1)/2 + 0.55, by*1.02, f"+{PCT_UP:.0f}%", ha="left", va="bottom",
          fontsize=22, weight="bold", color=INK, family=MONO)

axA.set_xlim(-0.65, 1.9)
axA.set_ylim(-ymax*0.22, ymax)
fig.text(0.05 + 0.26*0.62/1.9*0 + 0.16, 0.795, "TOTAL OBJECTIVE", ha="center",
          fontsize=11.5, color=FAINT, weight="bold", family=MONO)

# ==================================================================
# PANEL B -- term-by-term delta breakdown
# ==================================================================
fig.text(0.40, 0.795, "WHERE THE EXTRA COST COMES FROM", ha="left",
          fontsize=11.5, color=FAINT, weight="bold", family=MONO)
fig.text(0.40, 0.765, "Δ = (anatomically-correct − production); all nine active terms "
          "move the same direction — none reverses.", fontsize=11.5, color=MUTED)

axB = fig.add_axes([0.40, 0.10, 0.54, 0.63])
axB.set_facecolor(BG)
axB.axis("off")

names = [t[0] for t in TERMS][::-1]
deltas = [t[1] for t in TERMS][::-1]
pcts = [t[2] for t in TERMS][::-1]
n = len(TERMS)
row_h = 1.0
max_d = max(deltas) * 1.62

for i, (name, d, p) in enumerate(zip(names, deltas, pcts)):
    y = i * row_h
    axB.add_patch(mpatches.Rectangle((0, y - 0.30), d, 0.60, linewidth=0,
                   facecolor=CORRECT, alpha=0.92 if p >= 8 else 0.55))
    label = name.upper() if p >= 8 else name
    axB.text(-max_d*0.02, y, label, ha="right", va="center", fontsize=13,
              weight=("bold" if p >= 8 else "normal"),
              color=(INK if p >= 8 else MUTED))
    axB.text(d + max_d*0.02, y, f"+{d:.6f}    {p}%", ha="left", va="center",
              fontsize=11.5, color=CORRECT if p >= 8 else FAINT, family=MONO,
              weight=("bold" if p >= 8 else "normal"))

axB.set_xlim(-max_d*0.62, max_d*1.02)
axB.set_ylim(-0.7, (n-1)*row_h + 0.7)

# ---------------------------------------------------------------- footer
fig.text(0.045, 0.045,
          "Production fit vs. an alternative fit of the same 12 specimens, constrained so each "
          "human landmark can only be satisfied by its anatomically correct part (mandible,\n"
          "antenna, clypeus, occiput). Same objective, same data, different anatomy — and the "
          "objective prefers the wrong one on every single term, including raw chamfer.",
          fontsize=10.5, color=FAINT, family=MONO)

fig.savefig(os.path.join(OUT, "slide11_r9_objective.png"))
fig.savefig(os.path.join(OUT, "slide11_r9_objective_white.png"), facecolor="white")
print("saved")
