"""Mechanism plots for Intervention C, pose25 primary. Reads C_per_specimen.json and
C_d2b_correspondence_audit.json (already computed). No new model/GPU computation."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

per_spec = json.load(open(os.path.join(OUT, "C_per_specimen.json")))
d2b = json.load(open(os.path.join(OUT, "C_d2b_correspondence_audit.json")))

QUOTAS = ["baseline", "q05", "q10", "q20"]
QNUM = {"baseline": 0.113, "q05": 0.05, "q10": 0.10, "q20": 0.20}  # baseline ~= natural fraction

RUN_OF = {
    "baseline": "SYN_clean_w5",
    "q05": "SYN_clean_pose25_stratq05_w5",
    "q10": "SYN_clean_pose25_stratq10_w5",
    "q20": "SYN_clean_pose25_stratq20_w5",
}


def get(cond, qlabel):
    d = per_spec[cond]
    return d["base"] if qlabel == "baseline" else d["arms"][qlabel]


# ---- Plot 1: paired per-specimen slopes, leg_distal error, pose25 ----
fig, ax = plt.subplots(figsize=(7, 5))
specimens = sorted(per_spec["pose25"]["base"].keys())
xs = [0, 1, 2, 3]
for s in specimens:
    ys = [get("pose25", q)[s]["leg_distal"] for q in QUOTAS]
    catastrophic = s in per_spec["pose25"]["catastrophic"]
    ax.plot(xs, ys, marker="o", alpha=0.85, lw=2 if catastrophic else 1,
            color="crimson" if catastrophic else "steelblue", label=s if catastrophic else None)
ax.set_xticks(xs)
ax.set_xticklabels(QUOTAS)
ax.set_ylabel("leg_distal mean joint-position error (per specimen)")
ax.set_title("pose25: per-specimen leg_distal error, baseline -> quota\n(red = top-4 catastrophic at baseline)")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "C_plot1_paired_specimens_pose25.png"), dpi=130)
plt.close(fig)

# ---- Plot 2: antenna delta vs leg_distal delta, per specimen, q20 vs baseline ----
fig, ax = plt.subplots(figsize=(6, 6))
base = per_spec["pose25"]["base"]
for qlabel, color in [("q05", "gray"), ("q10", "orange"), ("q20", "crimson")]:
    arm = per_spec["pose25"]["arms"][qlabel]
    dd = np.array([base[s]["leg_distal"] - arm[s]["leg_distal"] for s in specimens])
    da = np.array([base[s]["antenna"] - arm[s]["antenna"] for s in specimens])
    ax.scatter(dd, da, label=qlabel, color=color, alpha=0.7)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("leg_distal error IMPROVEMENT (baseline - arm), + = better")
ax.set_ylabel("antenna error IMPROVEMENT (baseline - arm), + = better")
ax.set_title("pose25: per-specimen antenna vs leg_distal co-movement\n(bottom-right = leg improves, antenna regresses)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "C_plot2_antenna_vs_distal_pose25.png"), dpi=130)
plt.close(fig)

# ---- Plot 3: quota -> cross-leg confusion (pretarsus), pose0 and pose25 ----
fig, ax = plt.subplots(figsize=(7, 5))
for cond, runs, color in [
    ("pose0", ["SYN_clean_pose0_w5", "SYN_clean_pose0_stratq05_w5", "SYN_clean_pose0_stratq10_w5", "SYN_clean_pose0_stratq20_w5"], "steelblue"),
    ("pose25", ["SYN_clean_w5", "SYN_clean_pose25_stratq05_w5", "SYN_clean_pose25_stratq10_w5", "SYN_clean_pose25_stratq20_w5"], "crimson"),
]:
    xleg = []
    for run in runs:
        rows = [r for r in d2b if r["run"] == run]
        xleg.append(np.nanmean([r["cross_leg_pt"] for r in rows]))
    ax.plot([0, 0.05, 0.10, 0.20], xleg, marker="o", label=cond, color=color)
ax.set_xlabel("distal sampling quota (fraction of leg's own sample share)")
ax.set_ylabel("mean cross-leg confusion, pretarsus")
ax.set_title("Quota -> cross-leg confusion (pretarsus)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "C_plot3_quota_vs_xleg.png"), dpi=130)
plt.close(fig)

# ---- Plot 4: distribution (boxplot), leg_distal error, pose25, baseline vs each quota ----
fig, ax = plt.subplots(figsize=(7, 5))
data = [[get("pose25", q)[s]["leg_distal"] for s in specimens] for q in QUOTAS]
bp = ax.boxplot(data, labels=QUOTAS, showmeans=True)
for i, d in enumerate(data):
    x = np.random.default_rng(0).normal(i + 1, 0.04, size=len(d))
    ax.scatter(x, d, alpha=0.6, color="gray", s=15, zorder=3)
ax.set_ylabel("leg_distal mean joint-position error (per specimen)")
ax.set_title("pose25: distribution of per-specimen leg_distal error\n(points = individual specimens, not hidden inside the mean)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "C_plot4_distribution_pose25.png"), dpi=130)
plt.close(fig)

print("wrote 4 plots to", OUT)
