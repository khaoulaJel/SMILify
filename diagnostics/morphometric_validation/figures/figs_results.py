"""Presentation figures mined from already-completed experiments. No new experiments.
Tasks 5,6,7,9,10,11 of the final figure plan."""
import os, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(FIG), "out")
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9})
BLUE, GREY, RED, GREEN = "#1f5fa8", "#9aa0a6", "#c0392b", "#2e7d4f"

# M2 recovery error / M3 seed CV -- from RESULTS_A3_M3_20260909.md, roles from PHENOTYPE_PANEL.md
TRAITS = {  # trait: (recovery_err%, seedCV_mean%, seedCV_max%, role)
    "HW":   (4.31, 0.884,  6.57, "PRIMARY"),
    "HL":   (5.51, 1.081,  8.51, "PRIMARY"),
    "WL":   (4.90, 0.592,  1.39, "COVARIATE"),
    "SL":   (5.33, 2.419, 19.46, "CONDITIONAL"),
    "TBL":  (8.46, 0.936,  8.10, "COVARIATE"),
    "FL":   (9.93, 4.138, 40.89, "EXCLUDED"),
    "GL":  (14.81, 2.336, 29.23, "EXCLUDED"),
    "ML":  (21.73, 5.000, 44.33, "EXCLUDED"),
    "PetL":(34.41, 8.714, 93.10, "EXCLUDED"),
}
ROLE_C = {"PRIMARY": BLUE, "COVARIATE": "#6b9ac4", "CONDITIONAL": "#e6a020", "EXCLUDED": GREY}


def fig_trait_accuracy():
    """TASK 5 -- not every measurable quantity is equally trustworthy."""
    order = sorted(TRAITS, key=lambda t: TRAITS[t][0])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, t in enumerate(order):
        e, _, _, role = TRAITS[t]
        ax.barh(i, e, color=ROLE_C[role], height=0.62,
                edgecolor="white", lw=0.8)
        ax.text(e + 0.6, i, f"{e:.1f}%", va="center", fontsize=8.6, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("synthetic recovery error (%)  —  lower is better")
    ax.set_title("Not every measurable quantity is equally trustworthy", fontsize=11, pad=9)
    ax.set_xlim(0, 40)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.18, lw=0.5)
    h = [plt.Rectangle((0, 0), 1, 1, color=ROLE_C[r]) for r in ROLE_C]
    ax.legend(h, list(ROLE_C), fontsize=8, frameon=False, loc="lower right", title="panel role",
              title_fontsize=8)
    fig.tight_layout(); p = f"{FIG}/fig05_trait_accuracy.png"
    fig.savefig(p, dpi=300); fig.savefig(p.replace(".png", ".pdf")); print("wrote", p)


def fig_accuracy_vs_repeatability():
    """TASK 6 -- optimizer repeatability != measurement accuracy."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    # hand-placed offsets: the four good traits sit in a tight cluster and auto-placement collides
    OFF = {"HW": (-20, 10), "HL": (10, 6), "WL": (6, -13), "TBL": (10, 4), "SL": (10, 5),
           "FL": (10, 5), "GL": (10, 5), "ML": (10, 5), "PetL": (-14, 10)}
    for t, (e, cv, cvmax, role) in TRAITS.items():
        ax.scatter(e, cv, s=90, color=ROLE_C[role], edgecolor="white", lw=1.2, zorder=4)
        ax.annotate(t, (e, cv), textcoords="offset points", xytext=OFF[t], fontsize=9.5, zorder=6)
    ax.axvline(10, color=GREY, ls="--", lw=0.9); ax.axhline(3, color=GREY, ls="--", lw=0.9)
    ax.text(0.5, 2.55, "accurate\n+ reproducible", fontsize=8.4, color=GREEN, weight="bold", va="top")
    ax.text(11.0, 2.55, "reproducible\nbut inaccurate", fontsize=8.4, color="#b8860b",
            weight="bold", va="top")
    ax.text(11.0, 8.9, "unstable", fontsize=8.4, color=RED, weight="bold", va="top")
    ax.set_xlabel("measurement accuracy — synthetic recovery error (%)")
    ax.set_ylabel("optimizer repeatability — mean seed CV (%)")
    ax.set_title("Repeatability is not accuracy", fontsize=11, pad=9)
    ax.set_xlim(-1, 39); ax.set_ylim(-0.3, 10.0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.16, lw=0.5)
    fig.tight_layout(); p = f"{FIG}/fig06_accuracy_vs_repeatability.png"
    fig.savefig(p, dpi=300); fig.savefig(p.replace(".png", ".pdf")); print("wrote", p)


def fig_m4a_qc():
    """TASK 7 -- admissibility + the HL failure is structured, not random."""
    d = json.load(open(f"{OUT}/m4a_admissible_set.json"))
    rows = d["per_specimen"]
    g = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        g[r["genus"]][int(r["HL_class"] == "AP_inversion")] += 1
    big = {k: v for k, v in g.items() if sum(v) >= 5}
    aff = sorted(((k, v[1] / sum(v), v[1], sum(v)) for k, v in big.items() if v[1] > 0),
                 key=lambda t: t[1])
    nzero = sum(1 for v in big.values() if v[1] == 0)

    fig = plt.figure(figsize=(10.6, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.55)
    ax0 = fig.add_subplot(gs[0])
    for i, (lbl, ok, n) in enumerate([("head width", d["HW_admissible"], d["n"]),
                                      ("head length", d["HL_admissible"], d["n"])]):
        ax0.barh(i, n, color="#eeeeee", height=0.45)
        ax0.barh(i, ok, color=BLUE if i == 0 else "#6b9ac4", height=0.45)
        ax0.text(18, i, f"{ok} / {n}      {100*ok/n:.1f}%", va="center", ha="left",
                 color="white", fontsize=10.5, weight="bold")
        ax0.text(0, i + 0.36, lbl, fontsize=10)
    ax0.set_ylim(-0.5, 1.75); ax0.set_yticks([]); ax0.set_xlim(0, d["n"])
    ax0.set_xlabel("specimens admissible (of 757 fitted)")
    ax0.set_title("Measurements that pass QC", fontsize=11, pad=8)
    for s in ("top", "right", "left"): ax0.spines[s].set_visible(False)

    ax1 = fig.add_subplot(gs[1])
    yy = np.arange(len(aff))
    ax1.barh(yy, [100 * a[1] for a in aff], color=RED, height=0.6, edgecolor="white", lw=0.7)
    ax1.set_yticks(yy)
    ax1.set_yticklabels([f"{a[0]}  ({a[2]}/{a[3]})" for a in aff], fontsize=8.6)
    ax1.set_xlabel("head-length AP-inversion rate (%)")
    ax1.set_title(f"The HL failure is structured, not random\n"
                  f"{nzero} of {len(big)} genera (n≥5) have zero inversions",
                  fontsize=10.5, pad=8)
    ax1.text(0.98, 0.04, "permutation p = 0.025", transform=ax1.transAxes, ha="right",
             fontsize=9, style="italic", color="#444")
    for s in ("top", "right"): ax1.spines[s].set_visible(False)
    ax1.grid(axis="x", alpha=0.18, lw=0.5)
    p = f"{FIG}/fig07_m4a_qc.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", p)


def fig_m4b_genus():
    """TASK 9 -- validated head SHAPE differs systematically among genera."""
    d = json.load(open(f"{OUT}/m4a_admissible_set.json"))
    b = json.load(open(f"{OUT}/m4b_scalefree.json"))["ratios"]["HW/WL"]
    rows = [r for r in d["per_specimen"] if r["HW_admissible"]]
    by = collections.defaultdict(list)
    for r in rows: by[r["genus"]].append(r["HW_WL"])
    big = {k: v for k, v in by.items() if len(v) >= 5}
    order = sorted(big, key=lambda k: np.median(big[k]))
    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        v = np.array(big[k])
        ax.scatter(i + rng.uniform(-.16, .16, len(v)), v, s=11, color=BLUE, alpha=.45, lw=0)
        ax.plot([i - .3, i + .3], [np.median(v)] * 2, color="#0d2f52", lw=2.2, zorder=5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{k} ({len(big[k])})" for k in order], rotation=90, fontsize=7.4)
    ax.set_ylabel("HW / WL   (scale-free head shape)")
    ax.set_title("Validated head shape differs systematically among genera", fontsize=11.5, pad=9)
    ax.text(0.005, 0.965, f"rank $\\eta^2$ = {b['eta_sq_rank']:.3f}    permutation p < 5×10⁻⁵    "
                          f"n = {b['n_tested']} in {b['n_genera']} genera (n≥5)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", lw=0.8))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.16, lw=0.5)
    fig.tight_layout(); p = f"{FIG}/fig09_m4b_genus_structure.png"
    fig.savefig(p, dpi=300); fig.savefig(p.replace(".png", ".pdf")); print("wrote", p)


def fig_m4c_variance():
    """TASK 10 -- genus signal is not a singleton-species artefact."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), gridspec_kw={"width_ratios": [1.55, 1]})
    parts = [("genus", 23.7, BLUE), ("species\nin genus", 17.7, "#6b9ac4"),
             ("residual\n(within-species\n+ caste + measurement)", 58.6, "#dfe3e8")]
    left = 0
    ax = axes[0]
    for lbl, v, c in parts:
        ax.barh(0, v, left=left, color=c, height=0.5, edgecolor="white", lw=1.4)
        ax.text(left + v / 2, 0, f"{v:.1f}%", ha="center", va="center", fontsize=11,
                weight="bold", color="white" if c != "#dfe3e8" else "#333")
        ax.text(left + v / 2, -0.42, lbl, ha="center", va="top", fontsize=8.6)
        left += v
    ax.set_xlim(0, 100); ax.set_ylim(-1.15, 0.5); ax.axis("off")
    ax.set_title("Where HW/WL variation sits   (all admissible, n=753)", fontsize=10.5, pad=6)

    ax = axes[1]
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, [23.7, 17.7], w, color=[BLUE, "#6b9ac4"], label="all species")
    ax.bar(x + w/2, [22.5, 19.8], w, color=[BLUE, "#6b9ac4"], alpha=.55, hatch="//",
           edgecolor="white", label="replicated species only")
    for xi, (a, b_) in enumerate([(23.7, 22.5), (17.7, 19.8)]):
        ax.text(xi - w/2, a + .6, f"{a}", ha="center", fontsize=8.6)
        ax.text(xi + w/2, b_ + .6, f"{b_}", ha="center", fontsize=8.6)
    ax.set_xticks(x); ax.set_xticklabels(["genus", "species"])
    ax.set_ylabel("% of variance"); ax.set_ylim(0, 30)
    ax.set_title("Genus > species survives\nremoval of singleton species", fontsize=10.5, pad=6)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); p = f"{FIG}/fig10_m4c_variance.png"
    fig.savefig(p, dpi=300); fig.savefig(p.replace(".png", ".pdf")); print("wrote", p)


def fig_m4e_bridge():
    """TASK 11 -- the scalar marks a genus-structured direction in the dense phenotype."""
    e = json.load(open(f"{OUT}/m4e_bridge.json"))["traits"]["HW/WL"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    ax.text(.5, .62, f"$R^2$ = {e['dense_predicts_trait_R2']:.3f}", ha="center", fontsize=26,
            color=BLUE, weight="bold", transform=ax.transAxes)
    ax.text(.5, .40, "dense 3D shape predicts\nthe validated scalar", ha="center", fontsize=11,
            transform=ax.transAxes)
    ax.text(.5, .17, "the interpretable measurement is a component\nof the dense phenotype, "
                     "not separate from it", ha="center", fontsize=8.6, color="#555",
            transform=ax.transAxes, style="italic")
    ax.axis("off")
    ax.set_title("A   Dense shape → head-width ratio", fontsize=10.5, loc="left")

    ax = axes[1]
    vals = [0.409, 0.236]; errs = [0.044, 0.049]
    ax.bar([0, 1], vals, 0.5, yerr=errs, capsize=6, color=[BLUE, GREY],
           edgecolor="white", lw=1.2, error_kw=dict(lw=1.2, ecolor="#444"))
    for i, (v, er) in enumerate(zip(vals, errs)):
        ax.text(i, v + er + .015, f"{v:.3f} ± {er:.3f}", ha="center", fontsize=9.6, weight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["trait-aligned\ndirection", "random\ndirection"], fontsize=9.4)
    ax.set_ylabel("held-out genus $\\eta^2$")
    ax.set_ylim(0, 0.52)
    ax.set_title("B   Biological structure in dense shape", fontsize=10.5, loc="left")
    ax.text(0.5, 0.055, "cross-validated, 50 splits\nWilcoxon p = 1.8×10⁻¹⁵", transform=ax.transAxes,
            ha="center", fontsize=8.4, style="italic", color="#444")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.16, lw=0.5)
    fig.suptitle("Validated morphometrics recover a biologically structured component "
                 "of the dense 3D phenotype", fontsize=11, y=1.01)
    fig.tight_layout(); p = f"{FIG}/fig11_m4e_bridge.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    fig_trait_accuracy(); fig_accuracy_vs_repeatability(); fig_m4a_qc()
    fig_m4b_genus(); fig_m4c_variance(); fig_m4e_bridge()
