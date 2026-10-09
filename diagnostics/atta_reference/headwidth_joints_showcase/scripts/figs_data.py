"""Data figures: F01 (Fabian's log-log), F08 reproduction, F09 agreement, F10 HW~BL, F11 pose,
F12 bilateral symmetry. All on the rematch fit whose regressed joints reproduce the Blender export."""
import json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
from scipy import stats
from hw_core import *

BLUE, GREY, DARK, RED = "#1f5fa8", "#9aa0a6", "#0d2f52", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9, "font.family": "DejaVu Sans"})
FIG = f"{SHOW}/figures"


def ols(x, y):
    lx, ly = np.log10(x), np.log10(y)
    r = stats.linregress(lx, ly); n = len(x); tc = stats.t.ppf(.975, n - 2)
    return dict(slope=r.slope, ic=r.intercept, se=r.stderr, lo=r.slope - tc * r.stderr,
                hi=r.slope + tc * r.stderr, r2=r.rvalue ** 2, p=r.pvalue, n=n,
                resid_sd=np.std(ly - (r.intercept + r.slope * lx), ddof=2), lx=lx, ly=ly)


def band(ax, f, xmin, xmax, color, alpha=.16, lw=2.2, label=None):
    gx = np.linspace(np.log10(xmin), np.log10(xmax), 200)
    mlx = f["lx"].mean(); sxx = ((f["lx"] - mlx) ** 2).sum(); n = f["n"]
    s = f["resid_sd"] * np.sqrt(1 / n + (gx - mlx) ** 2 / sxx); tc = stats.t.ppf(.975, n - 2)
    gy = f["ic"] + f["slope"] * gx
    ax.fill_between(10 ** gx, 10 ** (gy - tc * s), 10 ** (gy + tc * s), color=color, alpha=alpha, lw=0)
    ax.plot(10 ** gx, 10 ** gy, color=color, lw=lw, label=label, zorder=3)


def logaxes(ax, xt, yt):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xt)); ax.yaxis.set_major_locator(FixedLocator(yt))
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=.2, lw=.5, which="major")


def save(fig, name):
    p = f"{FIG}/{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def main():
    d = load_all()
    lab = d["labels"]; mass = d["mass"]; BL = d["BL"]
    bh = blender_hw(EXPORT); bs = blender_hw(EXPORT_SUBMITTED)
    HW = np.array([bh[l] for l in lab])                  # the exported measurement itself
    HWsub = np.array([bs[l] for l in lab]); HWref = d["HW_ref"]
    summ = {}

    # ---------------- F01 -- Fabian's figure: head width over body mass, clean
    f = ols(mass, HW); fr = ols(mass, HWref); fs = ols(mass, HWsub)
    summ["HW_mass"] = {k: float(v) for k, v in f.items() if k not in ("lx", "ly")}
    summ["HW_mass_reference"] = {k: float(v) for k, v in fr.items() if k not in ("lx", "ly")}
    summ["HW_mass_submitted_export"] = {k: float(v) for k, v in fs.items() if k not in ("lx", "ly")}
    XMIN, XMAX, YMIN, YMAX = 0.8, 60, 0.9, 5.2
    for variant in ("clean", "with_reference"):
        fig, ax = plt.subplots(figsize=(5.6, 4.7))
        # isometry reference (slope 1/3) through the data centroid, as in the Atta 3D-shape literature
        cx, cy = f["lx"].mean(), f["ly"].mean(); gx = np.log10([XMIN, XMAX])
        ax.plot(10 ** gx, 10 ** (cy + (gx - cx) / 3), ls="--", color=GREY, lw=1.1, zorder=1,
                label="isometry (slope 1/3)")
        if variant == "with_reference":
            ax.scatter(mass, HWref, s=30, facecolor="none", edgecolor=GREY, lw=1.1, zorder=4,
                       label=f"2025 reference   slope {fr['slope']:.3f}")
        band(ax, f, XMIN * 1.15, XMAX * .9, BLUE, label="OLS fit ± 95% CI")
        ax.scatter(mass, HW, s=38, color=BLUE, edgecolor="white", lw=.7, zorder=5,
                   label="SMILify replication (b_h_l–b_h_r)")
        logaxes(ax, [1, 2, 5, 10, 20, 50], [1, 1.5, 2, 3, 4, 5])
        ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
        ax.set_xlabel("body mass (mg)"); ax.set_ylabel("head width (mm)")
        ax.set_title("Head width scales with body mass — Atta vollenweideri", fontsize=11, pad=8)
        txt = (f"slope = {f['slope']:.3f}  (95% CI {f['lo']:.3f}–{f['hi']:.3f})\n"
               f"R² = {f['r2']:.3f}     n = {f['n']}")
        if variant == "with_reference":
            txt += f"\nreference slope = {fr['slope']:.3f}  ({fr['lo']:.3f}–{fr['hi']:.3f})"
        ax.text(.04, .96, txt, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=.45", fc="white", ec="#cfcfcf", lw=.8))
        ax.legend(loc="lower right", fontsize=7.8, frameon=False)
        save(fig, f"F01{'a' if variant=='clean' else 'b'}_headwidth_vs_mass_loglog_{variant}")

    # ---------------- F08 -- the visualised joints ARE the exported joints
    py = d["HW_mm"]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lim = [HW.min() * .9, HW.max() * 1.06]
    ax.plot(lim, lim, color=GREY, lw=1, ls="--")
    ax.scatter(HW, py, s=36, color=BLUE, edgecolor="white", lw=.7, zorder=3)
    rel = 100 * np.abs(py / HW - 1)
    ax.set_xlabel("head width exported by Blender addon (mm)")
    ax.set_ylabel("head width re-computed in Python (mm)")
    ax.set_title("Regressed joints reproduced exactly", fontsize=10.5, pad=8)
    ax.text(.04, .96, f"mean |difference| {rel.mean():.3f}%\nmax {rel.max():.3f}%   n = 20",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=.4", fc="white", ec="#cfcfcf", lw=.8))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(alpha=.2, lw=.5)
    save(fig, "F08_python_vs_blender_export")
    summ["python_vs_blender"] = dict(mean_abs_pct=float(rel.mean()), max_abs_pct=float(rel.max()))

    # ---------------- F09 -- agreement with the 2025 reference: identity + Bland–Altman
    def ccc(a, b):
        return 2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    lim = [0.9, 4.7]
    a1.plot(lim, lim, color=GREY, lw=1, ls="--", label="identity")
    a1.scatter(HWref, HW, s=36, color=BLUE, edgecolor="white", lw=.7, zorder=3)
    for i, l in enumerate(lab):
        a1.annotate(l, (HWref[i], HW[i]), xytext=(3, -8), textcoords="offset points", fontsize=6.3, color="#777")
    a1.set_xlim(lim); a1.set_ylim(lim); a1.set_aspect("equal")
    a1.set_xlabel("2025 reference head width (mm)"); a1.set_ylabel("SMILify replication (mm)")
    a1.set_title("A   Agreement", fontsize=10.5, loc="left")
    c = ccc(HWref, HW)
    a1.text(.04, .96, f"Lin's CCC = {c:.3f}\nmedian |error| {np.median(100*np.abs(HW/HWref-1)):.2f}%",
            transform=a1.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=.4", fc="white", ec="#cfcfcf", lw=.8))
    # Bland–Altman on log scale -> differences are % and do not grow with size
    mean_ = (HW + HWref) / 2; diff = 100 * (HW - HWref) / mean_
    bias = diff.mean(); sd = diff.std(ddof=1)
    a2.scatter(mean_, diff, s=36, color=BLUE, edgecolor="white", lw=.7, zorder=3)
    for yv, ls_, t in ((bias, "-", f"bias {bias:+.2f}%"), (bias + 1.96 * sd, "--", f"+1.96 SD {bias+1.96*sd:+.2f}%"),
                       (bias - 1.96 * sd, "--", f"−1.96 SD {bias-1.96*sd:+.2f}%")):
        a2.axhline(yv, color=DARK if ls_ == "-" else GREY, ls=ls_, lw=1.1)
        a2.text(4.72, yv, " " + t, va="center", fontsize=8, color="#444")
    a2.axhline(0, color="#dddddd", lw=.8, zorder=0)
    slope_ba = stats.linregress(mean_, diff)
    a2.set_xlim(0.9, 4.7)
    a2.set_xlabel("mean of the two measurements (mm)"); a2.set_ylabel("difference, replication − reference (%)")
    a2.set_title("B   Bland–Altman", fontsize=10.5, loc="left")
    a2.text(.03, .97, f"proportional bias: slope {slope_ba.slope:+.2f} %/mm, p = {slope_ba.pvalue:.2f}",
            transform=a2.transAxes, fontsize=8.4, color="#555", style="italic", va="top")
    for a in (a1, a2):
        for s in ("top", "right"): a.spines[s].set_visible(False)
        a.grid(alpha=.2, lw=.5)
    fig.tight_layout(); save(fig, "F09_agreement_with_2025_reference")
    summ["agreement"] = dict(ccc=float(c), bias_pct=float(bias), loa_lo=float(bias - 1.96 * sd),
                             loa_hi=float(bias + 1.96 * sd), prop_bias_p=float(slope_ba.pvalue))

    # ---------------- F10 -- head width over body length (the allometry the replication targets)
    g = ols(BL, HW); gr = ols(BL, HWref)
    summ["HW_BL"] = {k: float(v) for k, v in g.items() if k not in ("lx", "ly")}
    summ["HW_BL_reference"] = {k: float(v) for k, v in gr.items() if k not in ("lx", "ly")}
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    cx, cy = g["lx"].mean(), g["ly"].mean(); gx = np.log10([2.3, 10])
    ax.plot(10 ** gx, 10 ** (cy + (gx - cx)), ls="--", color=GREY, lw=1.1, label="isometry (slope 1)")
    ax.scatter(BL, HWref, s=30, facecolor="none", edgecolor=GREY, lw=1.1, zorder=4,
               label=f"2025 reference   slope {gr['slope']:.3f}")
    band(ax, g, 2.55, 9.2, BLUE, label="OLS fit ± 95% CI")
    ax.scatter(BL, HW, s=38, color=BLUE, edgecolor="white", lw=.7, zorder=5, label="SMILify replication")
    logaxes(ax, [2, 3, 4, 5, 7, 10], [1, 1.5, 2, 3, 4, 5])
    ax.set_xlim(2.3, 10); ax.set_ylim(.9, 5.2)
    ax.set_xlabel("body length, b_t–b_a_5 (mm)"); ax.set_ylabel("head width (mm)")
    ax.set_title("Positive allometry of head width on body length", fontsize=11, pad=8)
    ax.text(.04, .96, f"slope = {g['slope']:.3f}  (95% CI {g['lo']:.3f}–{g['hi']:.3f})\n"
                      f"reference = {gr['slope']:.3f}  ({gr['lo']:.3f}–{gr['hi']:.3f})\nR² = {g['r2']:.3f}   n = 20",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=.45", fc="white", ec="#cfcfcf", lw=.8))
    ax.legend(loc="lower right", fontsize=7.8, frameon=False)
    save(fig, "F10_headwidth_vs_bodylength_loglog")

    # ---------------- F11 -- pose invariance of the reporter-bone width
    R = d["R"]; can = d["can"]
    J_c = np.einsum("ij,njk->nik", d["M"]["Jr"], can); jn = d["M"]["jnames"]
    hw_obs = np.linalg.norm(d["PL"] - d["PR"], axis=1)
    hw_can = np.linalg.norm(np.einsum("v,nvc->nc", R[0], can) - np.einsum("v,nvc->nc", R[1], can), axis=1)
    bl_obs = np.linalg.norm(d["J"][:, jn.index("b_t")] - d["J"][:, jn.index("b_a_5")], axis=1)
    bl_can = np.linalg.norm(J_c[:, jn.index("b_t")] - J_c[:, jn.index("b_a_5")], axis=1)
    dhw = 100 * (hw_obs / hw_can - 1); dbl = 100 * (bl_obs / bl_can - 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    xs = np.arange(20)
    ax.bar(xs - .2, np.abs(dhw), .4, color=BLUE, label=f"head width   mean {np.abs(dhw).mean():.2f}%")
    ax.bar(xs + .2, np.abs(dbl), .4, color=GREY, label=f"body length (b_t–b_a_5)   mean {np.abs(dbl).mean():.2f}%")
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7.6)
    ax.set_xlabel("specimen (ordered as in the reference CSV)")
    ax.set_ylabel("|change| when pose is removed (%)")
    ax.set_title("Head width is insensitive to articulation", fontsize=10.5, pad=8)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=.2, lw=.5)
    save(fig, "F11_pose_invariance")
    summ["pose"] = dict(hw_mean_abs_pct=float(np.abs(dhw).mean()), hw_max_abs_pct=float(np.abs(dhw).max()),
                        bl_mean_abs_pct=float(np.abs(dbl).mean()), bl_max_abs_pct=float(np.abs(dbl).max()))

    # ---------------- F12 -- bilateral symmetry of the two regressed joints about the head midplane
    M = d["M"]; asym = []
    for i in range(20):
        Rk, cc, idx, bf = head_frame(d["obs"][i], M)
        pl = to_head_coords(d["PL"][i], Rk, cc, bf)[0]; pr = to_head_coords(d["PR"][i], Rk, cc, bf)[0]
        H = to_head_coords(d["obs"][i][idx], Rk, cc, bf)
        mid = (np.percentile(H[:, 0], 99.5) + np.percentile(H[:, 0], .5)) / 2
        asym.append((abs(pl[0] - mid), abs(pr[0] - mid)))
    asym = np.array(asym)
    ai = 100 * (asym[:, 0] - asym[:, 1]) / asym.mean(1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0, color=GREY, lw=1)
    ax.bar(np.arange(20), ai, color=[BLUE if abs(v) < 5 else RED for v in ai], width=.62)
    ax.set_xticks(np.arange(20)); ax.set_xticklabels(lab, fontsize=7.6)
    ax.set_ylabel("left − right distance to midplane (%)")
    ax.set_title("The two joints sit symmetrically about the head midplane", fontsize=10.5, pad=8)
    ax.text(.01, .95, f"mean |asymmetry| {np.abs(ai).mean():.2f}%   max {np.abs(ai).max():.2f}%",
            transform=ax.transAxes, va="top", fontsize=8.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=.2, lw=.5)
    save(fig, "F12_bilateral_symmetry")
    summ["symmetry"] = dict(mean_abs_pct=float(np.abs(ai).mean()), max_abs_pct=float(np.abs(ai).max()))

    # ---------------- F15 -- coherence with the scaling law: replication vs 2025 reference
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sets = [("2025 reference", HWref, GREY), ("export as first submitted", HWsub, "#8fa9c7"),
            ("SMILify replication\n(reproducible)", HW, BLUE)]
    for k, (nm, y, col) in enumerate(sets):
        ff = ols(mass, y); res = 100 * (10 ** (ff["ly"] - (ff["ic"] + ff["slope"] * ff["lx"])) - 1)
        jit = np.random.default_rng(k).uniform(-.12, .12, len(res))
        ax.scatter(np.full(len(res), k) + jit, res, s=30, color=col, edgecolor="white", lw=.6, zorder=3)
        sd_ = np.std(res, ddof=2)
        ax.plot([k - .3, k + .3], [sd_, sd_], color=DARK, lw=1); ax.plot([k - .3, k + .3], [-sd_, -sd_], color=DARK, lw=1)
        ax.text(k, 15.6, f"SD {sd_:.1f}%\nR² {ff['r2']:.3f}\nslope {ff['slope']:.3f}", ha="center", fontsize=8.4, va="top")
    ax.axhline(0, color="#cccccc", lw=1, zorder=0)
    ax.set_xticks(range(3)); ax.set_xticklabels([s_[0] for s_ in sets], fontsize=8.8)
    ax.set_ylabel("deviation from the fitted scaling line (%)"); ax.set_ylim(-9, 16.5)
    ax.set_title("The replication follows the head width–mass scaling law more tightly", fontsize=10.5, pad=8)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.grid(axis="y", alpha=.2, lw=.5)
    save(fig, "F15_coherence_with_scaling_law")

    # per-specimen table
    wp = {w["label"]: w for w in json.load(open(f"{SHOW}/data/widest_point_check.json"))}
    with open(f"{SHOW}/data/head_width_per_specimen.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["specimen", "mass_mg", "body_length_mm", "HW_reference_2025_mm", "HW_replication_mm",
                     "HW_python_mm", "error_vs_reference_pct", "widest_point_ratio_scan",
                     "joint_to_scan_surface_pct_diag", "pose_change_pct", "LR_asymmetry_pct"])
        for i, l in enumerate(lab):
            wr.writerow([l, mass[i], BL[i], HWref[i], round(HW[i], 4), round(py[i], 4),
                         round(100 * (HW[i] / HWref[i] - 1), 2), round(wp[l]["ratio_rep_to_widest_scan"], 4),
                         round(wp[l]["rep_to_scan_surface_pct_diag"], 3), round(dhw[i], 3), round(ai[i], 2)])
    json.dump(summ, open(f"{SHOW}/data/summary_statistics.json", "w"), indent=1)
    print(f"F01 slope {f['slope']:.4f} [{f['lo']:.4f},{f['hi']:.4f}] R2 {f['r2']:.4f} | ref {fr['slope']:.4f} | submitted-export {fs['slope']:.4f}")
    print(f"F10 slope {g['slope']:.4f} [{g['lo']:.4f},{g['hi']:.4f}] | ref {gr['slope']:.4f}")
    print(f"CCC {c:.4f} bias {bias:+.2f}% LoA [{bias-1.96*sd:+.2f},{bias+1.96*sd:+.2f}] propbias p={slope_ba.pvalue:.3f}")
    print(f"pose HW {np.abs(dhw).mean():.3f}% (max {np.abs(dhw).max():.3f}) vs BL {np.abs(dbl).mean():.3f}%")
    print(f"asym mean {np.abs(ai).mean():.2f}% max {np.abs(ai).max():.2f}%")


if __name__ == "__main__":
    main()
