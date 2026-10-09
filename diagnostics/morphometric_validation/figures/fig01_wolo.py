"""FIGURE 1 (Fabian / WOLO) -- automatically recovered head width vs body mass, log-log.

Message: the head width recovered automatically from the fitted 3D model follows the expected
biological scaling relationship with body mass. Reference (hand-measured) head width is plotted
alongside as the control, so the figure validates RECOVERY rather than re-showing a known law.
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [REPO, f"{REPO}/diagnostics/groundtruth", f"{REPO}/diagnostics/morphometrics",
                f"{REPO}/diagnostics/morphometric_validation"]
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
import trait_extract as TX
from measure import load_model
from mv_framework import forward_pair, recal_landmarks, TEMPL_TO_RECAL, ATTA_NPZ, ATTA_REF

FIG = os.path.dirname(os.path.abspath(__file__))
PUBLISHED_HW_MASS = 0.3949


def fit_ll(x, y):
    s, ic, r, p, se = stats.linregress(np.log10(x), np.log10(y))
    tc = stats.t.ppf(0.975, len(x) - 2)
    return dict(slope=s, ic=ic, lo=s - tc * se, hi=s + tc * se, r2=r ** 2, p=p, se=se)


def band(ax, x, f, color, n):
    """95% CI band for the fitted line."""
    lx = np.log10(np.linspace(x.min() * 0.9, x.max() * 1.1, 100))
    mlx = np.log10(x).mean()
    sxx = ((np.log10(x) - mlx) ** 2).sum()
    resid_se = f["se"] * np.sqrt(sxx)
    se_line = resid_se * np.sqrt(1 / n + (lx - mlx) ** 2 / sxx)
    tc = stats.t.ppf(0.975, n - 2)
    ly = f["ic"] + f["slope"] * lx
    ax.fill_between(10 ** lx, 10 ** (ly - tc * se_line), 10 ** (ly + tc * se_line),
                    color=color, alpha=0.15, lw=0, zorder=1)
    ax.plot(10 ** lx, 10 ** ly, color=color, lw=2, zorder=3)


def main():
    M = load_model()
    lm = dict(TX.load_landmarks(M)); raw = recal_landmarks(M)
    for t, r in TEMPL_TO_RECAL.items():
        if r in raw: lm[t] = raw[r]
    obs, _c, labels = forward_pair(ATTA_NPZ)
    J = np.einsum("ij,njk->nik", M["Jr"], obs)
    jn = M["jnames"]
    bl_model = np.linalg.norm(J[:, jn.index("b_t")] - J[:, jn.index("b_a_5")], axis=-1)
    tr = TX.traits(obs, M=M, lm=lm)

    ref = {}
    with open(ATTA_REF) as fh:
        for row in csv.DictReader(fh):
            ref[int(row["Shape"])] = (float(row["b_t to b_a_5 [mm]"]),
                                      float(row["head width [mm]"]), float(row["mass [mg]"]))
    idx, bl, hwref, mass = [], [], [], []
    for i, lab in enumerate(labels):
        try: s = int(str(lab).split("_")[0])
        except ValueError: continue
        if s in ref:
            idx.append(i); bl.append(ref[s][0]); hwref.append(ref[s][1]); mass.append(ref[s][2])
    idx, bl, hwref, mass = map(np.asarray, (idx, bl, hwref, mass))
    scale = bl / bl_model[idx]                       # mm per model unit, per specimen
    hw_model = np.asarray(tr["HW"], float)[idx] * scale
    n = len(idx)

    fm, fr = fit_ll(mass, hw_model), fit_ll(mass, hwref)

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    band(ax, mass, fr, "#9aa0a6", n)
    band(ax, mass, fm, "#1f5fa8", n)
    ax.scatter(mass, hwref, s=26, facecolor="none", edgecolor="#9aa0a6", lw=1.1, zorder=4,
               label=f"reference (hand-measured)   slope {fr['slope']:.3f}")
    ax.scatter(mass, hw_model, s=34, color="#1f5fa8", edgecolor="white", lw=0.6, zorder=5,
               label=f"SMILify (auto-recovered)   slope {fm['slope']:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("body mass (mg)")
    ax.set_ylabel("head width (mm)")
    ax.set_title("Automatically recovered head width scales with body mass", fontsize=11, pad=9)
    txt = (f"SMILify   slope {fm['slope']:.3f}  [{fm['lo']:.3f}, {fm['hi']:.3f}]"
           f"   $R^2$ = {fm['r2']:.3f}\n"
           f"reference slope {fr['slope']:.3f}  [{fr['lo']:.3f}, {fr['hi']:.3f}]"
           f"   $R^2$ = {fr['r2']:.3f}\n"
           f"published  {PUBLISHED_HW_MASS:.3f}          n = {n}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.2,
            family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc", lw=0.8))
    ax.legend(loc="lower right", fontsize=8.2, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.18, which="both", lw=0.5)
    fig.tight_layout()
    p = os.path.join(FIG, "fig01_wolo_headwidth_mass.png")
    fig.savefig(p, dpi=300); fig.savefig(p.replace(".png", ".pdf"))
    print(f"n={n}")
    print(f"SMILify  slope={fm['slope']:.4f} [{fm['lo']:.4f},{fm['hi']:.4f}] R2={fm['r2']:.4f} p={fm['p']:.3g}")
    print(f"referenceslope={fr['slope']:.4f} [{fr['lo']:.4f},{fr['hi']:.4f}] R2={fr['r2']:.4f}")
    print(f"published={PUBLISHED_HW_MASS}")
    resid = np.log10(hw_model) - (fm["ic"] + fm["slope"] * np.log10(mass))
    print(f"residual sd (log10) = {resid.std(ddof=2):.4f}  -> {100*(10**resid.std(ddof=2)-1):.1f}% typical deviation")
    print("wrote", p)


if __name__ == "__main__":
    main()
