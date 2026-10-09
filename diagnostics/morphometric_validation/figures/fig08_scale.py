"""FIGURE 8 -- absolute scale is not identifiable from the current corpus.

Message: this is estimand discipline, not a failed analysis. The data do not identify the quantity
absolute allometry requires, so the claim is not made.
"""
import os, sys, json, glob, re, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.dirname(os.path.abspath(__file__))
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#9aa0a6"
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9})

# approximate published worker body lengths (mm). Deliberately coarse: they are used only to show
# a 7x SPREAD, never as specimen-level ground truth (see M4 pre-registration, not-allowed list).
KNOWN = {"Megaponera_analis": 12.0, "Tetramorium_polymorphum": 3.0, "Solenopsis_afrc": 2.5,
         "Cephalotes_pusillus": 4.0, "Dorylus_fulvus": 4.0, "Carebara_atoma": 1.0,
         "Atta_sexdens": 9.0, "Acromyrmex_octospinosus": 7.0, "Labidus_praedator": 6.0,
         "Hypoponera_sp.": 3.0, "Cephalotes_minutus": 4.0, "Eciton_burchellii": 8.0,
         "Cataulacus_sp.": 5.0, "Strumigenys_sp.": 2.5, "Paraponera_clavata": 22.0}


def diag(f):
    v = np.asarray([[float(x) for x in ln.split()[1:4]] for ln in open(f) if ln.startswith("v ")])
    return float(np.linalg.norm(v.max(0) - v.min(0)))


def main():
    vox = json.load(open("/hpcwork/nao48500/antscan_voxel.json"))
    raw, corr = collections.defaultdict(list), collections.defaultdict(list)
    for f in sorted(glob.glob("/rwthfs/rz/cluster/hpcwork/nao48500/worker_ALT/*.obj")):
        k = os.path.basename(f).replace("_processed.obj", "").replace(".obj", "")
        sp = re.match(r"([A-Za-z]+_[A-Za-z.]+)", k).group(1)
        if sp not in KNOWN or k not in vox:
            continue
        d = diag(f)
        raw[sp].append(d); corr[sp].append(d * vox[k]["vox"] / 1000.0)

    sp = sorted(raw, key=lambda s: KNOWN[s])
    kn = np.array([KNOWN[s] for s in sp])
    r = np.array([np.median(raw[s]) for s in sp])
    c = np.array([np.median(corr[s]) for s in sp])
    imp_r, imp_c = r / kn, c / kn

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    y = np.arange(len(sp))
    ax.scatter(imp_r / np.median(imp_r), y, s=64, color=RED, edgecolor="white", lw=1.1,
               zorder=4, label="raw mesh units")
    ax.scatter(imp_c / np.median(imp_c), y, s=64, color=BLUE, edgecolor="white", lw=1.1,
               zorder=4, marker="s", label="after voxel-size correction")
    ax.axvline(1.0, color="#333", lw=1.0, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([s.replace("_", " ") for s in sp], fontsize=8.4)
    ax.set_xscale("log")
    ax.set_xlabel("implied scale factor, relative to the corpus median\n"
                  "(a physically consistent unit would put every species on the dashed line)")
    ax.set_title("Implied physical scale disagrees by ~7× across species", fontsize=10.8, pad=8)
    ax.legend(fontsize=8.2, frameon=False, loc="lower right")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.grid(axis="x", alpha=0.18, lw=0.5, which="both")

    ax = axes[1]
    sr = imp_r.max() / imp_r.min(); sc = imp_c.max() / imp_c.min()
    rr = np.corrcoef(np.log(kn), np.log(r))[0, 1]; rc = np.corrcoef(np.log(kn), np.log(c))[0, 1]
    ax.bar([0, 1], [sr, sc], 0.5, color=[RED, BLUE], edgecolor="white", lw=1.2)
    for i, v in enumerate([sr, sc]):
        ax.text(i, v + 0.15, f"{v:.1f}×", ha="center", fontsize=12, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["raw", "voxel-corrected"], fontsize=9.4)
    ax.set_ylabel("spread of implied scale (max / min)")
    ax.set_ylim(0, max(sr, sc) * 1.32)
    ax.set_title("The obvious fix does not fix it", fontsize=10.8, pad=8)
    ax.text(0.5, 0.80, f"correlation with known body length\nraw r={rr:.2f}   corrected r={rc:.2f}\n"
                       f"(a physical unit would give r ≈ 0.95)",
            transform=ax.transAxes, ha="center", fontsize=8.4, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", lw=0.8))
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

    fig.text(0.5, -0.055,
             "Absolute scale is not identifiable from the current corpus.  →  absolute allometry "
             "is excluded, not attempted.",
             ha="center", fontsize=11.5, weight="bold", color="#0d2f52")
    fig.text(0.5, -0.105,
             "247 of 279 metadata-matched specimens share one voxel size, so voxel size cannot "
             "produce a 7× spread; the preprocessing script applies no rescaling.",
             ha="center", fontsize=8.4, color="#666", style="italic")
    fig.tight_layout()
    p = f"{FIG}/fig08_scale_identifiability.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"n_species={len(sp)}  raw spread={sr:.2f}x  corrected={sc:.2f}x  r_raw={rr:.3f} r_corr={rc:.3f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
