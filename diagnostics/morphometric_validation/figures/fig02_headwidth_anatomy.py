"""FIGURE 2 -- what head width actually IS, and where it lands on real specimens.

Answers Fabian's question directly: "are you measuring at the correct, widest point on the head?"

HONEST NOTE, and the reason this figure is not a picture of added joints: the production head width
is TRAITS["HW"] = ("ext", ("b_h", 0)) -- the MAXIMUM transverse extent of the head part after rigid
alignment to the template's head frame. The `b_h_l`/`b_h_r` reporter bones from the Atta
head-width replication are NOT joints in the production model (which has only `b_h`) and are not
what M2/M3/M4 or the WOLO figure measure. This figure therefore shows the real measurement locus:
the two head vertices that realise the maximum.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [REPO, f"{REPO}/diagnostics/groundtruth", f"{REPO}/diagnostics/morphometrics",
                f"{REPO}/diagnostics/morphometric_validation"]
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
from measure import load_model, kabsch, body_frame
from mv_framework import forward_pair
import glob

FIG = os.path.dirname(os.path.abspath(__file__))
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#9aa0a6"


def head_extent_points(verts, M):
    """Return, per specimen, the two head vertices realising the max transverse extent, in the
    template's head frame -- i.e. exactly the pair whose separation IS the reported HW."""
    V0, jn = M["v_template"], M["jnames"]
    idx = np.where(M["dominant"] == jn.index("b_h"))[0]
    ax = body_frame(M)[0]
    T = V0[idx] - V0[idx].mean(0)
    out = []
    for i in range(len(verts)):
        P = verts[i][idx]
        Pc = P - P.mean(0)
        R = kabsch(Pc, T)
        Q = Pc @ R
        proj = Q @ ax
        a, b = int(proj.argmax()), int(proj.argmin())
        out.append((idx[a], idx[b], Q, proj, R, P.mean(0), float(proj[a] - proj[b])))
    return idx, out


def draw_head(ax, Q, proj, ia_local, ib_local, title, bf, color=BLUE, show_axis=True):
    """Frontal view: horizontal = the LATERAL body axis, vertical = dorsoventral.
    These are body_frame rows, not raw coordinate axes -- projecting on raw axes puts both
    extent points on top of each other, which is wrong."""
    x, y = Q @ bf[0], Q @ bf[2]
    ax.scatter(x, y, s=1.1, color="#c9d4e0", lw=0, zorder=1)
    ax.plot([x[ia_local], x[ib_local]], [y[ia_local], y[ib_local]], color=color, lw=1.8, zorder=3)
    ax.scatter([x[ia_local], x[ib_local]], [y[ia_local], y[ib_local]], s=62, color=color,
               edgecolor="white", lw=1.2, zorder=5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=8.8, pad=3)


def main():
    M = load_model()
    idx, _ = head_extent_points(M["v_template"][None], M)

    # --- real specimens
    f0 = sorted(glob.glob(f"{REPO}/diagnostics/moonshot/runs/M4_W*/Stage_3_deform_fine.npz"))[0]
    obs, _c, labels = forward_pair(f0)
    d = json.load(open(f"{os.path.dirname(FIG)}/out/m4a_admissible_set.json"))
    adm = {r["specimen"]: r for r in d["per_specimen"]}
    # pick 4 admissible specimens spanning the HW/WL range present in this chunk
    cands = [(i, l) for i, l in enumerate(labels) if l in adm and adm[l]["HW_admissible"]]
    cands.sort(key=lambda t: adm[t[1]]["HW_WL"])
    pick = [cands[1], cands[len(cands)//3], cands[2*len(cands)//3], cands[-2]]
    sel = np.array([p[0] for p in pick])
    _, info = head_extent_points(obs[sel], M)

    fig = plt.figure(figsize=(12.2, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.52, wspace=0.18)

    # ---------- Panel A: the template head, and the measured axis
    axA = fig.add_subplot(gs[0, 0:2])
    _, tinfo = head_extent_points(M["v_template"][None], M)
    ia, ib, Q, proj, *_ = tinfo[0]
    la, lb = int(proj.argmax()), int(proj.argmin())
    # whole body in grey, head in colour
    V0 = M["v_template"]
    bf = body_frame(M)
    # DORSAL view: horizontal = antero-posterior, vertical = lateral, so head WIDTH is visible.
    AX, LA = V0 @ bf[1], V0 @ bf[0]
    axA.scatter(AX[::5], LA[::5], s=1.0, color="#e4e8ec", lw=0)
    axA.scatter(AX[idx], LA[idx], s=1.8, color="#c9d4e0", lw=0)
    pax, pal = AX[[ia, ib]], LA[[ia, ib]]
    axA.plot(pax, pal, color=BLUE, lw=2.0, zorder=4)
    axA.scatter(pax, pal, s=70, color=BLUE, edgecolor="white", lw=1.2, zorder=5)
    axA.annotate("head part\n(`b_h` skinning group)", xy=(AX[idx].mean(), LA[idx].max()),
                 xytext=(-40, 40), textcoords="offset points", fontsize=8.4, color="#444",
                 arrowprops=dict(arrowstyle="->", color="#888", lw=0.9))
    axA.annotate("maximum transverse extent\n= reported head width",
                 xy=(pax.mean(), pal.mean()),
                 xytext=(-150, -58), textcoords="offset points", fontsize=8.6, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    axA.text(0.99, 0.02, "dorsal view", transform=axA.transAxes, ha="right", fontsize=8,
             color="#888", style="italic")
    axA.set_aspect("equal"); axA.axis("off")
    axA.set_title("A   The anatomical definition encoded in the model", fontsize=10.5, loc="left")

    # ---------- Panel B: schematic
    axB = fig.add_subplot(gs[0, 2:4]); axB.axis("off")
    axB.set_xlim(0, 10); axB.set_ylim(0, 6)
    th = np.linspace(0, 2*np.pi, 200)
    axB.plot(5 + 2.5*np.cos(th), 3.4 + 1.5*np.sin(th), color="#666", lw=1.6)
    axB.text(5, 3.4, "HEAD", ha="center", va="center", fontsize=11, color="#666", weight="bold")
    axB.plot([2.5, 7.5], [3.4, 3.4], color=BLUE, lw=2.0)
    for xx in (2.5, 7.5):
        axB.scatter([xx], [3.4], s=110, color=BLUE, edgecolor="white", lw=1.4, zorder=5)
    axB.annotate("", xy=(2.5, 1.75), xytext=(7.5, 1.75),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.4))
    axB.text(5, 1.35, "head width", ha="center", fontsize=9.6, color=BLUE, weight="bold")
    axB.text(5, 5.35, "extreme points along the head's own lateral axis,\n"
                      "after rigid alignment of the head to the template frame",
             ha="center", fontsize=8.4, color="#444")
    axB.text(5, 0.45, "not a fixed vertex pair — recomputed per specimen,\n"
                      "so it tracks the true widest point as head shape changes",
             ha="center", fontsize=8.0, color="#777", style="italic")
    axB.set_title("B   How the measurement is defined", fontsize=10.5, loc="left")

    # ---------- Panel C: real registered specimens
    for k, (i_sel, (ia, ib, Q, proj, R, cen, ext)) in enumerate(zip(sel, info)):
        axC = fig.add_subplot(gs[1, k])
        lab = labels[i_sel]
        draw_head(axC, Q, proj, int(proj.argmax()), int(proj.argmin()),
                  f"{lab.split('_CASENT')[0].replace('_',' ')}\nHW/WL = {adm[lab]['HW_WL']:.2f}",
                  body_frame(M))
    fig.text(0.5, 0.478, "C   Where the measurement lands on registered real specimens "
                         "(frontal view, head frame)",
             fontsize=10.5, ha="center")
    fig.suptitle("Head width is measured at the widest transverse point of the head, "
                 "recomputed per specimen", fontsize=12, y=0.985)
    p = f"{FIG}/fig02_headwidth_anatomy.png"
    fig.savefig(p, dpi=250, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", p)
    for lab, (_, _, _, _, _, _, ext) in zip([labels[i] for i in sel], info):
        print(f"   {lab:44s} extent={ext:.4f}")


if __name__ == "__main__":
    main()
