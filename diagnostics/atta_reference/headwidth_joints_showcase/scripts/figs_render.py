"""3D render figures: F03 template screenshots, F04 regressor weights, F05 all 20 registered heads,
F06 overlay on the real scans, F07 widest-point evidence, F13 placement consistency."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree
from hw_core import *
from hw_render import *

BLUE, GREY, DARK, RED, ORANGE = "#1f5fa8", "#9aa0a6", "#0d2f52", "#c0392b", "#e67e22"
CL, CR = "#d35400", "#1f5fa8"          # b_h_l, b_h_r
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
FIG = f"{SHOW}/figures"


def save(fig, name, dpi=250):
    p = f"{FIG}/{name}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def joints_on(ax, pts, axes, view, labels=None, colors=None, s=70, line=True, lw=2.0, fs=8.5, offs=None):
    Q = project(np.asarray(pts), axes, view)
    if line and len(Q) >= 2:
        ax.plot(Q[:2, 0], Q[:2, 1], color=DARK, lw=lw, zorder=20, solid_capstyle="round")
    for k, q in enumerate(Q):
        ax.scatter(q[0], q[1], s=s, color=(colors or [BLUE] * len(Q))[k], edgecolor="white",
                   lw=1.2, zorder=21)
        if labels:
            dx, dy = (offs or [(8, 8)] * len(Q))[k]
            ax.annotate(labels[k], (q[0], q[1]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=fs, color=(colors or [BLUE] * len(Q))[k], weight="bold", zorder=22)
    return Q


def head_aligned(V, M):
    Rk, c, idx, _ = head_frame(V, M)
    return (V - c) @ Rk, Rk, c, idx


def main():
    d = load_all(); M = d["M"]; F = d["faces"]; V0 = M["v_template"]; R = d["R"]
    axes = orient_frame(M); jn = M["jnames"]; J0 = M["Jr"] @ V0
    idx_head = np.where(M["dominant"] == jn.index("b_h"))[0]
    fh = faces_of(F, idx_head)
    PL0, PR0 = R[0] @ V0, R[1] @ V0

    # ================= F03 template screenshots
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.08], hspace=.12, wspace=.04)
    for k, v in enumerate(["dorsal", "lateral", "anterior"]):
        ax = fig.add_subplot(gs[0, k]); draw_mesh(ax, V0, F, axes, v)
        joints_on(ax, [PL0, PR0], axes, v, colors=[CL, CR], s=46, lw=1.4)
        ax.set_title(f"{v} view", fontsize=10.5)
    # head close-ups with context joints
    ctx = ["b_h", "ma_l", "ma_r"]
    for k, v in enumerate(["anterior", "dorsal", "lateral"]):
        ax = fig.add_subplot(gs[1, k])
        draw_mesh(ax, V0, F, axes, v, face_mask=fh)
        Qc = project(J0[[jn.index(n) for n in ctx]], axes, v)
        ax.scatter(Qc[:, 0], Qc[:, 1], s=34, color=GREY, edgecolor="white", lw=.8, zorder=19)
        for n, q in zip(ctx, Qc):
            ax.annotate(n, q[:2], xytext=(5, -11), textcoords="offset points", fontsize=7.8, color="#666")
        offs = {"anterior": [(10, 6), (-44, 6)], "dorsal": [(6, -16), (-10, -16)], "lateral": [(8, 8), (8, -14)]}[v]
        joints_on(ax, [PL0, PR0], axes, v, labels=["b_h_l", "b_h_r"], colors=[CL, CR], offs=offs)
        ax.set_title(f"head close-up — {v}{' (full-face)' if v=='anterior' else ''}", fontsize=10.5)
    fig.suptitle("OmniAnt template with the two added head-width joints (b_h_l, b_h_r)", fontsize=13, y=.995)
    save(fig, "F03_template_with_headwidth_joints")

    # ================= F04 the regressor: 10 nearest vertices, inverse-distance weights (zoomed)
    fig = plt.figure(figsize=(15.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=.14)
    ax = fig.add_subplot(gs[0])
    draw_mesh(ax, V0, F, axes, "anterior", face_mask=fh)
    joints_on(ax, [PL0, PR0], axes, "anterior", colors=[CL, CR], labels=["b_h_l", "b_h_r"],
              offs=[(-6, 12), (-30, 12)])
    ax.set_title("A   the two joints on the template head (full-face)", fontsize=10.5, loc="left")
    lat_ax = axes[0]
    for k, (row, col, nm) in enumerate(((0, CL, "b_h_l"), (1, CR, "b_h_r"))):
        ax = fig.add_subplot(gs[k + 1])
        jp = (PL0, PR0)[row]
        view = "lateral" if (jp - V0.mean(0)) @ lat_ax > 0 else "lateral_neg"
        near = np.nonzero(np.linalg.norm(V0 - jp, axis=1) < 0.09)[0]
        fm = faces_of(F, near) & fh
        draw_mesh(ax, V0, F, axes, view, face_mask=fm, color=(.86, .88, .91), edge=True)
        nz = np.nonzero(R[row])[0]; w = R[row, nz]
        Q = project(V0[nz], axes, view); qj = project(jp, axes, view)[0]
        for q in Q:
            ax.plot([q[0], qj[0]], [q[1], qj[1]], color=col, lw=.7, alpha=.45, zorder=17)
        sc = ax.scatter(Q[:, 0], Q[:, 1], s=90, c=w, cmap="viridis", vmin=w.min(), vmax=w.max(),
                        edgecolor="white", lw=.8, zorder=18)
        for q, wi in zip(Q, w):
            big = wi == w.max()
            ax.annotate(f"{wi:.3f}", q[:2], xytext=(-34, -16) if big else (5, 4), textcoords="offset points",
                        fontsize=8.2 if big else 7.2, color="#000" if big else "#333",
                        weight="bold" if big else "normal", zorder=21)
        ax.scatter(qj[0], qj[1], s=170, color=col, marker="*", edgecolor="white", lw=.9, zorder=20)
        r_ = 0.05; ax.set_xlim(qj[0] - r_, qj[0] + r_); ax.set_ylim(qj[1] - r_ * .8, qj[1] + r_ * .8)
        ax.set_title(f"{'BC'[k]}   {nm} — its 10 vertices and weights", fontsize=10.5, loc="left")
    fig.suptitle("How each added joint is tied to the mesh", fontsize=13, y=1.02)
    fig.text(.5, -.03, r"joint on any registered specimen:   $\mathbf{j} = \sum_{i=1}^{10} w_i\,\mathbf{v}_i$"
             r",     $w_i = \frac{1/d_i}{\sum_k 1/d_k}$,   $d_i$ = distance of template vertex $i$ to the placed bone"
             "     (★ = joint; the same linear form as SMIL's J_regressor)", ha="center", fontsize=10.5)
    save(fig, "F04_regressor_weights")

    # ================= F05 all 20 registered specimens, full-face, head frame
    fig, AX = plt.subplots(4, 5, figsize=(15, 12.6))
    for i, ax in enumerate(AX.ravel()):
        V = d["obs"][i]; Vh, Rk, c, _ = head_aligned(V, M)
        pl, pr = (d["PL"][i] - c) @ Rk, (d["PR"][i] - c) @ Rk
        draw_mesh(ax, Vh, F, axes, "anterior", face_mask=fh)
        joints_on(ax, [pl, pr], axes, "anterior", colors=[CL, CR], s=46, lw=1.6)
        l = d["labels"][i]
        ax.set_title(f"{l}   ·   {d['mass'][i]:.1f} mg   ·   HW {d['HW_mm'][i]:.2f} mm", fontsize=9)
    fig.suptitle("Where b_h_l / b_h_r end up after registration — all 20 Atta specimens (full-face view)",
                 fontsize=13.5, y=.995)
    fig.text(.5, .005, "Each head shown in its own head-aligned frame and scaled to fill its panel; "
             "ordered by body size (01 smallest → 20 largest).", ha="center", fontsize=9.5, color="#555")
    save(fig, "F05_all20_registered_heads_fullface", dpi=220)

    # dorsal companion
    fig, AX = plt.subplots(4, 5, figsize=(15, 12.6))
    for i, ax in enumerate(AX.ravel()):
        V = d["obs"][i]; Vh, Rk, c, _ = head_aligned(V, M)
        pl, pr = (d["PL"][i] - c) @ Rk, (d["PR"][i] - c) @ Rk
        draw_mesh(ax, Vh, F, axes, "dorsal", face_mask=fh)
        joints_on(ax, [pl, pr], axes, "dorsal", colors=[CL, CR], s=46, lw=1.6)
        ax.set_title(f"{d['labels'][i]}   ·   {d['mass'][i]:.1f} mg", fontsize=9)
    fig.suptitle("Same 20 specimens, dorsal view", fontsize=13.5, y=.995)
    save(fig, "F05b_all20_registered_heads_dorsal", dpi=220)

    # ================= F06 overlay on the real scans + F07 data
    wp = {w["label"]: w for w in json.load(open(f"{SHOW}/data/widest_point_check.json"))}
    profiles = []
    pick = ["01", "10", "20"]
    fig, AX = plt.subplots(3, 4, figsize=(15.5, 11.5))
    for i, lab in enumerate(d["labels"]):
        V = d["obs"][i]; Vh, Rk, c, idx = head_aligned(V, M)
        S, SF = load_scan_mesh_normalised(lab)
        edge = np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))
        dist, _ = cKDTree(V[idx]).query(S)
        near = dist < 4 * edge
        Sh = (S[near] - c) @ Rk
        S_all = (S - c) @ Rk
        sfm = faces_of(SF, np.nonzero(near)[0])
        pl, pr = (d["PL"][i] - c) @ Rk, (d["PR"][i] - c) @ Rk
        # scan width profile (lat extent per antero-posterior slice), head coordinates
        ax3 = np.stack(axes)
        C = Sh @ ax3.T; cp = np.array([pl, pr]) @ ax3.T
        lo_, hi_ = C[:, 1].min(), C[:, 1].max()
        edges = np.linspace(lo_, hi_, 26); mids = (edges[1:] + edges[:-1]) / 2; prof = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (C[:, 1] >= a) & (C[:, 1] < b)
            prof.append(np.percentile(C[m, 0], 99.5) - np.percentile(C[m, 0], .5) if m.sum() > 15 else np.nan)
        prof = np.array(prof)
        sm = prof.copy()
        for j in range(len(prof)):                        # 3-bin running median, NaN gaps preserved
            win = prof[max(0, j - 1):j + 2]
            sm[j] = np.nanmedian(win) if np.isfinite(prof[j]) and np.isfinite(win).sum() >= 2 else np.nan
        t = (mids - lo_) / (hi_ - lo_); sm[(t < .06) | (t > .94)] = np.nan
        wmax = np.nanmax(sm); t_rep = (cp[:, 1].mean() - lo_) / (hi_ - lo_); f_ = np.isfinite(sm)
        profiles.append(dict(lab=lab, t=t, w=sm / wmax, t_rep=t_rep,
                             w_rep=float(np.interp(t_rep, t[f_], (sm / wmax)[f_]))))
        if lab in pick:
            r = pick.index(lab)
            for k, v in enumerate(["anterior", "dorsal"]):
                ax = AX[r, 2 * k]
                draw_mesh(ax, S_all, SF, axes, v, face_mask=sfm, color=(.93, .86, .74))
                joints_on(ax, [pl, pr], axes, v, colors=[CL, CR], s=60)
                ax.set_title(f"{lab} · real scan (head) · {v}", fontsize=9.5)
                ax = AX[r, 2 * k + 1]
                draw_mesh(ax, Vh, F, axes, v, face_mask=fh)
                joints_on(ax, [pl, pr], axes, v, colors=[CL, CR], s=60)
                ax.set_title(f"{lab} · registered SMIL head · {v}", fontsize=9.5)
    fig.suptitle("The same joint positions shown on the real scan and on the registered model — smallest (01), middle (10), largest (20)",
                 fontsize=13, y=.995)
    save(fig, "F06_overlay_on_real_scans", dpi=220)

    # ================= F07 widest-point evidence
    fig = plt.figure(figsize=(15, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=.42)
    ax = fig.add_subplot(gs[0])
    for k, p in enumerate(profiles):
        ax.plot(p["t"], p["w"], color="#8fa9c7", lw=1.0, alpha=.8)
        ax.scatter(p["t_rep"], p["w_rep"], color=BLUE, s=30, edgecolor="white", lw=.6, zorder=5)
    ax.axhline(1, color=GREY, ls="--", lw=.9)
    ax.set_xlabel("position along the head, posterior (0) → anterior (1)")
    ax.set_ylabel("head width at that level / maximum head width")
    ax.set_title("A   Head width along the head (real scans);\n     dot = level of the joints", fontsize=10.5, loc="left")
    ax.set_ylim(.55, 1.04); ax.set_xlim(0, 1)
    ax.axvspan(min(p["t_rep"] for p in profiles), max(p["t_rep"] for p in profiles), color=BLUE, alpha=.07, lw=0)
    ax.text(.5, .06, "shaded band = range of joint levels across the 20 specimens",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555", style="italic")

    ax = fig.add_subplot(gs[1])
    labs = d["labels"]
    r_scan = np.array([wp[l]["ratio_rep_to_widest_scan"] for l in labs])
    r_mesh = np.array([wp[l]["ratio_rep_to_widest_mesh"] for l in labs])
    x = np.arange(20)
    ax.axhline(1, color=GREY, ls="--", lw=1)
    ax.scatter(x, 100 * r_scan, s=40, color=BLUE, edgecolor="white", lw=.7, zorder=3, label="vs real scan")
    ax.scatter(x, 100 * r_mesh, s=26, facecolor="none", edgecolor=DARK, lw=.9, zorder=3, label="vs registered mesh")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.8, rotation=90)
    ax.set_ylim(90, 101.5)
    ax.set_ylabel("joint separation / maximum head width (%)")
    ax.set_title(f"B   Joints capture {100*np.median(r_scan):.1f}% of max width", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[2])
    surf = np.array([wp[l]["rep_to_scan_surface_pct_diag"] for l in labs])
    ax.bar(x, surf, color=BLUE, width=.65)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.8, rotation=90)
    ax.set_ylabel("joint → nearest scan surface (% body diagonal)")
    ax.set_title(f"C   Joints lie on the scan surface (median {np.median(surf):.2f}%)", fontsize=10.5, loc="left")
    for a in fig.axes[:1] + fig.axes[2:]:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle("Evidence that the joints measure the widest point of the head", fontsize=13, y=1.03)
    save(fig, "F07_widest_point_evidence")

    # ================= F13 placement consistency in normalised head coordinates
    ax3 = np.stack(axes); pts = []
    for i in range(20):
        Vh, Rk, c, idx = head_aligned(d["obs"][i], M)
        C = Vh[idx] @ ax3.T
        mn, mx = C.min(0), C.max(0)
        for P in (d["PL"][i], d["PR"][i]):
            q = ((P - c) @ Rk) @ ax3.T
            pts.append((q - mn) / (mx - mn))
    pts = np.array(pts).reshape(20, 2, 3)
    CT = V0[idx_head] @ ax3.T; mn0, mx0 = CT.min(0), CT.max(0)
    V0n = (V0 - 0) @ ax3.T; V0n = (V0n - mn0) / (mx0 - mn0)
    fig, AX = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, v, (ix, iy), ttl in ((AX[0], "anterior", (0, 2), "full-face (lateral × dorso-ventral)"),
                                 (AX[1], "dorsal", (0, 1), "dorsal (lateral × antero-posterior)")):
        ax.scatter(V0n[idx_head, ix], V0n[idx_head, iy], s=.6, color="#d7dde4", lw=0)
        for side, col, nm in ((0, CL, "b_h_l"), (1, CR, "b_h_r")):
            P = pts[:, side, [ix, iy]]
            ax.scatter(P[:, 0], P[:, 1], s=30, color=col, edgecolor="white", lw=.6, zorder=5, label=nm)
            mu, cov = P.mean(0), np.cov(P.T); ev, evec = np.linalg.eigh(cov)
            ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
            ax.add_patch(Ellipse(mu, 2 * 2.448 * np.sqrt(ev[1]), 2 * 2.448 * np.sqrt(ev[0]), angle=ang,
                                 fc="none", ec=col, lw=1.3, zorder=4))
        ax.set_aspect("equal"); ax.set_xlim(-.05, 1.05)
        ax.set_xlabel("normalised " + ["lateral", "antero-posterior", "dorso-ventral"][ix])
        ax.set_ylabel("normalised " + ["lateral", "antero-posterior", "dorso-ventral"][iy])
        ax.set_title(ttl, fontsize=10.5)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    AX[0].legend(fontsize=8.5, frameon=False, loc="upper center", ncol=2)
    spread = pts.std(0).mean(0) * 100
    fig.suptitle("The joints land at the same anatomical locus on every specimen "
                 "(20 specimens, 95% ellipses)", fontsize=12.5, y=1.01)
    fig.text(.5, -.02, f"spread (SD, % of head extent): lateral {spread[0]:.2f}  ·  antero-posterior {spread[1]:.2f}"
             f"  ·  dorso-ventral {spread[2]:.2f}", ha="center", fontsize=9.5, color="#555")
    save(fig, "F13_placement_consistency")
    json.dump(dict(spread_sd_pct_head_extent=dict(lat=float(spread[0]), ap=float(spread[1]), dv=float(spread[2]))),
              open(f"{SHOW}/data/placement_consistency.json", "w"), indent=1)
    print("spread", spread)


if __name__ == "__main__":
    main()
