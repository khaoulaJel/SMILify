"""Publication figures for the joint-alignment benchmark. Reads results/results.json and data/.

Design: emphasis over rainbow -- the reference arm in ink, others in gray, direct labels on the
marks that matter; diverging blue<->red with a gray midpoint for signed effects; one sequential
blue for magnitudes; verdicts are always written out, never carried by colour alone.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
FIG = os.path.join(BENCH, "figures")
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, RED, ORANGE, AQUA = "#2a78d6", "#e34948", "#eb6834", "#1baf7a"
GRAY_MARK = "#b9b8b1"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a9a", "#b52f2f"])
LABEL = {"A_prod": "A  D1_PROD (reference)", "B_nohier": "B  no hierarchical placement",
         "C_nopartition": "C  no part partition", "D_nolimits": "D  no joint limits",
         "E_anteriorlimits": "E  + anterior limits", "F_noscalecap": "F  no scale cap",
         "G_allometric": "G  allometric prior", "H_learnedinit": "H  learned pose init",
         "I_cse": "I  CSE correspondence", "J_nooffnorm": "J  no offset/normal penalty",
         "K_posefrozen": "K  pose frozen in surface stages", "L_splitdistal": "L  split distal groups"}
SHORT = {k: v.split("  ")[0] + " " + v.split("  ")[1].split(" (")[0] for k, v in LABEL.items()}
REGION_LABEL = {"body_axis": "body axis", "coxa": "coxae", "leg_proximal": "trochanter/femur",
                "leg_distal": "tibia/tarsus", "mandible": "mandibles", "antenna": "antennae"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def verdict_mark(v):
    return {"IMPROVES": "▼ improves", "WORSENS": "▲ worsens", "EQUIVALENT": "= equivalent",
            "INCONCLUSIVE": "? inconclusive"}[v]


# ---------------------------------------------------------------- F1 ground-truth audit
def fig_gt_audit():
    audit = json.load(open(os.path.join(BENCH, "data/gt_registration_audit.json")))
    ia = json.load(open(os.path.join(BENCH, "probes/instrument_audit_PROBE_out.json")))
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    tiers = ["EXACT", "MIRRORED", "CHAIN"]
    for k, t in enumerate(tiers):
        v = [a["reg_resid_rel"] for a in audit if a["tier"] == t]
        ax[0].scatter(np.full(len(v), k) + np.linspace(-.12, .12, len(v)), v, s=28, color=BLUE,
                      edgecolor="white", linewidth=1.2, zorder=3)
        ax[0].text(k + .2, np.median(v), f"n={len(v)}", ha="left", va="center", color=INK2, fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(3), ["exact", "mirrored\n+ L/R swap", "older mesh\n+ ICP"])
    ax[0].set_ylabel("registration residual / mesh diagonal")
    ax[0].set_xlim(-.6, 2.6)
    ax[0].set_title("a  GT → scan, without any fit")
    ex = [a for a in audit if a["tier"] == "EXCLUDED"]
    if ex:
        ax[0].text(0.02, 0.98, f"excluded: {ex[0]['sid'].split('_CASENT')[0].replace('_', ' ')}\n(its scene mesh is another specimen)",
                   transform=ax[0].transAxes, fontsize=7, color=INK2, va="top")
    ax[0].set_ylim(1e-8, 1e-1)

    mir = [r for r in ia if "FK_new_noswap" in r]
    ys = sorted((r["FK_new_abs"], r["sid"].split("_")[0]) for r in mir)
    placed = []
    for y, name in ys:
        yy = y if not placed or y - placed[-1] > 3.5 else placed[-1] + 3.5
        placed.append(yy)
        ax[1].text(1.06, yy, name, va="center", fontsize=7, color=INK2)
    for i, r in enumerate(mir):
        ax[1].plot([0, 1], [r["FK_new_noswap"], r["FK_new_abs"]], color=GRAY_MARK, lw=1.5, zorder=2)
        ax[1].scatter([0, 1], [r["FK_new_noswap"], r["FK_new_abs"]], s=26, color=[MUTED, BLUE],
                      edgecolor="white", linewidth=1.2, zorder=3)
    ax[1].set_xticks([0, 1], ["labels as clicked", "reflected + L/R swap"])
    ax[1].set_xlim(-.3, 1.55)
    ax[1].set_ylabel("production joint error (% WL)")
    ax[1].set_title(f"b  swap confirmed {sum(r['FK_new_abs'] < r['FK_new_noswap'] for r in mir)}/{len(mir)}")

    for r in ia:
        ax[2].plot([0, 1], [r["REG_old_V9"], r["REG_new_abs"]], color=GRAY_MARK, lw=1.5, zorder=2)
        ax[2].scatter([0, 1], [r["REG_old_V9"], r["REG_new_abs"]], s=26, color=[MUTED, BLUE],
                      edgecolor="white", linewidth=1.2, zorder=3)
    mo, mn = np.median([r["REG_old_V9"] for r in ia]), np.median([r["REG_new_abs"] for r in ia])
    ax[2].text(-.08, mo, f"{mo:.1f}", ha="right", va="center", fontsize=8, color=INK)
    ax[2].text(1.08, mn, f"{mn:.1f}", ha="left", va="center", fontsize=8, color=INK)
    ax[2].set_xticks([0, 1], ["old ruler\n(GT fitted to prod. joints)", "new ruler\n(fit-independent)"])
    ax[2].set_xlim(-.4, 1.4)
    ax[2].set_ylabel("production joint error (% WL)")
    ax[2].set_title("c  the old ruler changed the answer")
    fig.tight_layout()
    save(fig, "F1_gt_audit")


# ---------------------------------------------------------------- F2 forest plot
def fig_forest(res):
    rows = sorted(res["primary"], key=lambda r: r["HL"])
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(rows) + 1.2))
    s = res["sesoi"]
    ax.axvspan(-s, s, color="#f0efec", zorder=0, lw=0)
    ax.axvline(0, color=BASE, lw=1)
    for i, r in enumerate(rows):
        c = BLUE if r["verdict"] == "IMPROVES" else RED if r["verdict"] == "WORSENS" else INK2
        ax.plot(r["CI95"], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter(r["HL"], i, s=46, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(1.02, i, f"{verdict_mark(r['verdict'])}   {r['wins']}/{r['n']} better · p_Holm {r['p_holm']:.3f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows)), [LABEL[r["arm"]] for r in rows])
    ax.set_xlabel("paired change in joint error vs A  (Hodges–Lehmann, % of Weber's length; 95% exact CI)")
    ax.text(0, len(rows) - .35, f"±{s:g}% WL: annotation-precision band", ha="center", fontsize=7, color=MUTED)
    ax.set_title(f"Which strategy moves joint alignment?  (n = {res['n_specimens']} specimens, 3 seeds each)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-.7, len(rows) - .1)
    save(fig, "F2_forest_primary")


# ---------------------------------------------------------------- F3 paired specimen lines
def fig_paired(res):
    M = pd.read_csv(os.path.join(BENCH, "results/primary_specimen_matrix.csv"), index_col=0)
    arms = [a for a in res["arms"] if a != "A_prod"]
    order = {r["arm"]: r for r in res["primary"]}
    nc = 4
    nr = int(np.ceil(len(arms) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(10, 2.35 * nr), sharey=True)
    ymax = np.nanmax(M.values) * 1.05
    for ax, a in zip(axs.ravel(), arms):
        for sid, row in M.iterrows():
            d = row[a] - row["A_prod"]
            c = BLUE if d < -res["sesoi"] else RED if d > res["sesoi"] else GRAY_MARK
            ax.plot([0, 1], [row["A_prod"], row[a]], color=c, lw=1.3)
            ax.scatter([0, 1], [row["A_prod"], row[a]], s=12, color=c, zorder=3, edgecolor="white", linewidth=.6)
        r = order[a]
        ax.set_title(f"{SHORT[a]}\n", fontsize=8.5)
        ax.text(0, 1.02, f"{r['wins']} better · {r['losses']} worse · {verdict_mark(r['verdict'])}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks([0, 1], ["A", a[0]])
        ax.set_xlim(-.25, 1.25)
        ax.set_ylim(0, ymax)
    for ax in axs.ravel()[len(arms):]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("joint error (% WL)")
    fig.suptitle("Per-specimen paired change vs D1_PROD  (blue: better by more than the precision band; red: worse)",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "F3_paired_specimens")


# ---------------------------------------------------------------- F4 region heatmap
def fig_regions(res):
    reg = res["S3_regions"]
    regions = [r for r in REGION_LABEL if r in reg]
    arms = [a for a in res["arms"] if a != "A_prod"]
    H = np.array([[reg[r]["contrasts"][a]["HL"] for r in regions] for a in arms])
    lim = max(5, np.nanmax(np.abs(H)))
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(arms) + 1.6))
    im = ax.imshow(H, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i, a in enumerate(arms):
        for j, r in enumerate(regions):
            c = reg[r]["contrasts"][a]
            sig = c["CI95"][1] < 0 or c["CI95"][0] > 0
            ax.text(j, i, f"{c['HL']:+.1f}{'*' if sig else ''}", ha="center", va="center", fontsize=7,
                    color="white" if abs(c["HL"]) > .6 * lim else INK)
    ax.set_xticks(range(len(regions)), [f"{REGION_LABEL[r]}\nA: {reg[r]['ref_median']:.0f}% (n={reg[r]['n']})" for r in regions])
    ax.set_yticks(range(len(arms)), [SHORT[a] for a in arms])
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label("Δ region error vs A (% WL)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Region-level change vs A: blue helps, red hurts  (* 95% CI excludes 0; exploratory)")
    save(fig, "F4_region_effects")


# ---------------------------------------------------------------- F5 PCK
def fig_pck(res):
    grid = np.array(res["pck_grid"])
    auc = res["S2_AUC"]
    best = max((a for a in res["arms"] if a != "A_prod"), key=lambda a: auc[a])
    worst = min(res["arms"], key=lambda a: auc[a])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for a in res["arms"]:
        if a in ("A_prod", best, worst):
            continue
        ax.plot(grid, res["S2_PCK"][a]["curve"], color=GRAY_MARK, lw=1.2)
    for a, c in (("A_prod", INK), (best, BLUE), (worst, RED)):
        y = res["S2_PCK"][a]["curve"]
        ax.plot(grid, y, color=c, lw=2)
        ax.text(grid[-1] + .8, y[-1], f"{SHORT[a]}  AUC {auc[a]:.3f}", va="center", fontsize=7.5, color=INK)
    ax.axvline(res["sesoi"], color=MUTED, lw=.8)
    ax.set_xlabel("threshold τ (% of Weber's length)")
    ax.set_ylabel("PCK: share of joints within τ")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)
    ax.set_title("Percentage of correct keypoints  (gray: other arms)")
    save(fig, "F5_pck_curves")


# ---------------------------------------------------------------- F6 accuracy vs surface cost
def fig_pareto(res):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    pts = {a: (res["S8_chamfer_ratio_vs_ref"][a], res["summary"][a]["median"]) for a in res["arms"]}
    for a, (x, y) in pts.items():
        lo, hi = res["summary"][a]["CI95"]
        ax.plot([x, x], [lo, hi], color=GRID, lw=3, solid_capstyle="round", zorder=1)
    cluster = [a for a, (x, y) in pts.items() if 0.9 < x < 1.1 and a != "A_prod"]
    for a, (x, y) in pts.items():
        ax.scatter(x, y, s=48, color=INK if a == "A_prod" else BLUE, edgecolor="white", linewidth=1.5, zorder=3)
        if a == "A_prod":
            ax.annotate("A (reference)", (x, y), xytext=(-8, -14), textcoords="offset points", fontsize=8, ha="right", color=INK)
        elif a not in cluster:
            ax.annotate(SHORT[a], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK)
    ax.text(0.99, 0.02, "unlabelled, within ±10% of A's chamfer:\n" + "\n".join(SHORT[a] for a in sorted(cluster, key=lambda a: pts[a][1])),
            transform=ax.transAxes, fontsize=7, color=INK2, va="bottom", ha="right")
    ax.axvline(1, color=BASE, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 3], ["0.5×", "1×", "2×", "3×"])
    ax.minorticks_off()
    ax.set_xlabel("surface chamfer, median ratio to A (log scale; < 1 = closer surface fit)")
    ax.set_ylabel("median joint error (% WL); bar = 95% CI")
    ax.set_title("A closer surface does not mean a better skeleton")
    save(fig, "F6_accuracy_vs_chamfer")


# ---------------------------------------------------------------- F7 stage progression
def fig_stages(res):
    order = ["H0_body", "H1_legs", "H2_joint", "Stage_2_deform_coarse", "Stage_3_deform_fine"]
    names = ["H0 body", "H1 legs", "H2 joints", "S2 coarse", "S3 fine"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ends = []
    for a, sp in res["stage_progression"].items():
        y = [sp.get(s) for s in order]
        if sum(v is not None for v in y) < 2:
            continue
        xs = [i for i, v in enumerate(y) if v is not None]
        ys = [v for v in y if v is not None]
        c, lw = (INK, 2.2) if a == "A_prod" else (GRAY_MARK, 1.2)
        ax.plot(xs, ys, color=c, lw=lw, marker="o", ms=4, zorder=3 if a == "A_prod" else 2)
        ends.append((ys[-1], a))
    last = -1e9
    for y, a in sorted(ends):
        yy = max(y, last + 1.1)
        last = yy
        ax.text(4.12, yy, a[0], fontsize=7.5, va="center", color=INK if a == "A_prod" else INK2)
    ax.set_xticks(range(5), names)
    ax.set_ylabel("median joint error (% WL)")
    sc = res.get("stage_contrasts_ref", {})
    txt = "\n".join(f"{k.replace('Stage_', 'S').replace('_deform_coarse', '').replace('_deform_fine', '')}: "
                    f"{v['HL']:+.1f} [{v['CI95'][0]:+.1f}, {v['CI95'][1]:+.1f}]" for k, v in sc.items())
    ax.text(1.02, .5, "A, per step (HL, 95% CI)\n" + txt, transform=ax.transAxes, fontsize=7, color=INK2, va="center")
    ax.set_title("Where in the pipeline is joint alignment won or lost?")
    save(fig, "F7_stage_progression")


# ---------------------------------------------------------------- F8 per-joint error on the skeleton
def fig_skeleton(res):
    sys.path.insert(0, HERE)
    import model_joints as MJ
    import pickle
    names = MJ.joint_names()
    REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    Jr = np.asarray(dd["J_regressor"].todense() if hasattr(dd["J_regressor"], "todense") else dd["J_regressor"])
    V = np.asarray(dd["v_template"])
    J = Jr @ V if Jr.shape[1] == V.shape[0] else Jr.T @ V
    par = np.asarray(dd["kintree_table"])[0].astype(int)
    Jall = pd.read_csv(os.path.join(BENCH, "data/scores_joints.csv"))
    Jf = Jall[(Jall.stage == "Stage_3_deform_fine") & (Jall.arm == "A_prod")]
    per = Jf.groupby(["sid", "joint"]).err_fk.mean().groupby("joint").median()
    # template axes: PCA of template vertices -> long axis x, then the two others
    from score import local_frame
    Fr = local_frame({n: J[k] for k, n in enumerate(names)})
    P = (J - J.mean(0)) @ Fr.T                        # anterior, dorsal, right
    vmax = np.nanpercentile(per.values, 95)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (i, j, t) in zip(axs, [(0, 2, "top view (anterior →, right ↑)"), (0, 1, "side view (anterior →, dorsal ↑)")]):
        for k, p in enumerate(par):
            if p >= 0 and not names[k].startswith("w_"):
                ax.plot(P[[p, k], i], P[[p, k], j], color=BASE, lw=1.2, zorder=1)
        for k, n in enumerate(names):
            if n in per.index:
                ax.scatter(P[k, i], P[k, j], s=30 + 3 * per[n], c=[per[n]], cmap=SEQ, vmin=0, vmax=vmax,
                           edgecolor="white", linewidth=1.2, zorder=3)
        if i == 0 and j == 2:
            for n in per.sort_values().index[-4:]:
                k = names.index(n)
                ax.annotate(f"{n}: {per[n]:.0f}%", (P[k, i], P[k, j]), xytext=(6, 6), textcoords="offset points",
                            fontsize=7, color=INK)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(t, fontsize=8.5)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axs, fraction=.02, pad=.04)
    cb.set_label("median joint error, D1_PROD (% WL)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Where the production skeleton is wrong  (template skeleton; dot size and colour = error)",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=INK)
    save(fig, "F8_joint_error_skeleton")


# ---------------------------------------------------------------- F9 critical difference
def fig_cd(res):
    fr = res.get("friedman")
    if not fr:
        return
    ranks = sorted(fr["mean_ranks"].items(), key=lambda kv: kv[1])
    k = len(ranks)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(1, k)
    ax.set_ylim(-.05, 1.05)
    ax.axis("off")
    ax.plot([1, k], [.8, .8], color=INK2, lw=1)
    for t in range(1, k + 1):
        ax.plot([t, t], [.8, .83], color=INK2, lw=1)
        ax.text(t, .86, str(t), ha="center", fontsize=7, color=INK2)
    half = int(np.ceil(k / 2))
    for idx, (a, r) in enumerate(ranks):
        left = idx < half
        y = .55 - .09 * (idx if left else k - 1 - idx)
        xe = 1 - .3 if left else k + .3
        ax.plot([r, r, xe], [.8, y, y], color=INK if a == "A_prod" else INK2, lw=1)
        ax.text(xe - .05 if left else xe + .05, y, f"{SHORT[a]} ({r:.1f})", ha="right" if left else "left",
                va="center", fontsize=7, color=INK)
    cd = fr["nemenyi_CD"]
    ax.plot([1, 1 + cd], [.97, .97], color=INK, lw=2)
    ax.text(1 + cd + .1, .97, f"critical difference (Nemenyi, α=0.05) = {cd:.2f}", va="center", fontsize=7.5, color=INK)
    # cliques: maximal runs of arms whose mean ranks lie within CD (not significantly different)
    rv = [r for _, r in ranks]
    cl, yb = [], .76
    for i in range(k):
        j = max(t for t in range(i, k) if rv[t] - rv[i] <= cd)
        if j > i and not any(c0 <= i and j <= c1 for c0, c1 in cl):
            cl.append((i, j))
    for n_, (i, j) in enumerate(cl):
        y = yb - .025 * n_
        ax.plot([rv[i] - .03, rv[j] + .03], [y, y], color=INK, lw=2.5, solid_capstyle="round")
    ax.set_title(f"Mean rank over specimens (lower = better; bars join arms not significantly different).  "
                 f"Friedman χ² = {fr['chi2']:.1f}, p = {fr['p']:.2g}", loc="left", pad=14)
    save(fig, "F9_critical_difference")


# ---------------------------------------------------------------- F10 seed noise vs effect
def fig_seed(res):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for r in res["primary"]:
        sd = max(res["seed_sd"]["A_prod"], res["seed_sd"][r["arm"]])
        ax.scatter(sd, abs(r["HL"]), s=36, color=BLUE if r["exceeds_seed_sd"] else GRAY_MARK,
                   edgecolor="white", linewidth=1.3, zorder=3)
        ax.annotate(r["arm"][0], (sd, abs(r["HL"])), xytext=(4, 3), textcoords="offset points", fontsize=7.5)
    m = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, m], [0, m], color=BASE, lw=1)
    ax.text(m * .6, m * .52, "effect = seed noise", rotation=33, fontsize=7, color=MUTED)
    ax.set_xlabel("seed SD of per-specimen error (% WL)")
    ax.set_ylabel("|paired effect vs A| (% WL)")
    ax.set_title("Is each effect larger than optimiser noise?")
    save(fig, "F10_seed_noise")


def main():
    res = json.load(open(os.path.join(BENCH, "results/results.json")))
    fig_gt_audit()
    fig_forest(res)
    fig_paired(res)
    fig_regions(res)
    fig_pck(res)
    fig_pareto(res)
    fig_stages(res)
    fig_cd(res)
    fig_seed(res)
    fig_oracle(res)
    try:
        fig_skeleton(res)
    except Exception as e:
        print("skeleton figure failed:", repr(e))



# ---------------------------------------------------------------- F11 qualitative views
def fig_qualitative(arm_npz, sids, name, title):
    """Scan (gray points), expert joints (open ink rings), model FK joints (filled blue), error
    segments, in each specimen's GT anatomical frame. arm_npz: {label: [npz paths]}."""
    sys.path.insert(0, HERE)
    import model_joints as MJ
    from score import local_frame, read_obj, EXCL
    names = MJ.joint_names()
    G = json.load(open(os.path.join(BENCH, "data/gt_joints_fitframe.json")))
    fits = {lab: {} for lab in arm_npz}
    for lab, paths in arm_npz.items():
        for p in paths:
            fits[lab].update(MJ.load_fit(p, wanted=set(sids)))
    labs = list(arm_npz)
    fig, axs = plt.subplots(len(labs) * 2, len(sids), figsize=(3.4 * len(sids), 2.1 * 2 * len(labs)))
    axs = np.atleast_2d(axs)
    for c, sid in enumerate(sids):
        g = G[sid]
        P = {n: np.asarray(r["fit"]) for n, r in g["joints"].items() if n not in EXCL}
        Fr = local_frame(P)
        V, _ = read_obj(os.path.join("/hpcwork/nao48500/worker_alt_data", f"{sid}_processed.obj"))
        cc = V.mean(0)
        V = (V - cc) / np.abs(V - cc).max()
        V = V[np.random.default_rng(0).choice(len(V), min(12000, len(V)), replace=False)]
        o = np.mean(list(P.values()), 0)
        to = lambda X: (np.asarray(X) - o) @ Fr.T / g["WL_fit"]        # units of WL, anterior/dorsal/right
        Vl = to(V)
        for r, lab in enumerate(labs):
            J = fits[lab][sid]["FK"]
            use = [n for n in names if n in P]
            gt = to([P[n] for n in use])
            pr = to([J[names.index(n)] for n in use])
            err = np.median(np.linalg.norm(gt - pr, axis=1)) * 100
            for k, (i, j, vn) in enumerate([(0, 2, "top (anterior →, right ↑)"), (0, 1, "side (anterior →, dorsal ↑)")]):
                ax = axs[r * 2 + k, c]
                ax.scatter(Vl[:, i], Vl[:, j], s=.3, color="#d8d7d0", rasterized=True, zorder=1)
                for a_, b_ in zip(gt, pr):
                    ax.plot([a_[i], b_[i]], [a_[j], b_[j]], color=RED, lw=.8, zorder=2)
                ax.scatter(gt[:, i], gt[:, j], s=16, facecolor="none", edgecolor=INK, linewidth=.9, zorder=3)
                ax.scatter(pr[:, i], pr[:, j], s=9, color=BLUE, zorder=4)
                ax.set_aspect("equal")
                ax.axis("off")
                if k == 0:
                    ax.set_title(f"{sid.split('_CASENT')[0].replace('_', ' ')} — {lab}\nmedian {err:.1f}% WL",
                                 fontsize=8)
                if c == 0:
                    ax.text(-.02, .5, vn, transform=ax.transAxes, rotation=90, ha="right", va="center",
                            fontsize=7, color=INK2)
    fig.suptitle(title + "   ○ expert joint · ● model joint · red: error", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- F12 oracle reach vs transfer
def fig_oracle(res):
    ora = res.get("oracle", {})
    rows = [v for k, v in ora.items() if k.startswith("O2_")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, r in enumerate(rows):
        xs = [r["start_median"], r["heldout_median"], r["supervised_median_O1"]]
        ax.plot([min(xs), max(xs)], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(r["start_median"], i, s=46, color=INK, zorder=3, edgecolor="white", linewidth=1.3)
        ax.scatter(r["heldout_median"], i, s=46, color=ORANGE, zorder=3, edgecolor="white", linewidth=1.3)
        ax.scatter(r["supervised_median_O1"], i, s=46, color=BLUE, zorder=3, edgecolor="white", linewidth=1.3)
        gain = r["start_median"] - r["supervised_median_O1"]
        tr = (f"transfer {100 * r['transfer_fraction']:.0f}%" if gain > res["sesoi"]
              else "supervision itself gains < 2.5% WL")
        ax.text(1.02, i, f"held out: {r['HL_vs_start']:+.1f} [{r['CI95'][0]:+.1f}, {r['CI95'][1]:+.1f}] · "
                         f"{r['improved']}/{r['n']} improved\n{tr}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)), [REGION_LABEL[r["region"]] for r in rows])
    ax.set_xlabel("median region joint error (% WL)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
                       ((INK, "D1_PROD (start)"), (ORANGE, "region held out (O2)"), (BLUE, "region supervised (O1)"))],
              loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=7.5)
    ax.set_title("Oracle: does supervising the other regions fix this one?", pad=26)
    save(fig, "F12_oracle_transfer")


if __name__ == "__main__":
    main()
