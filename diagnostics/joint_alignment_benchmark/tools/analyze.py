"""Pre-registered analysis (PREREGISTRATION.md §7-8). Reads data/scores_*.csv, writes results/.

Every number in REPORT.md is produced here; nothing is computed by hand.
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
OUT = os.path.join(BENCH, "results")
REF = "A_prod"
ARMS = ["A_prod", "B_nohier", "C_nopartition", "D_nolimits", "E_anteriorlimits", "F_noscalecap",
        "G_allometric", "H_learnedinit", "I_cse", "J_nooffnorm", "K_posefrozen", "L_splitdistal"]
FINAL = "Stage_3_deform_fine"
SESOI = 2.5
REGIONS = ["body_axis", "coxa", "leg_proximal", "leg_distal", "mandible", "antenna"]
TAUS = [5, 10, 15, 20, 25]
RNG = np.random.default_rng(20260914)


# ---------------------------------------------------------------- exact signed-rank machinery
def _null_counts(n):
    """Exact null distribution of the Wilcoxon T+ statistic for n untied, nonzero differences."""
    M = n * (n + 1) // 2
    c = np.zeros(M + 1, dtype=object)
    c[0] = 1
    for k in range(1, n + 1):
        new = c.copy()
        new[k:] = c[k:] + c[:M + 1 - k]
        c = new
    return np.array([float(x) for x in c]) / 2 ** n


def hl_ci(d, level):
    """Hodges-Lehmann estimate and the exact CI from inverting the signed-rank test."""
    d = np.asarray(d, float)
    n = len(d)
    walsh = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    cdf = np.cumsum(_null_counts(n))
    alpha = 1 - level
    k = int(np.searchsorted(cdf, alpha / 2, side="right"))      # P(T <= k-1) <= alpha/2
    k = max(k, 1)
    return float(np.median(walsh)), float(walsh[k - 1]), float(walsh[len(walsh) - k])


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def boot_median_ci(x, B=10000):
    x = np.asarray(x, float)
    bs = np.median(RNG.choice(x, (B, len(x)), replace=True), axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def classify(p_holm, hl, lo90, hi90):
    if p_holm < 0.05 and hl < 0:
        return "IMPROVES"
    if p_holm < 0.05 and hl > 0:
        return "WORSENS"
    if lo90 > -SESOI and hi90 < SESOI:
        return "EQUIVALENT"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------- analyses
def specimen_matrix(S, col, arms, stage=FINAL, drop=None):
    s = S[(S.stage == stage) & S.arm.isin(arms)]
    if drop is not None:
        s = s[~s.sid.isin(drop)]
    return s.groupby(["sid", "arm"])[col].mean().unstack("arm")[arms]


def contrasts(mat, arms):
    rows = []
    for a in arms:
        if a == REF:
            continue
        pair = mat[[REF, a]].dropna()
        d = (pair[a] - pair[REF]).values
        hl, lo95, hi95 = hl_ci(d, 0.95)
        _, lo90, hi90 = hl_ci(d, 0.90)
        p = stats.wilcoxon(d, method="exact").pvalue if np.any(d != 0) else 1.0
        rows.append(dict(arm=a, n=len(d), ref_median=float(np.median(pair[REF])),
                         arm_median=float(np.median(pair[a])), HL=hl, CI95=[lo95, hi95],
                         CI90=[lo90, hi90], p_exact=float(p), wins=int((d < 0).sum()),
                         losses=int((d > 0).sum()), rel_change_median=float(np.median(d / pair[REF].values))))
    adj = holm([r["p_exact"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = float(q)
        r["verdict"] = classify(q, r["HL"], *r["CI90"])
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    S = pd.read_csv(os.path.join(BENCH, "data/scores_specimen.csv"))
    J = pd.read_csv(os.path.join(BENCH, "data/scores_joints.csv"))
    arms = [a for a in ARMS if a in set(S.arm)]
    res = dict(arms=arms, n_specimens=int(S[S.arm == REF].sid.nunique()), sesoi=SESOI)

    # run completeness (failed runs are reported, never dropped)
    comp = S[S.stage == FINAL].groupby(["arm"]).apply(lambda g: sorted(g.seed.unique().tolist()))
    res["seeds_present"] = {a: comp.get(a, []) for a in ARMS}

    # --- primary
    M = specimen_matrix(S, "med_fk", arms)
    M.to_csv(os.path.join(OUT, "primary_specimen_matrix.csv"))
    res["summary"] = {a: dict(median=float(M[a].median()), CI95=boot_median_ci(M[a].dropna()),
                              IQR=[float(M[a].quantile(.25)), float(M[a].quantile(.75))])
                      for a in arms}
    res["primary"] = contrasts(M, arms)

    # seed noise (§8.7): per specimen SD across seeds of med_fk, median over specimens
    sd = S[S.stage == FINAL].groupby(["arm", "sid"]).med_fk.std().groupby("arm").median()
    res["seed_sd"] = {a: float(sd.get(a, np.nan)) for a in arms}
    for r in res["primary"]:
        r["exceeds_seed_sd"] = bool(abs(r["HL"]) > max(res["seed_sd"][REF], res["seed_sd"][r["arm"]]))

    # omnibus Friedman + Nemenyi CD (§8.5)
    full = M.dropna()
    if full.shape[1] >= 3 and len(full) >= 3:
        fr = stats.friedmanchisquare(*[full[a] for a in arms])
        ranks = full.rank(axis=1).mean()
        k, N = len(arms), len(full)
        q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        res["friedman"] = dict(chi2=float(fr.statistic), p=float(fr.pvalue), mean_ranks=ranks.to_dict(),
                               nemenyi_CD=float(q * np.sqrt(k * (k + 1) / (6 * N))), N=N)

    # --- secondary S1/S4/S5/S8
    for col, key in [("med_pa", "S1_PA"), ("bone_med", "S4_bone"), ("side_confusion", "S5_side"),
                     ("chamfer", "S8_chamfer"), ("deform_rms", "S8_deform")]:
        m = specimen_matrix(S, col, arms)
        res[key] = dict(summary={a: float(m[a].median()) for a in arms}, contrasts=contrasts(m, arms))
    ch = specimen_matrix(S, "chamfer", arms)
    res["S8_chamfer_ratio_vs_ref"] = {a: float(np.median(ch[a] / ch[REF])) for a in arms}

    # --- S2 PCK (specimen-weighted), AUC over [0,50]
    Jf = J[J.stage == FINAL]
    grid = np.linspace(0, 50, 101)
    pck, auc = {}, {}
    for a in arms:
        ja = Jf[Jf.arm == a]
        per = ja.groupby(["sid", "seed"]).err_fk.apply(lambda e: np.array([(e.values <= t).mean() for t in grid]))
        curve = np.mean(np.stack(per.groupby(level=0).apply(lambda s: np.mean(np.stack(s.values), 0)).values), 0)
        pck[a] = {str(t): float(np.interp(t, grid, curve)) for t in TAUS}
        auc[a] = float(np.trapz(curve, grid) / 50)
        pck[a]["curve"] = curve.tolist()
    res["S2_PCK"] = pck
    res["S2_AUC"] = auc
    res["pck_grid"] = grid.tolist()

    # --- S3 regions (exploratory)
    R = Jf.groupby(["arm", "seed", "sid", "region"]).err_fk.median().groupby(["arm", "sid", "region"]).mean()
    reg = {}
    for rg in REGIONS:
        m = R.xs(rg, level="region").unstack("arm").reindex(columns=arms)
        if m[REF].notna().sum() < 5:
            continue
        cs = contrasts(m, arms)
        reg[rg] = dict(n=int(m[REF].notna().sum()), ref_median=float(m[REF].median()),
                       medians={a: float(m[a].median()) for a in arms},
                       contrasts={c["arm"]: dict(HL=c["HL"], CI95=c["CI95"], p_exact=c["p_exact"],
                                                 wins=c["wins"], losses=c["losses"]) for c in cs})
    res["S3_regions"] = reg

    # --- S6 systematic bias per joint (reference arm): mean signed error, LOSO-corrected sensitivity
    ja = Jf[Jf.arm == REF].groupby(["sid", "joint"])[["d_ant", "d_dors", "d_right", "err_fk"]].mean().reset_index()
    bias = ja.groupby("joint")[["d_ant", "d_dors", "d_right"]].mean()
    bias["bias_norm"] = np.linalg.norm(bias[["d_ant", "d_dors", "d_right"]].values, axis=1)
    bias["median_err"] = ja.groupby("joint").err_fk.median()
    bias["bias_share"] = bias.bias_norm / bias.median_err
    # Hotelling-free check: sign consistency of the dominant component across specimens
    res["S6_bias_per_joint"] = bias.round(3).reset_index().to_dict("records")
    corr = []
    for sid, g in ja.groupby("sid"):
        others = ja[ja.sid != sid].groupby("joint")[["d_ant", "d_dors", "d_right"]].mean()
        for _, r in g.iterrows():
            if r.joint in others.index:
                v = np.array([r.d_ant, r.d_dors, r.d_right]) - others.loc[r.joint].values
                corr.append(dict(sid=sid, joint=r.joint, raw=float(np.linalg.norm([r.d_ant, r.d_dors, r.d_right])),
                                 corrected=float(np.linalg.norm(v))))
    corr = pd.DataFrame(corr)
    cs = corr.groupby("sid")[["raw", "corrected"]].median()
    res["S6_loso_bias_corrected"] = dict(raw_median=float(cs.raw.median()), corrected_median=float(cs.corrected.median()),
                                         share_removable=float(1 - cs.corrected.median() / cs.raw.median()))

    # --- S7 variance decomposition: MixedLM on log error, crossed specimen & joint, run-level vc
    try:
        import statsmodels.formula.api as smf
        d = Jf[Jf.arm.isin(arms)].copy()
        d["lerr"] = np.log(np.clip(d.err_fk, 0.1, None))
        d["run"] = d.sid + ":" + d.arm + ":" + d.seed.astype(str)
        d["arm"] = pd.Categorical(d.arm, categories=arms)
        d["one"] = 1
        md = smf.mixedlm("lerr ~ C(arm)", d, groups="one",
                         vc_formula={"sid": "0 + C(sid)", "joint": "0 + C(joint)", "run": "0 + C(run)"})
        fit = md.fit(method="lbfgs", reml=True)
        fe = {k.replace("C(arm)[T.", "").rstrip("]"): dict(ratio=float(np.exp(v)),
                                                           CI95=[float(np.exp(fit.conf_int().loc[k, 0])),
                                                                 float(np.exp(fit.conf_int().loc[k, 1]))])
              for k, v in fit.fe_params.items() if k.startswith("C(arm)")}
        vcn = dict(zip(md.exog_vc.names, fit.vcomp))
        res["S7_mixed"] = dict(fixed_ratio_vs_ref=fe, var_components={**{k: float(v) for k, v in vcn.items()},
                                                                      "residual": float(fit.scale)})
    except Exception as e:                                                   # reported, not hidden
        res["S7_mixed"] = dict(error=repr(e))

    # --- sensitivity analyses: does each primary verdict survive?
    G = S[(S.stage == FINAL) & (S.arm == REF)].drop_duplicates("sid")
    chain = G[G.tier == "CHAIN"].sid.tolist()
    mirr = G[G.tier == "MIRRORED"].sid.tolist()
    sens = {}
    for name, col, drop in [("REG_joints", "med_reg", None), ("SKIN_joints", "med_skin", None), ("mean_not_median", "mean_fk", None),
                            ("centroid_size_norm", "med_fk_csize", None), ("excl_CHAIN", "med_fk", chain),
                            ("excl_MIRRORED", "med_fk", mirr)]:
        m = specimen_matrix(S, col, arms, drop=drop)
        cs = contrasts(m, arms)
        sens[name] = {c["arm"]: dict(HL=c["HL"], CI95=c["CI95"], p_holm=c["p_holm"], verdict=c["verdict"],
                                     n=c["n"]) for c in cs}
    res["sensitivity"] = sens

    # --- stage progression (exploratory)
    prog = S.groupby(["arm", "stage", "sid"]).med_fk.mean().groupby(["arm", "stage"]).median().unstack("stage")
    order = ["H0_body", "H1_legs", "H2_joint", "Stage_2_deform_coarse", FINAL]
    res["stage_progression"] = {a: {st: (float(prog.loc[a, st]) if st in prog.columns and not np.isnan(prog.loc[a, st]) else None)
                                    for st in order} for a in arms if a in prog.index}
    stage_contr = {}
    pa = S[S.arm == REF].groupby(["sid", "stage"]).med_fk.mean().unstack("stage")
    present = [st for st in order if st in pa.columns]
    for s0, s1 in zip(present[:-1], present[1:]):
        if True:
            d = (pa[s1] - pa[s0]).dropna().values
            hl, lo, hi = hl_ci(d, 0.95)
            stage_contr[f"{s0}->{s1}"] = dict(HL=hl, CI95=[lo, hi], p_exact=float(stats.wilcoxon(d, method="exact").pvalue),
                                             improved=int((d < 0).sum()), n=len(d))
    res["stage_contrasts_ref"] = stage_contr

    # --- oracle references O0/O1/O2 (exploratory, never ranked with deployable arms)
    ora = {}
    a0 = Jf[(Jf.arm == REF) & (Jf.seed == 0)]
    if "O0_control" in set(Jf.arm):
        def spec_med(df):
            return df.groupby("sid").err_fk.median()
        base = spec_med(a0)
        for arm in ("O0_control", "O1_all"):
            m = spec_med(Jf[Jf.arm == arm]).reindex(base.index)
            d = (m - base).values
            hl, lo, hi = hl_ci(d, 0.95)
            ora[arm] = dict(start_median=float(base.median()), median=float(m.median()), HL=hl, CI95=[lo, hi],
                            improved=int((d < 0).sum()), n=len(d))
        for rg in REGIONS:
            arm = f"O2_heldout_{rg}"
            if arm not in set(Jf.arm):
                continue
            b = a0[a0.region == rg].groupby("sid").err_fk.median()
            o = Jf[(Jf.arm == arm) & (Jf.region == rg)].groupby("sid").err_fk.median().reindex(b.index)
            o1 = Jf[(Jf.arm == "O1_all") & (Jf.region == rg)].groupby("sid").err_fk.median().reindex(b.index)
            d = (o - b).dropna().values
            hl, lo, hi = hl_ci(d, 0.95)
            ora[arm] = dict(region=rg, start_median=float(b.median()), heldout_median=float(o.median()),
                            supervised_median_O1=float(o1.median()), HL_vs_start=hl, CI95=[lo, hi],
                            improved=int((d < 0).sum()), n=len(d),
                            transfer_fraction=float((b.median() - o.median()) / max(b.median() - o1.median(), 1e-9)))
    res["oracle"] = ora

    # legacy continuity: Z8 production vs the new reference arm
    if "Z8_legacy_prod" in set(S.arm):
        z = S[(S.arm == "Z8_legacy_prod") & (S.stage == FINAL)].set_index("sid").med_fk
        a = M[REF]
        res["legacy_vs_ref"] = dict(z8_median=float(z.median()), ref_median=float(a.median()),
                                    spearman=float(stats.spearmanr(z.reindex(a.index), a).correlation))

    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("summary", "seed_sd")}, indent=1))
    print(pd.DataFrame(res["primary"])[["arm", "arm_median", "HL", "CI95", "p_exact", "p_holm", "wins",
                                        "losses", "verdict"]].to_string())


if __name__ == "__main__":
    main()
