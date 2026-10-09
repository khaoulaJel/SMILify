"""M4-B -- scale-free shape morphology on the FROZEN M4-A admissible set.

Per PREREGISTRATION_M4_corpus_analysis.md §2. B-iii (pooled absolute allometry) is prohibited.
B-i (within-genus allometry) is reported as NOT EXECUTABLE -- see RESULTS doc.
This runs B-ii only: scale-free ratios, which survive the scale blocker by construction.
"""
import json, os, collections
import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "diagnostics/morphometric_validation/out")
SEED = 20260910


def perm_kruskal(vals, groups, n_perm=20000, seed=SEED):
    """Kruskal-Wallis H with a permutation null. Reported instead of the asymptotic chi2 p for the
    same reason as M4-A: many small groups."""
    keys = sorted(set(groups))
    idx = {k: np.array([i for i, g in enumerate(groups) if g == k]) for k in keys}
    def H(v):
        return stats.kruskal(*[v[idx[k]] for k in keys]).statistic
    obs = H(vals)
    rng = np.random.default_rng(seed)
    null = np.array([H(rng.permutation(vals)) for _ in range(n_perm)])
    return float(obs), float((null >= obs).mean()), len(keys)


def eta_sq(vals, groups):
    """Proportion of variance between groups (on ranks, to match the KW framing)."""
    r = stats.rankdata(vals)
    keys = sorted(set(groups))
    gm = r.mean()
    ss_b = sum(len([1 for g in groups if g == k]) * (r[[i for i, g in enumerate(groups) if g == k]].mean() - gm) ** 2
               for k in keys)
    ss_t = ((r - gm) ** 2).sum()
    return float(ss_b / ss_t)


def main():
    d = json.load(open(os.path.join(OUT, "m4a_admissible_set.json")))
    rows = d["per_specimen"]
    print(f"[m4b] frozen admissible set: {d['n']} specimens "
          f"(HW {d['HW_admissible']}, HL {d['HL_admissible']})")

    # ratio -> (needs HW, needs HL)
    ratios = {"HW/WL": (True, False), "HL/WL": (False, True), "HW/HL": (True, True)}
    res = {"preregistration": "PREREGISTRATION_M4_corpus_analysis.md §2 (B-ii only)",
           "B_i_status": "NOT EXECUTABLE -- see RESULTS_M4B doc; premise (within-genus scale "
                         "comparability) fails for 12/45 genera with n>=5",
           "B_iii_status": "PROHIBITED by pre-registration",
           "ratios": {}}

    for name, (nhw, nhl) in ratios.items():
        sub = [r for r in rows
               if (r["HW_admissible"] or not nhw) and (r["HL_admissible"] or not nhl)]
        if name == "HW/WL":   v = np.array([r["HW_WL"] for r in sub])
        elif name == "HL/WL": v = np.array([r["HL_WL"] for r in sub])
        else:                 v = np.array([r["HW"] / r["HL"] for r in sub])
        g = [r["genus"] for r in sub]
        cnt = collections.Counter(g)
        keep = {k for k, c in cnt.items() if c >= 5}
        m = np.array([x in keep for x in g])
        vv, gg = v[m], [x for x, ok in zip(g, m) if ok]
        Hs, p, ngr = perm_kruskal(vv, gg)
        e = eta_sq(vv, gg)
        res["ratios"][name] = dict(n_total=len(sub), n_tested=int(m.sum()), n_genera=ngr,
                                   H=Hs, p_perm=p, eta_sq_rank=e,
                                   median=float(np.median(v)),
                                   q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)),
                                   hl_conditioned=bool(nhl))
        print(f"\n[m4b] {name}: n={len(sub)} ({int(m.sum())} in {ngr} genera with n>=5)")
        print(f"        median {np.median(v):.3f}  IQR [{np.quantile(v,.25):.3f}, {np.quantile(v,.75):.3f}]")
        print(f"        between-genus: H={Hs:.1f}  permutation p={p:.5f}  rank eta^2={e:.3f}")
        # genus medians, extremes
        gm = sorted(((k, float(np.median(vv[[i for i, x in enumerate(gg) if x == k]])), cnt[k])
                     for k in keep), key=lambda t: t[1])
        res["ratios"][name]["genus_medians"] = [dict(genus=k, median=mm, n=c) for k, mm, c in gm]
        print("        lowest :", ", ".join(f"{k}={mm:.3f}" for k, mm, _ in gm[:4]))
        print("        highest:", ", ".join(f"{k}={mm:.3f}" for k, mm, _ in gm[-4:]))

    p = os.path.join(OUT, "m4b_scalefree.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"\n[m4b] wrote {p}")


if __name__ == "__main__":
    main()
