"""M4-A -- the QC gate. Per PREREGISTRATION_M4_corpus_analysis.md §1.

Establishes WHICH specimens and WHICH measurements are admissible, tests whether failure is
taxonomically structured, and FREEZES the admissible set. Runs no biology.
"""
import json, os, sys, collections
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, "diagnostics/groundtruth"),
                os.path.join(REPO, "diagnostics/morphometrics"),
                os.path.join(REPO, "diagnostics/morphometric_validation")]
import trait_extract as TX                              # noqa: E402
from measure import load_model, kabsch, body_frame      # noqa: E402
from mv_framework import forward_pair, recal_landmarks, TEMPL_TO_RECAL  # noqa: E402
from taxonomy import GENUS_TO_SUBFAMILY                 # noqa: E402
from m3b_extraction_audit import head_lateral_halves, ap_order  # noqa: E402

import glob
OUT = os.path.join(REPO, "diagnostics/morphometric_validation/out")


def main():
    os.makedirs(OUT, exist_ok=True)
    M = load_model()
    lm = dict(TX.load_landmarks(M))
    raw = recal_landmarks(M)
    for t, r in TEMPL_TO_RECAL.items():
        if r in raw:
            lm[t] = raw[r]

    V, labels = [], []
    for f in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/M4_W*/Stage_3_deform_fine.npz"))):
        o, _c, lb = forward_pair(f)
        V.append(o); labels += lb
    V = np.concatenate(V, 0)
    n = len(labels)
    print(f"[m4a] {n} specimens, verts {V.shape}")

    tr = TX.traits(V, M=M, lm=lm)
    HW, HL, WL = tr["HW"], tr["HL"], tr["WL"]
    hw_wl, hl_wl = HW / WL, HL / WL
    hi, lo = head_lateral_halves(V, M)
    lat = hi / np.maximum(lo, 1e-12)
    chain, proj, ap_ok = ap_order(V, M, lm)

    finite = np.isfinite(HW) & np.isfinite(HL) & np.isfinite(WL) & (HW > 0) & (HL > 0) & (WL > 0)
    # HW gate: same bars as M3b C2/C4
    hw_ok = finite & (hw_wl > 0.15) & (hw_wl < 2.0) & (lat > 0.5) & (lat < 2.0)
    # HL gate: HL additionally requires the AP ordering that defines it
    hl_ratio_ok = finite & (hl_wl > 0.15) & (hl_wl < 2.0)
    hl_ok = hl_ratio_ok & ap_ok

    cls = []
    for i in range(n):
        if not finite[i]:                 cls.append("extraction_failure")
        elif not hl_ratio_ok[i]:          cls.append("extraction_failure")
        elif not ap_ok[i]:                cls.append("AP_inversion")
        else:                             cls.append("valid")
    genus = [l.split("_")[0] for l in labels]
    subfam = [GENUS_TO_SUBFAMILY.get(g.lower(), "UNKNOWN") for g in genus]  # map keys are lowercase

    print(f"\n[m4a] HW admissible: {hw_ok.sum()}/{n} ({100*hw_ok.mean():.1f}%)")
    print(f"[m4a] HL admissible: {hl_ok.sum()}/{n} ({100*hl_ok.mean():.1f}%)")
    print(f"[m4a] HL failure classes: {dict(collections.Counter(cls))}")

    # ---- failure structure: is AP inversion random w.r.t. taxonomy? (§1, pre-registered)
    from scipy import stats
    inv = np.array([c == "AP_inversion" for c in cls])
    struct = {}
    for lvl, keys in (("genus", genus), ("subfamily", subfam)):
        tab = collections.defaultdict(lambda: [0, 0])
        for k, b in zip(keys, inv):
            tab[k][int(b)] += 1
        # chi-square over groups with n>=5, else the test is meaningless
        big = {k: v for k, v in tab.items() if sum(v) >= 5}
        obs = np.array([[v[1], v[0]] for v in big.values()])
        if len(obs) >= 2 and obs[:, 0].sum() > 0:
            # chi2's asymptotics are NOT valid here (few inversions spread over many small groups,
            # most expected counts << 5). Use a permutation test on the same statistic instead:
            # shuffle the inversion labels across the tested specimens and rebuild the table.
            chi2 = stats.chi2_contingency(obs)[0]
            memb = np.concatenate([[gi] * sum(v) for gi, v in enumerate(big.values())])
            lab = np.concatenate([[1] * v[1] + [0] * v[0] for v in big.values()])
            rng = np.random.default_rng(20260910)
            null = np.empty(20000)
            for b in range(20000):
                sh = rng.permutation(lab)
                t = np.zeros((len(big), 2))
                for gi, y in zip(memb, sh):
                    t[gi, 1 - y] += 1
                null[b] = stats.chi2_contingency(t)[0] if t[:, 0].sum() > 0 else 0.0
            pval = float((null >= chi2).mean())
            struct[lvl] = dict(n_groups=len(obs), chi2=float(chi2), p=pval,
                               test="permutation (20000), chi2 statistic",
                               n_inversions_tested=int(obs[:, 0].sum()),
                               note="chi2 asymptotic p invalid here; permutation used")
            worst = sorted(big.items(), key=lambda kv: -kv[1][1] / max(sum(kv[1]), 1))[:8]
            struct[lvl]["most_affected"] = [
                dict(group=k, inverted=v[1], total=sum(v), rate=v[1] / sum(v)) for k, v in worst if v[1] > 0]
        else:
            struct[lvl] = dict(note="insufficient inversions or groups for a test")
        print(f"\n[m4a] AP-inversion structure by {lvl}: {struct[lvl].get('p', 'n/a')}"
              f" (groups n>=5: {struct[lvl].get('n_groups','n/a')})")
        for e in struct[lvl].get("most_affected", [])[:8]:
            print(f"        {e['group']:22s} {e['inverted']:3d}/{e['total']:3d}  {100*e['rate']:5.1f}%")

    rows = [dict(specimen=labels[i], genus=genus[i], subfamily=subfam[i],
                 HW=float(HW[i]), HL=float(HL[i]), WL=float(WL[i]),
                 HW_WL=float(hw_wl[i]), HL_WL=float(hl_wl[i]), head_lat_ratio=float(lat[i]),
                 HW_admissible=bool(hw_ok[i]), HL_admissible=bool(hl_ok[i]), HL_class=cls[i])
            for i in range(n)]
    res = dict(n=n, preregistration="PREREGISTRATION_M4_corpus_analysis.md §1",
               fits="diagnostics/moonshot/runs/M4_W*/Stage_3_deform_fine.npz",
               HW_admissible=int(hw_ok.sum()), HL_admissible=int(hl_ok.sum()),
               HL_classes=dict(collections.Counter(cls)),
               failure_structure=struct,
               distributions={k: dict(median=float(np.median(v[m])), q25=float(np.quantile(v[m], .25)),
                                      q75=float(np.quantile(v[m], .75)), q05=float(np.quantile(v[m], .05)),
                                      q95=float(np.quantile(v[m], .95)))
                              for k, v, m in (("HW_WL", hw_wl, hw_ok), ("HL_WL", hl_wl, hl_ok),
                                              ("head_lat_ratio", lat, hw_ok))},
               per_specimen=rows)
    p = os.path.join(OUT, "m4a_admissible_set.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"\n[m4a] distributions (admissible only):")
    for k, d in res["distributions"].items():
        print(f"        {k:15s} median={d['median']:.3f}  IQR[{d['q25']:.3f},{d['q75']:.3f}]"
              f"  5-95%[{d['q05']:.3f},{d['q95']:.3f}]")
    print(f"[m4a] FROZEN admissible set -> {p}")


if __name__ == "__main__":
    main()
