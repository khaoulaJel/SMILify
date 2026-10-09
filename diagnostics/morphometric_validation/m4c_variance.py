"""M4-C -- hierarchical variance decomposition of validated head shape.

Per PREREGISTRATION_M4_corpus_analysis.md §3. Variance decomposition is the PRIMARY analysis;
classification is secondary and is not run here.

Question: how much of the validated head-shape variation is associated with subfamily, genus and
species, and how much remains within those levels? Specifically -- is the 0.26-0.33 genus-level
rank signal from M4-B genuinely concentrated at GENUS, or largely a consequence of SPECIES
composition within genera?

Runs on the FROZEN M4-A admissible set. Ratios are used as-is; no size correction is attempted
(M4-B limitation 1 stands, M4-D is deferred pending a resolved scale reference).
"""
import json, os, collections, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "diagnostics/morphometric_validation/out")


def decompose(df, levels, label):
    """REML nested random-effects decomposition. `levels` is outermost -> innermost.
    Returns % of total variance at each level plus residual."""
    d = df.copy()
    d["grp"] = 1
    vc = {}
    # build nested keys so 'species' means species-within-genus, etc.
    for i, lv in enumerate(levels):
        key = "_".join(levels[: i + 1])
        d[key] = d[levels[: i + 1]].astype(str).agg("|".join, axis=1)
        vc[key] = f"0 + C({key})"
    m = smf.mixedlm("y ~ 1", d, groups="grp", vc_formula=vc).fit(reml=True)
    comps = {k: float(m.vcomp[i]) for i, k in enumerate(vc.keys())}
    comps["residual"] = float(m.scale)
    tot = sum(comps.values())
    pct = {k: 100.0 * v / tot for k, v in comps.items()}
    print(f"\n[m4c] {label}   n={len(d)}")
    for k, v in pct.items():
        nm = k.split("_")[-1] if k != "residual" else "residual (within-species + measurement)"
        print(f"        {nm:42s} {v:6.2f}%")
    return dict(n=int(len(d)), variance_pct=pct, variance_raw=comps, converged=bool(m.converged))


def main():
    d = json.load(open(os.path.join(OUT, "m4a_admissible_set.json")))
    rows = d["per_specimen"]
    for r in rows:
        r["species"] = "_".join(r["specimen"].split("_")[:2])
        r["HW_HL"] = r["HW"] / r["HL"]
    df = pd.DataFrame(rows)

    res = {"preregistration": "PREREGISTRATION_M4_corpus_analysis.md §3",
           "note": "variance decomposition is primary; classification NOT run",
           "models": {}}

    specs = [("HW/WL", "HW_WL", "HW_admissible", "primary"),
             ("HL/WL", "HL_WL", "HL_admissible", "secondary -- carries the genus-structured M4-A exclusion")]

    for name, col, gate, tier in specs:
        sub = df[df[gate]].copy()
        sub["y"] = sub[col]
        print(f"\n{'='*72}\n[m4c] {name}  ({tier})")
        # Model A: genus / species, on ALL admissible specimens
        a = decompose(sub, ["genus", "species"], "A: genus / species  [all admissible]")
        # Model B: subfamily / genus / species, subfamily-mapped subset only
        mapped = sub[sub["subfamily"] != "UNKNOWN"].copy()
        b = decompose(mapped, ["subfamily", "genus", "species"],
                      "B: subfamily / genus / species  [mapped subset only]")
        res["models"][name] = dict(tier=tier, model_A=a, model_B=b,
                                   mapped_coverage=float(len(mapped) / len(sub)))
        print(f"        [model B covers {len(mapped)}/{len(sub)} = {100*len(mapped)/len(sub):.0f}% "
              f"-- subfamily result is coverage-limited, not an absence of effect]")

    p = os.path.join(OUT, "m4c_variance.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"\n[m4c] wrote {p}")


if __name__ == "__main__":
    main()
