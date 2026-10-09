"""v4_annotation_targets.py -- pick WHICH specimens to annotate, stratified by severity.

The theta/beta factorial and every anatomically-supervised experiment are blocked on the same
fact: only 1 of the 69 pathological specimens is annotated. This selects a 15-specimen request.

Stratified rather than top-N, because V3 showed severity is a smooth continuum (68% within 25% of
the bound, none beyond 2x). Annotating only the worst cases would answer "can supervision rescue
catastrophes" when the question is "does it fix the continuum". Healthy controls are included so
an intervention can be shown NOT to damage fits that were already fine.

CONTROL: the flag set derived here is asserted identical to V2's before anything is selected.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOUNDS = {"HW/HL": (0.40, 2.00), "SL/HL": (0.20, 3.00), "PetL/WL": (0.02, 0.80),
          "ML/HL": (0.10, 2.00), "GL/WL": (0.50, 4.00)}
PICK = {"A_just_over": 3, "B_medium": 4, "C_extreme": 3}
N_HEALTHY = 5
SEED = 20260907


def main():
    tz = np.load(os.path.join(HERE, "traits_Z8.npz"))
    labs = [str(x) for x in tz["labels"]]
    T = {k: np.asarray(tz[k]) for k in tz.files if k != "labels"}
    R = {k: T[k.split("/")[0]] / np.maximum(T[k.split("/")[1]], 1e-12) for k in BOUNDS}

    n = len(labs)
    sev = np.zeros(n)
    which = [""] * n
    for k, (lo, hi) in BOUNDS.items():
        o = np.maximum(np.maximum(lo - R[k], R[k] - hi), 0) / (hi - lo)
        for i in np.where(o > sev)[0]:
            which[i] = k
        sev = np.maximum(sev, o)
    fl = sev > 0

    v2 = json.load(open(os.path.join(HERE, "v2_results.json")))
    assert set(np.array(labs)[fl]) == set(v2["flagged_labels_with"]), \
        "flag set does not reproduce V2 -- selection would not be of the studied population"
    print(f"CONTROL: flag set reproduces V2 exactly ({fl.sum()} specimens)\n")

    idx = np.where(fl)[0]
    s = sev[idx]
    order = idx[np.argsort(s)]
    lo_b, hi_b = np.percentile(s, [33, 67])
    strata = {"A_just_over": [i for i in order if sev[i] <= lo_b],
              "B_medium": [i for i in order if lo_b < sev[i] <= hi_b],
              "C_extreme": [i for i in order if sev[i] > hi_b]}

    rng = np.random.default_rng(SEED)
    out = {}
    for k, cnt in PICK.items():
        c = np.array(strata[k])
        sel = rng.choice(c, size=min(cnt, len(c)), replace=False)
        out[k] = [{"specimen": labs[i], "severity": round(float(sev[i]), 3),
                   "worst_ratio": which[i], "value": round(float(R[which[i]][i]), 3)}
                  for i in sorted(sel, key=lambda j: sev[j])]
    hs = rng.choice(np.where(~fl)[0], size=N_HEALTHY, replace=False)
    out["D_healthy_controls"] = [{"specimen": labs[i], "severity": 0.0,
                                  "worst_ratio": "-", "value": None} for i in hs]

    for k, v in out.items():
        print(f"--- {k} ({len(v)})")
        for e in v:
            print(f"    {e['specimen']:52s} sev={e['severity']:.3f}  "
                  f"{e['worst_ratio']} = {e['value']}")

    json.dump({"provenance": "v4_annotation_targets.py; severity = max fractional overshoot past "
                             "T1 bounds; flag set verified identical to V2",
               "n_flagged": int(fl.sum()),
               "strata_bounds": {"p33": float(lo_b), "p67": float(hi_b)},
               "targets": out},
              open(os.path.join(HERE, "v4_annotation_targets.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
