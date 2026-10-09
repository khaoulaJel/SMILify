"""trait_plausibility.py -- how many fitted specimens produce biologically impossible traits,
and can that be detected automatically without any ground truth?

WHY THIS IS WORTH RUNNING. Bilateral asymmetry (trait_extract.asymmetry) says a fit is NOISY.
Biological impossibility says a fit is WRONG, which is strictly stronger, and neither needs an
annotator. If a meaningful fraction of the corpus emits impossible traits, that alone could sink
correlations across the whole Z-series, and it converts directly into the practitioner's
"what scan quality do I need" table.

PRE-REGISTERED BOUNDS. Set BEFORE looking at any outcome, and set DELIBERATELY WIDE: these are not
tuned thresholds, they are values no ant has. The point is to catch catastrophic failure, not to
adjudicate borderline morphology. Sources are the standard index definitions on AntWiki plus the
observed family-wide envelope; each bound is at least ~25% outside the most extreme real taxon.

    HW/HL   cephalic index. Real ants span roughly 0.6-1.6 (Cephalotes near the broad extreme,
            Odontomachus near the narrow one). IMPOSSIBLE outside [0.40, 2.00].
    SL/HL   scape index. Roughly 0.4-2.0 across the family. IMPOSSIBLE outside [0.20, 3.00].
    PetL/WL petiole is a small fraction of the mesosoma. IMPOSSIBLE outside [0.02, 0.80].
    ML/HL   mandible length vs head length. IMPOSSIBLE outside [0.10, 2.00].
    GL/WL   gaster vs mesosoma. IMPOSSIBLE outside [0.50, 4.00].

THE VALIDATION THAT MAKES THIS NON-CIRCULAR. A flag derived from the traits could just be flagging
its own noise. So it is cross-checked against TWO signals computed independently of the trait layer:
`registration_error.json` (the fit's own chamfer residual, 757/757 coverage) and bilateral
asymmetry. If flagged specimens are also the independently-bad fits, the flag detects real failure.
"""
import sys, os, glob, json, re
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
for _d in ("morphometrics", "groundtruth"):
    sys.path.insert(0, os.path.join(REPO, "diagnostics", _d))

BOUNDS = {
    "HW/HL":   (0.40, 2.00),
    "SL/HL":   (0.20, 3.00),
    "PetL/WL": (0.02, 0.80),
    "ML/HL":   (0.10, 2.00),
    "GL/WL":   (0.50, 4.00),
}


def main():
    d = np.load(os.path.join(REPO, "diagnostics", "groundtruth", "traits_Z8.npz"))
    labs = [str(x) for x in d["labels"]]
    T = {k: d[k] for k in d.files if k != "labels"}
    R = {
        "HW/HL": T["HW"] / T["HL"], "SL/HL": T["SL"] / T["HL"],
        "PetL/WL": T["PetL"] / T["WL"], "ML/HL": T["ML"] / T["HL"],
        "GL/WL": T["GL"] / T["WL"],
    }
    n = len(labs)

    print(f"{n} fitted specimens\n")
    print("=== per-ratio impossibility (pre-registered bounds) ===")
    print(f"{'ratio':9s} {'bounds':>14s} {'median':>8s} {'min':>8s} {'max':>8s} {'flagged':>9s}")
    flag = np.zeros(n, bool)
    for k, (lo, hi) in BOUNDS.items():
        v = R[k]
        bad = (v < lo) | (v > hi)
        flag |= bad
        print(f"{k:9s} [{lo:.2f},{hi:.2f}]".ljust(25)
              + f"{np.median(v):8.3f} {v.min():8.3f} {v.max():8.3f} "
                f"{bad.sum():5d} ({100*bad.mean():4.1f}%)")
    print(f"\nANY impossible trait: {flag.sum()} / {n}  ({100*flag.mean():.1f}%)\n")

    # ---- independent signal 1: the fit's own registration error -----------
    reg = json.load(open(os.path.join(REPO, "diagnostics", "morphometrics",
                                      "out", "registration_error.json")))
    key = {re.sub(r"_processed\.obj$", "", k): v for k, v in reg.items()}
    e = np.array([key.get(re.sub(r"_processed\.obj$", "", l), np.nan) for l in labs])
    have = ~np.isnan(e)
    print(f"=== cross-check 1: registration error (independent of the trait layer) ===")
    print(f"matched {have.sum()}/{n} specimens")
    if have.sum() > 10:
        a, b = e[have & flag], e[have & ~flag]
        print(f"  flagged   median reg-err {np.median(a):.5f}  (n={len(a)})")
        print(f"  unflagged median reg-err {np.median(b):.5f}  (n={len(b)})")
        print(f"  ratio {np.median(a)/max(np.median(b),1e-12):.2f}x")
        pct = 100 * (e[have] <= np.median(a)).mean()
        print(f"  flagged specimens sit at the {pct:.0f}th percentile of fit error")
        # rank-biserial: P(flagged worse than unflagged)
        import itertools
        rs = np.array([(x > b).mean() for x in a])
        print(f"  P(a flagged fit is worse than a random unflagged fit) = {rs.mean():.3f}"
              f"   [0.5 = no signal]")

    # ---- independent signal 2: bilateral asymmetry ------------------------
    print(f"\n=== cross-check 2: bilateral asymmetry (independent of the bounds) ===")
    for t in ("ML", "SL", "WL", "FL"):
        if t + "_l" not in T:
            continue
        asym = np.abs(T[t] - T[t + "_l"]) / np.maximum((T[t] + T[t + "_l"]) / 2, 1e-12)
        print(f"  {t:3s} asymmetry  flagged {100*np.median(asym[flag]):6.1f}%   "
              f"unflagged {100*np.median(asym[~flag]):6.1f}%   "
              f"ratio {np.median(asym[flag])/max(np.median(asym[~flag]),1e-12):5.2f}x")

    # ---- what the filter buys ---------------------------------------------
    print(f"\n=== effect of discarding the flagged fits ===")
    for k in BOUNDS:
        v = R[k]
        print(f"  {k:9s} CV {100*v.std()/v.mean():6.1f}%  ->  "
              f"{100*v[~flag].std()/v[~flag].mean():6.1f}%   "
              f"range [{v.min():.2f},{v.max():.2f}] -> [{v[~flag].min():.2f},{v[~flag].max():.2f}]")

    out = os.path.join(REPO, "diagnostics", "groundtruth", "trait_plausibility.json")
    json.dump(dict(n=n, bounds=BOUNDS, n_flagged=int(flag.sum()),
                   frac_flagged=float(flag.mean()),
                   flagged=[labs[i] for i in np.where(flag)[0]]), open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
