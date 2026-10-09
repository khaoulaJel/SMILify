"""t3_recipe.py -- convert T1/T2 findings into a measurement recipe.

Design and ship rule fixed in PREREGISTRATION_T3_recipe.md. This file only implements it.
"""
import sys, os, re, json, collections
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
for _d in ("morphometrics", "groundtruth"):
    sys.path.insert(0, os.path.join(REPO, "diagnostics", _d))

MIN_PER_GENUS = 8
MIN_GENERA = 15
MIN_RETAIN = 0.60
NBOOT = 2000
RATIOS = ("HW/HL", "SL/HL", "ML/HL", "PetL/WL")
BOUNDS = {"HW/HL": (0.40, 2.00), "SL/HL": (0.20, 3.00),
          "ML/HL": (0.10, 2.00), "PetL/WL": (0.02, 0.80)}
CONTRASTS = [("C1", "HW/HL", "cephalotes", "odontomachus"),
             ("C2", "HW/HL", "pheidole", "odontomachus"),
             ("C3", "SL/HL", "camponotus", "cephalotes"),
             ("C4", "SL/HL", "formica", "cephalotes")]


def ratios(T, bilateral):
    """bilateral=True -> mean of left and right for the traits that have both sides."""
    def s(n):
        if bilateral and (n + "_l") in T:
            return (T[n] + T[n + "_l"]) / 2.0
        return T[n]
    return {"HW/HL": T["HW"] / T["HL"], "SL/HL": s("SL") / T["HL"],
            "ML/HL": s("ML") / T["HL"], "PetL/WL": T["PetL"] / T["WL"]}


def f_ratio(vals, gen):
    """Between-genus MS / within-genus MS. Compression cannot inflate this: it shrinks both."""
    gs = [g for g, c in collections.Counter(gen).items() if c >= MIN_PER_GENUS]
    if len(gs) < 2:
        return np.nan, 0
    groups = [vals[gen == g] for g in gs]
    grand = np.concatenate(groups).mean()
    k = len(groups)
    n = sum(len(x) for x in groups)
    ssb = sum(len(x) * (x.mean() - grand) ** 2 for x in groups)
    ssw = sum(((x - x.mean()) ** 2).sum() for x in groups)
    if ssw <= 0 or n - k <= 0:
        return np.nan, len(gs)
    return (ssb / (k - 1)) / (ssw / (n - k)), len(gs)


def score(R, gen, keep):
    """Mean F across trait ratios, plus a bootstrap-over-genera distribution."""
    gsub = gen[keep]
    per, ngen = {}, 0
    for r in RATIOS:
        f, ng = f_ratio(R[r][keep], gsub)
        per[r] = f
        ngen = max(ngen, ng)
    return float(np.nanmean(list(per.values()))), per, ngen


def boot(R, gen, keep, rng):
    """Resample GENERA (not specimens): the unit of independence is the genus."""
    gsub = gen[keep]
    gs = np.array([g for g, c in collections.Counter(gsub).items() if c >= MIN_PER_GENUS])
    out = np.empty(NBOOT)
    idx_by_g = {g: np.where(gsub == g)[0] for g in gs}
    sub = {r: R[r][keep] for r in RATIOS}
    for b in range(NBOOT):
        pick = rng.choice(len(gs), len(gs), replace=True)
        sel = np.concatenate([idx_by_g[gs[i]] for i in pick])
        lab = np.concatenate([np.full(len(idx_by_g[gs[i]]), f"{i}_{j}")
                              for j, i in enumerate(pick)])
        vals = [f_ratio(sub[r][sel], lab)[0] for r in RATIOS]
        out[b] = np.nanmean(vals)
    return out


def main():
    t = np.load(os.path.join(REPO, "diagnostics", "groundtruth", "traits_Z8.npz"))
    labs = [str(x) for x in t["labels"]]
    T = {k: t[k] for k in t.files if k != "labels"}
    gen = np.array([re.match(r"([A-Za-z]+)_", l).group(1).lower() for l in labs])
    n = len(labs)

    reg = json.load(open(os.path.join(REPO, "diagnostics", "morphometrics",
                                      "out", "registration_error.json")))
    key = {re.sub(r"_processed\.obj$", "", k): v for k, v in reg.items()}
    err = np.array([key.get(re.sub(r"_processed\.obj$", "", l), np.nan) for l in labs])

    R_plain, R_bil = ratios(T, False), ratios(T, True)

    def plausible(R):
        k = np.ones(n, bool)
        for r, (lo, hi) in BOUNDS.items():
            k &= (R[r] >= lo) & (R[r] <= hi)
        return k

    q_keep = err <= np.nanpercentile(err, 90)          # drop the worst chamfer decile

    ARMS = [
        ("baseline",  R_plain, np.ones(n, bool)),
        ("A bilateral", R_bil, np.ones(n, bool)),
        ("B plausible", R_plain, plausible(R_plain)),
        ("C chamfer", R_plain, q_keep),
        ("A+B",       R_bil,   plausible(R_bil)),
        ("A+B+C",     R_bil,   plausible(R_bil) & q_keep),
    ]

    rng = np.random.default_rng(0)
    base_boot = None
    print(f"{'arm':14s} {'kept':>12s} {'genera':>7s} {'mean F':>8s} {'vs base':>18s}")
    results = {}
    for name, R, keep in ARMS:
        m, per, ngen = score(R, gen, keep)
        bs = boot(R, gen, keep, np.random.default_rng(0))
        if base_boot is None:
            base_boot, base_m = bs, m
            delta = "-"
        else:
            d = bs - base_boot
            lo, hi = np.percentile(d, [2.5, 97.5])
            ships = lo > 0
            delta = f"{m-base_m:+.2f} [{lo:+.2f},{hi:+.2f}]{' SHIP' if ships else ''}"
        retain = keep.mean()
        print(f"{name:14s} {keep.sum():5d} ({100*retain:4.1f}%) {ngen:7d} {m:8.2f} {delta:>18s}")
        results[name] = dict(mean_F=m, per_trait=per, n_kept=int(keep.sum()),
                             retain=float(retain), n_genera=ngen)
        if ngen < MIN_GENERA:
            print(f"    VOID: only {ngen} genera survive (min {MIN_GENERA})")
        if retain < MIN_RETAIN:
            print(f"    reported but NOT shippable: retains {100*retain:.1f}% (< {100*MIN_RETAIN:.0f}%)")

    print("\nper-trait F-ratio")
    print(f"{'arm':14s} " + " ".join(f"{r:>9s}" for r in RATIOS))
    for name in results:
        print(f"{name:14s} " + " ".join(f"{results[name]['per_trait'][r]:9.2f}" for r in RATIOS))

    print("\nsecondary check -- pre-specified taxonomic contrasts (direction only)")
    for name, R, keep in ARMS:
        out = []
        for cid, r, a, b in CONTRASTS:
            va, vb = R[r][keep & (gen == a)], R[r][keep & (gen == b)]
            out.append(f"{cid}:{'OK ' if (len(va) and len(vb) and np.median(va) > np.median(vb)) else 'REV'}")
        print(f"  {name:14s} " + "  ".join(out))

    json.dump(results, open(os.path.join(REPO, "diagnostics", "groundtruth",
                                         "t3_results.json"), "w"), indent=1)
    print("\nwrote t3_results.json")


if __name__ == "__main__":
    main()
