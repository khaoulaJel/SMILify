"""Z8 -- trait repeatability on the FULL 757-worker corpus, production recipe.

Z7 measured R = 1 - var_within-pair / var_total from 25 conspecific pairs and found 8 usable
traits of 71 (median R 0.217), with leg and antenna segment lengths at or below zero. That is a
trustworthy RANKING but a coarse estimate of any individual R -- 25 pairs gives roughly +-0.2 on a
single value. The full corpus has 191 species with >=2 determined specimens, so each R is
estimated from an order of magnitude more replicate pairs.

WHY THIS RECIPE. These fits come from run_m1_fit_all.sh, the production chain the morphometrics
report is built on, not the bare optimise_moonshot calls the Z2-Z6 diagnostics used. Repeatability
has to be measured on the pipeline whose outputs the report actually consumes.

DESIGN POINTS CARRIED FROM Z6/Z7
* Determined species only -- `sp.`/`cf.`/`aff.`/`nr.` are uncertain determinations, and two
  `Camponotus_sp.` specimens are not conspecifics.
* Species with >2 specimens contribute ALL within-species pairs, so a variance component is
  estimated per species rather than a single difference; species are then weighted equally so a
  species with 8 specimens does not dominate.
* Bilateral consistency is reported as an internal check: a trait whose left and right versions
  disagree in R is unreliable regardless of the better side (Z7: mandible_r 0.613 vs _l 0.254).
"""
import glob
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as MZ  # noqa: E402

UNCERTAIN = ("sp.", "cf.", "aff.", "nr.")


def species_of(label):
    return "_".join(os.path.basename(str(label)).split("_")[:2])


def determined(sp):
    ep = sp.split("_", 1)[1] if "_" in sp else ""
    return bool(ep) and not ep.startswith(UNCERTAIN)


def main():
    runs = sorted(
        (os.path.basename(d) for d in glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*"))
         if os.path.isfile(os.path.join(d, "Stage_3_deform_fine.npz"))),
        key=lambda t: int(re.sub(r"\D", "", t) or 0))
    if not runs:
        raise SystemExit("no Z8_W* runs with fits found")
    print(f"aggregating {len(runs)} chunk runs: {', '.join(runs)}")

    rows, cols, devcols = [], None, None
    for r in runs:
        rr, cc, dd_ = MZ.measure_run(r, corpus="worker")
        rows += rr
        cols = cols or cc
        devcols = devcols or dd_
    print(f"measured {len(rows)} specimens, {len(cols)} length traits, "
          f"{len(devcols)} shape-deviation traits")

    X, _ = MZ.log_shape_ratios(
        np.array([[r[c] for c in cols] for r in rows], dtype=np.float64))
    D = np.array([[r[c] for c in devcols] for r in rows], dtype=np.float64) if devcols \
        else np.zeros((len(rows), 0))
    allX = np.hstack([X, D])
    names = list(cols) + list(devcols)

    by_sp = defaultdict(list)
    for i, r in enumerate(rows):
        sp = species_of(r["label"])
        if determined(sp):
            by_sp[sp].append(i)
    reps = {s: v for s, v in by_sp.items() if len(v) >= 2}
    n_pairs = sum(len(list(itertools.combinations(v, 2))) for v in reps.values())
    print(f"replicate species: {len(reps)}  specimens in them: {sum(len(v) for v in reps.values())}"
          f"  within-species pairs: {n_pairs}")

    res = []
    for k, nm in enumerate(names):
        v = allX[:, k]
        if not np.all(np.isfinite(v)) or v.std() == 0:
            continue
        # equal weight per species: mean of that species' pairwise squared differences / 2
        per_sp = []
        for s, idxs in reps.items():
            d2 = [(v[a] - v[b]) ** 2 for a, b in itertools.combinations(idxs, 2)]
            per_sp.append(np.mean(d2) / 2.0)
        s2_within = float(np.mean(per_sp))
        s2_total = float(v.var(ddof=1))
        R = 1.0 - s2_within / s2_total if s2_total > 0 else np.nan
        res.append({"trait": nm, "R": R, "within_sd": float(np.sqrt(s2_within)),
                    "total_sd": float(np.sqrt(s2_total))})

    res.sort(key=lambda r: -r["R"])
    print("\n" + "=" * 92)
    print(f"TRAIT REPEATABILITY -- FULL CORPUS ({len(rows)} workers, {len(reps)} replicate species,"
          f" {n_pairs} pairs)")
    print("R = 1 - within-species var / total var.  R<=0: cannot distinguish a conspecific from a")
    print("random ant, i.e. the trait is measuring noise.")
    print("=" * 92)
    print(f"{'trait':<34}{'R':>9}{'within sd':>12}{'total sd':>11}")
    for r in res:
        flag = "" if r["R"] >= 0.5 else ("  <- weak" if r["R"] >= 0.2 else "  <- NOISE")
        print(f"{r['trait']:<34}{r['R']:>9.3f}{r['within_sd']:>12.5f}{r['total_sd']:>11.5f}{flag}")

    Rs = np.array([r["R"] for r in res])
    usable = [r["trait"] for r in res if r["R"] >= 0.5]
    print("\n" + "=" * 92)
    print(f"traits            : {len(res)}")
    print(f"  usable  R>=0.5  : {int((Rs >= 0.5).sum())}")
    print(f"  weak  0.2-0.5   : {int(((Rs >= 0.2) & (Rs < 0.5)).sum())}")
    print(f"  noise   R<0.2   : {int((Rs < 0.2).sum())}   ({int((Rs < 0).sum())} negative)")
    print(f"  median R        : {np.median(Rs):.3f}")
    print("=" * 92)

    # internal check: bilateral pairs must agree in R
    Rmap = {r["trait"]: r["R"] for r in res}
    print("\nBILATERAL CONSISTENCY CHECK (|R_left - R_right|; a large gap means distrust both)")
    print(f"{'trait':<28}{'R right':>10}{'R left':>10}{'|gap|':>9}")
    gaps = []
    for t in sorted(Rmap):
        if t.endswith("_r") and t[:-2] + "_l" in Rmap:
            a, b = Rmap[t], Rmap[t[:-2] + "_l"]
            gaps.append((abs(a - b), t, a, b))
    for g, t, a, b in sorted(gaps, reverse=True)[:12]:
        print(f"{t[:-2]:<28}{a:>10.3f}{b:>10.3f}{g:>9.3f}{'   <- distrust' if g > 0.2 else ''}")

    print(f"\nZ7 (25 pairs) found 8 usable traits, median R 0.217. Full corpus: "
          f"{len(usable)} usable, median R {np.median(Rs):.3f}.")
    od = os.path.join(HERE, "out_Z8")
    os.makedirs(od, exist_ok=True)
    json.dump({"n_specimens": len(rows), "n_species": len(reps), "n_pairs": n_pairs,
               "usable": usable, "traits": res},
              open(os.path.join(od, "z8_repeatability.json"), "w"), indent=2)
    print(f"wrote {od}/z8_repeatability.json")


if __name__ == "__main__":
    main()
