"""Z10 -- is 9-11% the classifier's limit or the measurement's?

`analyse.py` chose 1-NN deliberately and says why: "With ~60-150 specimens spread over dozens of
genera, most classifiers cannot be fit honestly. 1-NN with a permutation null is honest at this
sample size." There are now 757 fitted specimens, so that constraint is gone -- Z9's 9-11% is a
FLOOR under a deliberately weak classifier, not a ceiling on the information present.

PROTOCOL, held identical to Z9 so the comparison is like-for-like
* Lot-blind: leave-one-LOT-out, not leave-one-specimen-out. Specimens sharing an accession lot are
  near-replicates from one colony and scan session; `analyse.py` measured that plain LOO inflates
  species accuracy to 95.7% on accession number alone. Non-negotiable.
* The same permutation null, recomputed per model -- a stronger model has a HIGHER chance baseline
  (it can exploit class priors), so comparing a strong model's raw accuracy against 1-NN's null
  would manufacture a win. Every lift below is against that model's own null.
* Feature sets: R >= 0.1 (Z9's best, 45 traits) and all 71, so a feature-set effect and a model
  effect cannot be confused.

WHAT WOULD CHANGE THE CONCLUSION
Materially higher accuracy => the classifier was the limit and the morphometrics are more usable
than Z9 reads. Flat => the measurement is the limit and Z6-Z9 have found this pipeline's ceiling.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import analyse as AN  # noqa: E402
import measure as ms  # noqa: E402

N_PERM = int(os.environ.get("Z10_NPERM", "20"))
# 20 permutations per model. Leave-one-LOT-out refits the model once per lot, so a permutation
# null costs (n_lots x N_PERM) fits per model per feature set. 20 is enough to place the null
# mean (the quantity the lift divides by) while keeping the run tractable; the p-values are
# correspondingly coarse (resolution 1/20) and are reported as such rather than as exact.


def lot_blind_acc(make_model, F, y, lot):
    """Leave-one-LOT-out accuracy. A fresh model per fold; no leakage across lots."""
    correct, total = 0, 0
    for g in np.unique(lot):
        te = lot == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_model()
        m.fit(F[tr], y[tr])
        correct += int((m.predict(F[te]) == y[te]).sum())
        total += int(te.sum())
    return correct / total if total else float("nan")


def main():
    runs = sorted((os.path.basename(d) for d in
                   glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*"))
                   if os.path.isfile(os.path.join(d, "Stage_3_deform_fine.npz"))),
                  key=lambda t: int(re.sub(r"\D", "", t) or 0))
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)
    rows, cols, dev = AN.build_table([(r, "worker") for r in runs], M, bones, tpa)
    F, _, _, _ = AN.features(rows, cols, dev)
    names = list(cols) + list(dev)

    genus = np.array([str(r.get("genus") or str(r["label"]).split("_")[0]) for r in rows])
    spec = np.array([str(r.get("specimen") or r["label"]) for r in rows])
    lot = np.array([AN.accession_lot(s) for s in spec])
    keep = np.isin(genus, [g for g in set(genus) if len(set(lot[genus == g])) >= 2])
    F, genus, lot = F[keep], genus[keep], lot[keep]
    print(f"{len(F)} specimens, {len(set(genus))} genera, {len(set(lot))} lots")

    Rtab = {t["trait"]: t["R"] for t in
            json.load(open(os.path.join(HERE, "out_Z8/z8_repeatability.json")))["traits"]}
    SETS = {"R>=0.1 (Z9 best)": [i for i, n in enumerate(names) if Rtab.get(n, -9) >= 0.1],
            "all traits": list(range(len(names)))}

    MODELS = {
        "1-NN (Z9 baseline)": None,   # handled by analyse.loo_1nn, the exact Z9 protocol
        "LDA (shrinkage)": lambda: make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "logistic L2": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        # Random forest deliberately omitted: with 83 classes, many of 2-3 specimens, and a
        # leave-one-lot-out null, it costs far more than it can inform. LDA and L2 logistic are
        # the right comparators for a small-n, many-class, low-dimensional feature space.
    }

    print("\n" + "=" * 96)
    print("LOT-BLIND GENUS ACCURACY -- each model against ITS OWN permutation null")
    print("=" * 96)
    out = {}
    rng = np.random.default_rng(0)
    for sname, sel in SETS.items():
        Fs = F[:, sel]
        print(f"\n[{sname}]  {len(sel)} traits")
        print(f"{'model':<22}{'accuracy':>11}{'null':>9}{'lift':>8}{'p':>9}")
        for mname, mk in MODELS.items():
            if mk is None:
                a = AN.loo_1nn(Fs, genus, lot)
                null = np.array([AN.loo_1nn(Fs, rng.permutation(genus), lot)
                                 for _ in range(N_PERM)])
            else:
                a = lot_blind_acc(mk, Fs, genus, lot)
                null = np.array([lot_blind_acc(mk, Fs, rng.permutation(genus), lot)
                                 for _ in range(N_PERM)])
            mu = float(null.mean())
            p = float((null >= a).mean())
            out[f"{sname}|{mname}"] = {"acc": a, "null": mu,
                                       "lift": a / mu if mu else float("nan"), "p": p}
            print(f"{mname:<22}{a:>11.3f}{mu:>9.3f}{a/mu if mu else float('nan'):>8.2f}"
                  f"{p:>9.3f}")

    best = max(out.items(), key=lambda kv: kv[1]["acc"])
    base = out["R>=0.1 (Z9 best)|1-NN (Z9 baseline)"]
    print("\n" + "=" * 96)
    print(f"Z9 baseline (1-NN, R>=0.1) : accuracy {base['acc']:.3f}  lift {base['lift']:.2f}x")
    print(f"best model here            : {best[0]}  accuracy {best[1]['acc']:.3f}  "
          f"lift {best[1]['lift']:.2f}x")
    gain = best[1]["acc"] - base["acc"]
    print(f"absolute accuracy gain     : {gain:+.3f}")
    print()
    if gain >= 0.05:
        print("=> THE CLASSIFIER WAS THE LIMIT. 1-NN understated the information present;")
        print("   the morphometrics are more usable than Z9 reads.")
    else:
        print("=> THE MEASUREMENT IS THE LIMIT. A stronger classifier does not recover more,")
        print("   so Z6-Z9 have found this pipeline's ceiling, not 1-NN's.")
    print("=" * 96)

    od = os.path.join(HERE, "out_Z10")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "z10_results.json"), "w"), indent=2)
    print(f"wrote {od}/z10_results.json")


if __name__ == "__main__":
    main()
