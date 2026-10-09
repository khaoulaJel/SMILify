"""Z13 -- does the recommended configuration work at SUBFAMILY level?

Everything so far has been pushed at GENUS: 83 classes, many holding two or three specimens, on a
corpus whose registration sits at its noise floor. That is a brutal problem, and the honest ceiling
found there was top-1 0.179 / top-5 0.394 -- a shortlist assistant, not an identifier.

Subfamily is far coarser. `REPORT_MORPHOMETRICS.md` reports it only once, at 1.7x lift under 1-NN
on all traits -- i.e. under exactly the configuration Z10-Z12 showed understates the signal by a
third. It has never been tested with traits+betas or a regularised classifier.

WHY IT MATTERS RATHER THAN BEING JUST ANOTHER LEVEL. If subfamily lands somewhere usable, the
deliverable changes category: from "narrows 83 genera to a shortlist" to "makes an automated coarse
identification", which is a different and much stronger claim, and one a specialist can act on
directly. If it does not, the ceiling is confirmed as a property of the measurement rather than of
the class granularity, and the write-up should say so.

PROTOCOL identical to Z12 -- lot-blind leave-one-lot-out, per-model permutation null, traits at
R >= 0.1 plus the shape-space betas. Genus is re-run here on the same specimen subset so the two
levels are compared on the same data rather than across differently-filtered corpora.
"""
import glob
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from sklearn.metrics import recall_score  # noqa: E402

import analyse as AN  # noqa: E402
import measure as ms  # noqa: E402
from taxonomy import SUBFAMILY  # noqa: E402

N_PERM = int(os.environ.get("Z13_NPERM", "30"))
UNCERTAIN = ("sp.", "cf.", "aff.", "nr.")


def evaluate(make_model, F, y, groups, ks=(1, 3)):
    hits = {k: 0 for k in ks}
    total = 0
    pred = np.full(len(y), "?", dtype=object)
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_model()
        m.fit(F[tr], y[tr])
        pred[te] = m.predict(F[te])
        sc = m.predict_proba(F[te])
        ranked = m.classes_[np.argsort(-sc, axis=1)]
        for i, truth in enumerate(y[te]):
            total += 1
            for k in ks:
                if truth in ranked[i, :k]:
                    hits[k] += 1
    ok = pred != "?"
    macro = recall_score(y[ok], pred[ok], average="macro", zero_division=0)
    return {k: hits[k] / total for k in ks}, float(macro), pred


def main():
    runs = sorted((os.path.basename(d) for d in
                   glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*"))
                   if os.path.isfile(os.path.join(d, "Stage_3_deform_fine.npz"))),
                  key=lambda t: int(re.sub(r"\D", "", t) or 0))
    M = ms.load_model()
    rows, cols, dev = AN.build_table([(r, "worker") for r in runs], M, ms.bone_table(M),
                                     ms.template_part_axes(M))
    R = {t["trait"]: t["R"] for t in json.load(open(
        os.path.join(REPO, "diagnostics/full_corpus/out_Z8/z8_repeatability.json")))["traits"]}
    X, fnames = AN.features_with_betas(rows, cols, dev, repeatability=R, r_min=0.1)

    genus = np.array([str(r.get("genus") or str(r["label"]).split("_")[0]) for r in rows])
    spec = np.array([str(r.get("specimen") or r["label"]) for r in rows])
    species = np.array(["_".join(os.path.basename(str(r["label"])).split("_")[:2]) for r in rows])
    lot = np.array([AN.accession_lot(s) for s in spec])
    sub = np.array([SUBFAMILY.get(g, "?") for g in genus])

    unmapped = sorted({g for g, s in zip(genus, sub) if s == "?"})
    print(f"{len(X)} specimens, {X.shape[1]} features (traits R>=0.1 + betas)")
    print(f"genera without a subfamily mapping: {len(unmapped)}"
          + (f"  e.g. {', '.join(unmapped[:6])}" if unmapped else ""))

    # A specimen must have a subfamily AND its class must span >=2 lots to be classifiable
    # lot-blind. Both levels are then restricted to the SAME specimens so the comparison is
    # like-for-like rather than across differently-filtered corpora.
    ok = sub != "?"
    ok &= np.isin(sub, [c for c in set(sub[ok]) if len(set(lot[sub == c])) >= 2])
    ok &= np.isin(genus, [g for g in set(genus[ok]) if len(set(lot[genus == g])) >= 2])
    Xk, gk, sk, lk, spk = X[ok], genus[ok], sub[ok], lot[ok], species[ok]
    print(f"\ncomparable subset: {len(Xk)} specimens, {len(set(sk))} subfamilies, "
          f"{len(set(gk))} genera, {len(set(lk))} lots")
    cnt = Counter(sk).most_common()
    sizes = ", ".join("%s %d" % (k, v) for k, v in cnt[:8])
    print("subfamily sizes: " + sizes + (" ..." if len(cnt) > 8 else ""))

    MODELS = {"LDA shrinkage": AN.lda_shrinkage(), "logistic L2": AN.logistic_l2(1.0)}
    rng = np.random.default_rng(0)
    out = {}
    print("\n" + "=" * 90)
    print("LOT-BLIND ACCURACY BY TAXONOMIC LEVEL -- same specimens, same features")
    print("=" * 90)
    print(f"{'level':<12}{'classes':>8}{'model':<16}{'top-1':>8}{'null':>8}{'lift':>7}"
          f"{'top-3':>8}{'macroR':>9}")
    preds = {}
    for lname, y in [("subfamily", sk), ("genus", gk)]:
        for mname, mk in MODELS.items():
            acc, macro, pred = evaluate(mk, Xk, y, lk)
            null = np.array([evaluate(mk, Xk, rng.permutation(y), lk)[0][1]
                             for _ in range(N_PERM)])
            mu = float(null.mean())
            out[f"{lname}|{mname}"] = {"classes": len(set(y)), "top1": acc[1], "top3": acc[3],
                                       "macro": macro, "null": mu,
                                       "lift": acc[1] / mu if mu else float("nan"),
                                       "p": float((null >= acc[1]).mean())}
            preds[f"{lname}|{mname}"] = pred
            print(f"{lname:<12}{len(set(y)):>8}{mname:<16}{acc[1]:>8.3f}{mu:>8.3f}"
                  f"{acc[1]/mu if mu else np.nan:>7.2f}{acc[3]:>8.3f}{macro:>9.3f}", flush=True)

    # usability at each level: do two nest-mates get the same answer?
    by = defaultdict(list)
    for i, s in enumerate(spk):
        e = s.split("_", 1)[1] if "_" in s else ""
        if e and not e.startswith(UNCERTAIN):
            by[s].append(i)
    prs = [(a, b) for v in by.values() if len(v) >= 2 for a, b in itertools.combinations(v, 2)]
    print("\n" + "=" * 90)
    print(f"USABILITY -- do two nest-mates get the same answer?  ({len(prs)} pairs, lot-blind)")
    print("=" * 90)
    print(f"{'level / model':<30}{'same answer':>13}{'chance':>9}{'lift':>7}{'both correct':>15}")
    for key, pred in preds.items():
        good = [(a, b) for a, b in prs if pred[a] != "?" and pred[b] != "?"]
        y = sk if key.startswith("subfamily") else gk
        agree = float(np.mean([pred[a] == pred[b] for a, b in good]))
        both = float(np.mean([pred[a] == y[a] and pred[b] == y[b] for a, b in good]))
        nl = np.array([np.mean([p[a] == p[b] for a, b in good])
                       for p in (rng.permutation(pred) for _ in range(200))])
        out.setdefault("_usability", {})[key] = {"agree": agree, "both": both,
                                                 "null": float(nl.mean())}
        print(f"{key:<30}{agree*100:>12.1f}%{nl.mean()*100:>8.1f}%"
              f"{agree/nl.mean():>7.2f}{both*100:>14.1f}%")

    od = os.path.join(HERE, "out_Z13")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "z13_results.json"), "w"), indent=2)
    print(f"\nwrote {od}/z13_results.json")


if __name__ == "__main__":
    main()
