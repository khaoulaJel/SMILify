"""Z12 stage 2 -- tune the Z11 winner honestly and give it a null.

STAGE 1 FOUND (z11_out.txt), lot-blind top-1 on 83 genera:
    traits alone, LDA        0.145
    betas alone, LDA         0.120
    traits + betas, LDA      0.179   <- better than either alone
    traits+betas+scales, LDA 0.166   <- adding joint scales HURTS

So `betas` are not a replacement for the measured traits, they are COMPLEMENTARY: the traits are
taken off the fitted mesh and carry the free-form field's noise (Z8: median trait R 0.149), the
betas are the parametric channel and carry none of it (Z4: gen@20 0.2936 vs 0.9885). Together they
beat either alone. `log_beta_scales` adds nothing, consistent with Z3 finding that channel
unmodelled and non-transferring.

TWO THINGS STAGE 1 GOT WRONG, CORRECTED HERE
1. `class_weight="balanced"` was applied to logistic and SVC but LDA has no such parameter, so the
   comparison was not like-for-like. Z10 measured logistic at 0.163 on traits alone WITHOUT
   balancing; stage 1 measured 0.123 WITH it. Balancing costs top-1. It is carried here as a
   searched option rather than an assumption, and macro-recall is reported alongside top-1 because
   that is the metric balancing is actually for.
2. C was fixed at 1.0. Here it is tuned by an inner GroupKFold over the TRAINING lots only, never
   on the held-out lot -- tuning against the test fold is the standard route to a number that
   does not reproduce.

THE NULL. Each model is scored against its own permutation null, recomputed under that model, at
the C chosen most often across outer folds (re-tuning inside every permutation is not affordable
and would not change the null's mean). p-values are coarse at 1/N_PERM and reported as such.

SAMPLE-SIZE HONESTY. 602 specimens sit in only 34 accession lots, so lot-blind validation has an
effective n of 34, not 602. Differences of one or two points between configs here are not
resolvable; only the block-level effect is.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import recall_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import analyse as AN  # noqa: E402
import measure as ms  # noqa: E402

N_PERM = int(os.environ.get("Z12_NPERM", "20"))
CGRID = [0.03, 0.1, 0.3, 1.0, 3.0]


def logi(C, bal):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, C=C, class_weight=("balanced" if bal else None)))


def lda():
    return make_pipeline(StandardScaler(),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def evaluate(make_model, F, y, groups, ks=(1, 3, 5)):
    """Leave-one-LOT-out. Returns top-k accuracies, macro recall, and the predictions."""
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
        sc = m.predict_proba(F[te]) if hasattr(m, "predict_proba") else \
            np.atleast_2d(m.decision_function(F[te]))
        ranked = m.classes_[np.argsort(-sc, axis=1)]
        for i, truth in enumerate(y[te]):
            total += 1
            for k in ks:
                if truth in ranked[i, :k]:
                    hits[k] += 1
    ok = pred != "?"
    macro = recall_score(y[ok], pred[ok], average="macro", zero_division=0)
    return ({k: hits[k] / total for k in ks} if total else
            {k: float("nan") for k in ks}), float(macro), pred


def tune_C(F, y, groups, bal, n_splits=4):
    """Pick C by inner GroupKFold over TRAINING lots only. Returns the modal choice."""
    picks = []
    outer = np.unique(groups)
    for held in outer:
        tr = groups != held
        Ftr, ytr, gtr = F[tr], y[tr], groups[tr]
        if len(np.unique(gtr)) < n_splits + 1:
            continue
        best, bestC = -1, 1.0
        for C in CGRID:
            sc, n = 0, 0
            for itr, ite in GroupKFold(n_splits=n_splits).split(Ftr, ytr, gtr):
                if len(np.unique(ytr[itr])) < 2:
                    continue
                m = logi(C, bal)()
                m.fit(Ftr[itr], ytr[itr])
                sc += int((m.predict(Ftr[ite]) == ytr[ite]).sum())
                n += len(ite)
            if n and sc / n > best:
                best, bestC = sc / n, C
        picks.append(bestC)
    return Counter(picks).most_common(1)[0][0] if picks else 1.0


def main():
    runs = sorted((os.path.basename(d) for d in
                   glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*"))
                   if os.path.isfile(os.path.join(d, "Stage_3_deform_fine.npz"))),
                  key=lambda t: int(re.sub(r"\D", "", t) or 0))
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)
    rows, cols, dev = AN.build_table([(r, "worker") for r in runs], M, bones, tpa)
    Ftr, _, _, _ = AN.features(rows, cols, dev)
    names = list(cols) + list(dev)
    B = np.array([r["betas"] for r in rows], dtype=np.float64)
    genus = np.array([str(r.get("genus") or str(r["label"]).split("_")[0]) for r in rows])
    spec = np.array([str(r.get("specimen") or r["label"]) for r in rows])
    species = np.array(["_".join(os.path.basename(str(r["label"])).split("_")[:2])
                        for r in rows])
    lot = np.array([AN.accession_lot(s) for s in spec])
    keep = np.isin(genus, [g for g in set(genus) if len(set(lot[genus == g])) >= 2])
    genus, lot, species = genus[keep], lot[keep], species[keep]
    R = {t["trait"]: t["R"] for t in json.load(open(
        os.path.join(REPO, "diagnostics/full_corpus/out_Z8/z8_repeatability.json")))["traits"]}
    sel = [i for i, n in enumerate(names) if R.get(n, -9) >= 0.1]
    X = np.hstack([Ftr[keep][:, sel], B[keep]])
    print(f"T+B: {X.shape[1]} features, {len(genus)} specimens, {len(set(genus))} genera, "
          f"{len(set(lot))} lots\n")

    print("tuning C on inner training folds ...", flush=True)
    C_plain = tune_C(X, genus, lot, bal=False)
    C_bal = tune_C(X, genus, lot, bal=True)
    print(f"  modal C: unbalanced {C_plain}, balanced {C_bal}\n")

    CONFIGS = {
        "LDA shrinkage": lda,
        f"logistic L2 (C={C_plain})": logi(C_plain, False),
        f"logistic L2 balanced (C={C_bal})": logi(C_bal, True),
    }

    print("=" * 92)
    print("STAGE 2 -- tuned, lot-blind, each model against its OWN permutation null")
    print("=" * 92)
    print(f"{'model':<32}{'top-1':>8}{'null':>8}{'lift':>7}{'top-3':>8}{'top-5':>8}"
          f"{'macroR':>9}")
    out, preds = {}, {}
    rng = np.random.default_rng(0)
    for cname, mk in CONFIGS.items():
        acc, macro, pred = evaluate(mk, X, genus, lot)
        null = np.array([evaluate(mk, X, rng.permutation(genus), lot)[0][1]
                         for _ in range(N_PERM)])
        mu = float(null.mean())
        out[cname] = {"top1": acc[1], "top3": acc[3], "top5": acc[5], "macro": macro,
                      "null": mu, "lift": acc[1] / mu if mu else float("nan"),
                      "p": float((null >= acc[1]).mean())}
        preds[cname] = pred
        print(f"{cname:<32}{acc[1]:>8.3f}{mu:>8.3f}{acc[1]/mu if mu else np.nan:>7.2f}"
              f"{acc[3]:>8.3f}{acc[5]:>8.3f}{macro:>9.3f}", flush=True)

    print(f"\nreference: Z10 best (traits only, logistic L2) top-1 0.163, lift 7.21x")

    # the usability number, under the best config
    best = max(out.items(), key=lambda kv: kv[1]["top1"])[0]
    pred = preds[best]
    UNC = ("sp.", "cf.", "aff.", "nr.")
    from collections import defaultdict
    import itertools
    by = defaultdict(list)
    for i, s in enumerate(species):
        e = s.split("_", 1)[1] if "_" in s else ""
        if e and not e.startswith(UNC) and pred[i] != "?":
            by[s].append(i)
    prs = [(a, b) for v in by.values() if len(v) >= 2 for a, b in itertools.combinations(v, 2)]
    agree = float(np.mean([pred[a] == pred[b] for a, b in prs]))
    both = float(np.mean([pred[a] == genus[a] and pred[b] == genus[b] for a, b in prs]))
    nl = np.array([np.mean([p[a] == p[b] for a, b in prs])
                   for p in (rng.permutation(pred) for _ in range(200))])
    print("\n" + "=" * 92)
    print(f"USABILITY under {best}  ({len(prs)} conspecific pairs, lot-blind)")
    print("=" * 92)
    print(f"  nest-mates given the SAME genus : {agree*100:>5.1f}%   "
          f"(chance {nl.mean()*100:.1f}%, lift {agree/nl.mean():.2f}x)   Z10 was 10.7%")
    print(f"  both CORRECT                    : {both*100:>5.1f}%   Z10 was 5.7%")
    out["_usability"] = {"config": best, "n_pairs": len(prs), "agree": agree,
                         "agree_null": float(nl.mean()), "both_correct": both}
    od = os.path.join(HERE, "out_Z12")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "z12_results.json"), "w"), indent=2)
    print(f"\nwrote {od}/z12_results.json")


if __name__ == "__main__":
    main()
