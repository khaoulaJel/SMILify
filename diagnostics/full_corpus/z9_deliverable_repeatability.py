"""Z9 -- does the DELIVERABLE survive? Two tests on the 757-worker production fits.

Z8 established that no individual trait is repeatable between conspecific workers (median R 0.149,
nothing >= 0.5). The morphometrics report's headline is not a single trait, though -- it is 1-NN
genus classification, which is multivariate. A distributed signal read one trait at a time looks
exactly like Z8's table, so Z8 does not by itself contradict the headline. These two tests ask
whether it holds up.

TEST 1 -- CONSPECIFIC AGREEMENT. Two nest-mates should be assigned the same genus. This is
repeatability measured on the OUTPUT rather than on the inputs, and it is the number that decides
whether the pipeline is usable for its purpose: a method that puts two workers from one colony in
different genera is not usable however good its aggregate accuracy looks. Chance agreement is
estimated by permuting genus labels, so the baseline is earned rather than assumed.

TEST 2 -- ABLATION BY RELIABILITY. If the 49 "noise" traits (R < 0.2) contribute nothing, dropping
them should leave accuracy unchanged and the pipeline can be simplified and trusted more. If
accuracy FALLS, those traits carry pooled signal despite being individually unrepeatable -- which
would qualify Z8's reading and is the outcome that would most change what we say.

Everything reuses `diagnostics/morphometrics/analyse.py` -- `loo_1nn` (lot-blind), `perm_test`,
`accession_lot` -- rather than reimplementing the protocol, so these numbers sit on the same
validation the report uses. Lot-blinding is not optional: specimens sharing an accession lot are
near-replicates from one colony and scan session.
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

import analyse as AN  # noqa: E402
import measure as ms  # noqa: E402

UNCERTAIN = ("sp.", "cf.", "aff.", "nr.")


def main():
    runs = sorted((os.path.basename(d) for d in
                   glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*"))
                   if os.path.isfile(os.path.join(d, "Stage_3_deform_fine.npz"))),
                  key=lambda t: int(re.sub(r"\D", "", t) or 0))
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)
    rows, cols, dev = AN.build_table([(r, "worker") for r in runs], M, bones, tpa)
    F, Z, size, X = AN.features(rows, cols, dev)
    names = list(cols) + list(dev)
    print(f"\n{len(rows)} specimens, {F.shape[1]} features")

    genus = np.array([str(r.get("genus") or str(r["label"]).split("_")[0]) for r in rows])
    spec = np.array([str(r.get("specimen") or r["label"]) for r in rows])
    lot = np.array([AN.accession_lot(s) for s in spec])
    species = np.array(["_".join(os.path.basename(str(r["label"])).split("_")[:2]) for r in rows])

    # genera need >=2 lots to be classifiable lot-blind at all
    keep = np.isin(genus, [g for g in set(genus) if len(set(lot[genus == g])) >= 2])
    print(f"lot-blind classifiable: {keep.sum()} specimens, {len(set(genus[keep]))} genera")

    Rtab = {t["trait"]: t["R"] for t in
            json.load(open(os.path.join(HERE, "out_Z8/z8_repeatability.json")))["traits"]}

    print("\n" + "=" * 88)
    print("TEST 2 -- lot-blind LOO-1NN genus accuracy vs trait reliability")
    print("=" * 88)
    print(f"{'feature set':<28}{'traits':>8}{'accuracy':>11}{'null':>9}{'lift':>8}{'p':>9}")
    abl = {}
    for label, thr in [("all traits", -9.0), ("R >= 0.0", 0.0), ("R >= 0.1", 0.1),
                       ("R >= 0.2 (drop noise)", 0.2), ("R >= 0.3", 0.3)]:
        sel = [i for i, n in enumerate(names) if Rtab.get(n, -9) >= thr]
        if len(sel) < 3:
            continue
        Fs = F[np.ix_(keep, sel)]
        a, mu, sd, p = AN.perm_test(Fs, genus[keep], n=200, seed=0, groups=lot[keep])
        abl[label] = {"n_traits": len(sel), "acc": a, "null": mu, "lift": a / mu if mu else np.nan,
                      "p": p}
        print(f"{label:<28}{len(sel):>8}{a:>11.3f}{mu:>9.3f}{a/mu if mu else np.nan:>8.2f}"
              f"{p:>9.4f}")

    print("\n" + "=" * 88)
    print("TEST 1 -- do two nest-mates get the SAME genus?  (lot-blind 1-NN prediction)")
    print("=" * 88)
    Fk, gk, lk, spk, sk = F[keep], genus[keep], lot[keep], species[keep], spec[keep]
    D = ((Fk[:, None, :] - Fk[None, :, :]) ** 2).sum(-1)
    for g in np.unique(lk):
        k = np.where(lk == g)[0]
        D[np.ix_(k, k)] = np.inf
    ok = np.isfinite(D).any(1)
    pred = np.full(len(gk), "?", dtype=object)
    pred[ok] = gk[D[ok].argmin(1)]

    def determined(s):
        e = s.split("_", 1)[1] if "_" in s else ""
        return bool(e) and not e.startswith(UNCERTAIN)

    by = defaultdict(list)
    for i, s in enumerate(spk):
        if determined(s) and ok[i]:
            by[s].append(i)
    pairs = [(a, b) for v in by.values() if len(v) >= 2
             for a, b in itertools.combinations(v, 2)]
    agree = np.mean([pred[a] == pred[b] for a, b in pairs])
    both = np.mean([pred[a] == gk[a] and pred[b] == gk[b] for a, b in pairs])

    rng = np.random.default_rng(0)
    null = []
    for _ in range(200):
        pp = rng.permutation(pred)
        null.append(np.mean([pp[a] == pp[b] for a, b in pairs]))
    null = np.array(null)

    print(f"  conspecific pairs (determined species, lot-blind): {len(pairs)}")
    print(f"  BOTH nest-mates assigned the SAME genus : {agree*100:>6.1f}%")
    print(f"  chance agreement (label permutation)    : {null.mean()*100:>6.1f}% "
          f"+- {null.std()*100:.1f}")
    print(f"  lift                                    : {agree/null.mean():>6.2f}x")
    print(f"  both assigned the CORRECT genus         : {both*100:>6.1f}%")
    print("\n  A pair can agree and both be wrong, so 'same genus' is the repeatability number")
    print("  and 'both correct' is the accuracy number. Reported separately on purpose.")

    od = os.path.join(HERE, "out_Z9")
    os.makedirs(od, exist_ok=True)
    json.dump({"ablation": abl, "n_pairs": len(pairs), "agree": float(agree),
               "agree_null": float(null.mean()), "both_correct": float(both)},
              open(os.path.join(od, "z9_results.json"), "w"), indent=2)
    print(f"\nwrote {od}/z9_results.json")


if __name__ == "__main__":
    main()
