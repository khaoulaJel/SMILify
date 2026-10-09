"""Z7 -- THE GOAL QUESTION: which morphometric traits are trustworthy on worker registrations?

The workers exist for the morphometrics, not for gen@20/spread. Z6 established that the fitter
metric is at its floor (~0.60) -- a conspecific in the training set buys +0.0105, less than
run-to-run noise -- so the useful question is no longer "can the registration be improved" but
"which measurements survive the noise that is there".

THE INSTRUMENT
Z6's paired corpus is 25 species x 2 conspecific workers, mostly consecutive accessions (same lot,
probably nest-mates). Two nest-mates measured through the whole pipeline should give the same
answer for any trait that is real. So for each trait:

    sigma^2_within   from the 25 pairs        -- measurement error + within-species variation
    sigma^2_total    across all 50 specimens  -- the variation the analysis wants to explain
    R = 1 - sigma^2_within / sigma^2_total    -- repeatability (an ICC)

R ~ 1: differences between specimens are real signal.
R ~ 0: the trait is measuring noise, and any between-genus result built on it is unsafe.

Measurements come from `diagnostics/morphometrics/measure.py` -- the shipped measurement layer,
not a re-implementation -- so this reports on the traits the morphometrics report actually uses,
including its Mosimann log-shape-ratio step (size is not recoverable from worker scans; every
trait is a proportion).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as MZ  # noqa: E402

RUN = "Z6_paired50"


def main():
    pairs = json.load(open(os.path.join(HERE, "paired50/pairs.json")))
    rows, cols, devcols = MZ.measure_run(RUN, corpus="worker")
    print(f"measured {len(rows)} specimens, {len(cols)} length traits, "
          f"{len(devcols)} shape-deviation traits")

    lab2i = {r["label"]: i for i, r in enumerate(rows)}
    pair_idx = []
    for sp, (a, b) in pairs.items():
        ka = [k for k in lab2i if a.replace(".obj", "") in k or k in a]
        kb = [k for k in lab2i if b.replace(".obj", "") in k or k in b]
        if ka and kb:
            pair_idx.append((lab2i[ka[0]], lab2i[kb[0]], sp))
    print(f"resolved {len(pair_idx)}/25 conspecific pairs")
    assert len(pair_idx) == 25, "pair resolution failed -- refusing to report repeatability"

    # size-bearing traits go through the shipped Mosimann log-shape-ratio step; devcols are
    # already scale-free. Both are then treated identically.
    X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
    X, _log_size = MZ.log_shape_ratios(X)   # Mosimann: returns (Z, log_size)
    D = np.array([[r[c] for c in devcols] for r in rows], dtype=np.float64) if devcols \
        else np.zeros((len(rows), 0))
    allX = np.hstack([X, D])
    names = list(cols) + list(devcols)

    ia = np.array([p[0] for p in pair_idx])
    ib = np.array([p[1] for p in pair_idx])

    res = []
    for k, nm in enumerate(names):
        v = allX[:, k]
        if not np.all(np.isfinite(v)) or v.std() == 0:
            continue
        # within-pair variance: for a pair, s2 = d^2/2 where d is the difference
        d = v[ia] - v[ib]
        s2_within = float((d ** 2).mean() / 2.0)
        s2_total = float(v.var(ddof=1))
        R = 1.0 - s2_within / s2_total if s2_total > 0 else np.nan
        res.append((nm, R, np.sqrt(s2_within), np.sqrt(s2_total)))

    res.sort(key=lambda r: -r[1])
    print("\n" + "=" * 92)
    print("TRAIT REPEATABILITY on worker registrations   (25 conspecific pairs, n=50)")
    print("R = 1 - within-pair var / total var.  R<=0 means the trait cannot distinguish a")
    print("nest-mate from a random ant -- it is measuring noise.")
    print("=" * 92)
    print(f"{'trait':<34}{'R':>9}{'within sd':>12}{'total sd':>11}")
    for nm, R, sw, st in res:
        flag = "" if R >= 0.5 else ("  <- weak" if R >= 0.2 else "  <- NOISE")
        print(f"{nm:<34}{R:>9.3f}{sw:>12.5f}{st:>11.5f}{flag}")

    Rs = np.array([r[1] for r in res])
    print("\n" + "=" * 92)
    print(f"traits measured           : {len(res)}")
    print(f"  R >= 0.5  (usable)      : {int((Rs >= 0.5).sum())}")
    print(f"  0.2 <= R < 0.5 (weak)   : {int(((Rs >= 0.2) & (Rs < 0.5)).sum())}")
    print(f"  R < 0.2   (noise)       : {int((Rs < 0.2).sum())}")
    print(f"  median R                : {np.median(Rs):.3f}")
    print("=" * 92)

    od = os.path.join(HERE, "out_Z7")
    os.makedirs(od, exist_ok=True)
    json.dump([{"trait": n, "R": r, "within_sd": s, "total_sd": t} for n, r, s, t in res],
              open(os.path.join(od, "z7_repeatability.json"), "w"), indent=2)
    print(f"\nwrote {od}/z7_repeatability.json")


if __name__ == "__main__":
    main()
