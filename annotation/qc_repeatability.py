"""qc_repeatability.py -- how consistently can a human place these landmarks?

THE QUESTION THIS ANSWERS. Supervision drives held-out surface-landmark error to ~7% of Weber's
length in every configuration tested (V8 6.9%, V9 7.6/7.1%, V10 7.4/8.0/7.0/6.7) while the
SUPERVISED points reach 0.1%. That behaves like a floor. If a second, independent annotation of the
same specimen disagrees with the first by ~7%, every anatomical number in the V-series has reached
the instrument's ceiling and further tuning measures noise. If it disagrees by ~1-2%, the floor is
something else and real headroom remains.

USAGE
  Put the repeat exports in annotation/landmarks_repeat/ with the SAME filenames as the originals
  in annotation/landmarks/, then:
      python annotation/qc_repeatability.py

Distances are reported in % of that specimen's Weber's length, the same unit every V-series
anatomical number uses, so the comparison is direct.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.join(HERE, "landmarks")
B_DIR = os.path.join(HERE, "landmarks_repeat")


def load(p):
    d = json.load(open(p))
    return d["specimen_id"], {k: np.array(v["original"], float)
                              for k, v in d["landmarks"].items()}


def main():
    pairs = []
    for pb in sorted(glob.glob(os.path.join(B_DIR, "*_traits.json"))):
        pa = os.path.join(A_DIR, os.path.basename(pb))
        if not os.path.exists(pa):
            print(f"no round-1 file for {os.path.basename(pb)} -- skipped")
            continue
        pairs.append((load(pa), load(pb)))
    if not pairs:
        print(f"nothing to compare: put repeat exports in {B_DIR}")
        return 1

    print(f"{len(pairs)} specimen(s) annotated twice\n")
    per_lm, per_spec = {}, []
    for (sid, A), (_, B) in pairs:
        WL = np.linalg.norm(A["wl_anterior_r"] - A["wl_posterior_r"])
        d = {k: 100 * np.linalg.norm(A[k] - B[k]) / WL for k in A if k in B}
        for k, v in d.items():
            per_lm.setdefault(k, []).append(v)
        per_spec.append((sid, float(np.median(list(d.values())))))
        print(f"  {sid[:44]:46s} median {np.median(list(d.values())):5.1f}%  "
              f"worst {max(d, key=d.get)} {max(d.values()):.1f}%")

    print(f"\n{'landmark':24s}{'n':>3}{'median':>9}{'max':>8}")
    for k in sorted(per_lm, key=lambda k: -np.median(per_lm[k])):
        v = np.array(per_lm[k])
        print(f"{k:24s}{len(v):>3}{np.median(v):>8.1f}%{v.max():>7.1f}%")
    allv = np.concatenate([np.array(v) for v in per_lm.values()])
    R = float(np.median(allv))
    print(f"\n{'ALL':24s}{len(allv):>3}{R:>8.1f}%{allv.max():>7.1f}%")

    print("\n" + "=" * 72)
    print(f"HUMAN LANDMARK REPEATABILITY = {R:.1f}% of Weber's length (median)")
    print("=" * 72)
    print(f"  V-series held-out surface error floor: ~7%")
    if R > 5.0:
        print(f"  -> {R:.1f}% is COMPARABLE to the ~7% floor. The V-series anatomical numbers are")
        print( "     at the instrument's ceiling; pushing below ~7% would be fitting annotation")
        print( "     noise. Do NOT spend corpus-scale compute chasing it.")
    elif R > 2.5:
        print(f"  -> {R:.1f}% is BELOW the ~7% floor but not negligible. Roughly "
              f"{100*(1-R/7):.0f}% of the")
        print( "     remaining error is real; modest headroom exists, and any claimed improvement")
        print(f"     smaller than {R:.1f}% is uninterpretable.")
    else:
        print(f"  -> {R:.1f}% is WELL BELOW the ~7% floor. The floor is NOT annotation noise --")
        print( "     it is correspondence or the shape space, and real headroom remains.")
    json.dump({"repeatability_pct_WL": R, "per_landmark":
               {k: float(np.median(v)) for k, v in per_lm.items()},
               "per_specimen": dict(per_spec)},
              open(os.path.join(HERE, "landmark_repeatability.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
