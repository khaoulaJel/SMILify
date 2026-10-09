"""G12 -- settle J vs T over MANY noise draws.

G10 (seed 200) gave J-T = -0.164; G11 (seed 97) gave +0.044. Same sigma, same lambda, opposite
sign. The seed-to-seed swing is as large as the effect, so neither run's verdict on that comparison
is usable. This runs the same comparison over many independent noise realisations and reports the
PAIRED difference, which is the only way to answer it at n = 14.

Everything else is held exactly as G11: sigma = 15%, lambda = 0.1, clamp |z| <= 1, full production
objective, deform_verts free, production initialisation.
"""
import os
import sys

os.environ.setdefault("G11_ITERS", os.environ.get("G12_ITERS", "700"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import json  # noqa: E402

import numpy as np  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("G12_SEEDS", "11,23,37,52,71,97,113,151,200,229").split(",")]

if __name__ == "__main__":
    import subprocess
    rows = []
    for sd in SEEDS:
        env = dict(os.environ, G11_SEED=str(sd))
        out = subprocess.run([sys.executable, "-u", os.path.join(HERE, "g11_shrinkage.py")],
                             capture_output=True, text=True, env=env).stdout
        T = J = None
        for line in out.splitlines():
            if line.startswith("T targets direct"):
                T = float(line.split()[3])
            if line.startswith("J joint term, |z|<=1"):
                J = float(line.split()[4])
        if T is not None and J is not None:
            rows.append({"seed": sd, "T": T, "J": J, "diff": J - T})
            print(f"  seed {sd:>4}:  T {T:+.3f}   J {J:+.3f}   J-T {J - T:+.3f}", flush=True)
    d = np.array([r["diff"] for r in rows])
    Ts = np.array([r["T"] for r in rows]); Js = np.array([r["J"] for r in rows])
    from scipy import stats
    w = stats.wilcoxon(Js, Ts) if len(d) >= 6 else None
    print("\n" + "=" * 84)
    print(f"n = {len(d)} noise realisations")
    print(f"  T targets direct : mean {Ts.mean():+.3f}  sd {Ts.std(ddof=1):.3f}")
    print(f"  J joint-fit      : mean {Js.mean():+.3f}  sd {Js.std(ddof=1):.3f}")
    print(f"  paired J - T     : mean {d.mean():+.3f}  sd {d.std(ddof=1):.3f}  "
          f"{int((d > 0).sum())}/{len(d)} positive")
    if w is not None:
        print(f"  Wilcoxon signed-rank p = {w.pvalue:.4f}")
    print()
    if w is not None and w.pvalue < 0.05 and d.mean() > 0:
        print("=> J beats T: fitting adds value over reading the predicted skeleton directly.")
    elif w is not None and w.pvalue < 0.05 and d.mean() < 0:
        print("=> T beats J: below this noise level, skip the fit and measure the skeleton.")
    else:
        print("=> INDISTINGUISHABLE at this sample size. Neither G10's PARTIAL nor G11's PASS")
        print("   stands; the honest statement is that the two are equivalent for traits here.")
    print("=" * 84)
    json.dump(rows, open(os.path.join(HERE, "g12_results.json"), "w"), indent=2)
