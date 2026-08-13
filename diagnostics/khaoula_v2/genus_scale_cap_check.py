"""Does scale_cap move the genus classification signal? -- the actual downstream test.

Requested directly: Tier 0 validated scale_cap on a MAGNITUDE metric (anterior joint-scale
compression, 49-83% on bench50_clean) but never checked the thing that actually matters --
does it move the taxonomic signal morphometrics.py's genus classification measures, up or
down, on real data.

WHAT THIS IS NOT, stated precisely so it isn't over-read: a replication of FINAL_REPORT's
"6.1x lift, replicated at 3.8x" headline. That number came from 757 workers + 81 ALL_ANTS_CLEAN
specimens (diagnostics/morphometrics/run_m1_fit_all.sh), which live only on the original
author's own machine (/media/fabi/Data/...) and are NOT reachable from this HPC cluster --
confirmed: no MORPH_* runs exist under diagnostics/moonshot/runs/, no staging directory, same
dangling-symlink pattern already found for clean81/coreg_train/coreg_test/heldout_* when
diagnostics/ was synced. diagnostics/moonshot/cfg/D1_PROD.yaml and the updated
run_m1_fit_all.sh are ready for whoever DOES have that corpus to run; this script is not that.

WHAT THIS IS: the best real, honest test available on this machine. diagnostics/moonshot/
bench50_clean/ (50 real ethanol-preserved scans, genuinely present, not a symlink) was ALREADY
fitted twice this session for the T0.4 anterior-scale check -- once without scale_cap
(T04_bench_baseline), once with it (T04_bench_scale), IDENTICAL specimens, ONLY the recipe
differs. Reusing those existing fits (no new GPU time), this runs the SAME classification code
`diagnostics/morphometrics/analyse.py` uses (measure.py's feature extraction, Mosimann
log-shape-ratios, lot-blind leave-one-accession-lot-out 1-NN, permutation null) on both, and
compares the LIFT (accuracy / null), which is what analyse.py itself says is the number to
track between runs, not raw accuracy.

STATISTICAL POWER, stated honestly before any result: bench50_clean spans 37 genera across 50
specimens (built for diversity, the opposite of what classification needs -- most genera have
exactly 1 member). analyse.py's own default min_n=3 (genus must have >=3 specimens to enter the
test) leaves only 2 usable genera (cephalotes n=3, strumigenys n=5, 8 specimens) here -- too few
for a meaningful test. Lowered to min_n=2 for this check: 9 genera, 22 specimens. This is a
SEVERE reduction from the original's scale (dozens of genera, hundreds of specimens) and the
result should be read as a paired, same-specimens, directional signal check, not a powered
replication. The pairing (identical 50 specimens, single recipe change) is what makes it
informative despite the small n -- it is not an independent, unpaired comparison.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MORPH = os.path.join(REPO, "diagnostics", "morphometrics")
sys.path.insert(0, REPO)
sys.path.insert(0, MORPH)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from analyse import pcs, loo_1nn, perm_test, accession_lot, features, select_features  # noqa: E402


def run_one(run_name, M, bones, tpa, min_n, feature_set="core", npc=10, seed=0):
    rows, cols, dev = ms.measure_run(run_name, "worker", M, bones, tpa)
    if not rows:
        raise SystemExit(f"no data for {run_name} -- check diagnostics/moonshot/runs/{run_name} exists")
    sc_all, asym = ms.symmetrise(rows, cols)
    sdev_all, _ = ms.symmetrise(rows, dev, key="asym_shapedev")
    sc = select_features(sc_all, feature_set)
    sdev = select_features(sdev_all, feature_set)
    F, Z, size, X = features(rows, sc, sdev)

    gen = np.array([r["genus"] if r["genus"] else "?" for r in rows])
    lot = np.array([accession_lot(r["specimen"]) for r in rows])
    label = np.array([r["label"] for r in rows])
    cnt = {g: int((gen == g).sum()) for g in set(gen)}
    keep = np.array([cnt.get(g, 0) >= min_n and g != "?" for g in gen])

    Fk, gk, lk, labk = pcs(F[keep], min(npc, keep.sum() - 1)), gen[keep], lot[keep], label[keep]
    a, m, s, p = perm_test(Fk, gk, groups=lk, seed=seed)

    # per-specimen correctness (same lot-blind 1-NN Fk/gk/lk already computed), so a paired
    # (McNemar-style) comparison between runs is possible -- accuracy alone can't distinguish
    # "one specimen flipped" from "a consistent shift across many specimens", and that
    # distinction matters at n=22.
    D = ((Fk[:, None, :] - Fk[None, :, :]) ** 2).sum(-1)
    for g in np.unique(lk):
        k = np.where(lk == g)[0]
        D[np.ix_(k, k)] = np.inf
    per_specimen_correct = dict(zip(labk, (gk[D.argmin(1)] == gk).tolist()))

    return dict(
        run=run_name, n_total=len(rows), n_kept=int(keep.sum()), n_genera=len(set(gk)),
        genus_counts={g: int(cnt[g]) for g in sorted(set(gk))},
        acc=a, null=m, null_sd=s, p=p, lift=a / max(m, 1e-9),
        per_specimen_correct=per_specimen_correct,
    )


def main():
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)

    print("=" * 90)
    print("genus classification, lot-blind LOO-1NN, min_n=2 (lowered from analyse.py's default 3")
    print("-- see module docstring for why), same 50 bench50_clean specimens, only scale_cap differs")
    print("=" * 90)

    results = {}
    for run_name, label in [("T04_bench_baseline", "baseline (no scale_cap)"),
                             ("T04_bench_scale", "scale_cap (w_scale=0.052)")]:
        r = run_one(run_name, M, bones, tpa, min_n=2)
        results[run_name] = r
        print(f"\n--- {label} ({run_name}) ---")
        print(f"  {r['n_kept']}/{r['n_total']} specimens kept, {r['n_genera']} genera (n>=2): "
              f"{r['genus_counts']}")
        print(f"  lot-blind LOO-1NN accuracy: {100*r['acc']:.1f}%   "
              f"null {100*r['null']:.1f}+-{100*r['null_sd']:.1f}%   p={r['p']:.4f}   "
              f"LIFT {r['lift']:.2f}x")

    rb, rs = results["T04_bench_baseline"], results["T04_bench_scale"]
    print("\n" + "=" * 90)
    print("COMPARISON (paired -- identical 50 specimens, recipe is the only difference)")
    print("=" * 90)
    print(f"  {'':<28}{'baseline':>12}{'scale_cap':>12}{'delta':>12}")
    print(f"  {'accuracy':<28}{100*rb['acc']:>11.1f}%{100*rs['acc']:>11.1f}%{100*(rs['acc']-rb['acc']):>+11.1f}pp")
    print(f"  {'null (chance)':<28}{100*rb['null']:>11.1f}%{100*rs['null']:>11.1f}%"
          f"{100*(rs['null']-rb['null']):>+11.1f}pp")
    print(f"  {'LIFT (acc/null)':<28}{rb['lift']:>11.2f}x{rs['lift']:>11.2f}x{rs['lift']-rb['lift']:>+11.2f}x")
    direction = "UP" if rs["lift"] > rb["lift"] else ("DOWN" if rs["lift"] < rb["lift"] else "UNCHANGED")
    print(f"\n  scale_cap moves genus-classification lift {direction} on this corpus "
          f"({rb['lift']:.2f}x -> {rs['lift']:.2f}x).")
    print(f"  n={rb['n_kept']} specimens, {rb['n_genera']} genera -- read this as directional, "
          f"not as a powered replication of the original 6.1x/3.8x figures (see module docstring).")

    # paired (McNemar-style) per-specimen comparison: does the aggregate delta come from a
    # consistent shift, or from noise flipping a specimen in each direction?
    cb, cs = rb["per_specimen_correct"], rs["per_specimen_correct"]
    common = sorted(set(cb) & set(cs))
    up = sum(1 for lab in common if not cb[lab] and cs[lab])
    down = sum(1 for lab in common if cb[lab] and not cs[lab])
    print(f"\n  PAIRED CHECK (per-specimen, same {len(common)} specimens in both runs):")
    print(f"    baseline-wrong -> scale_cap-correct: {up}")
    print(f"    baseline-correct -> scale_cap-wrong: {down}")
    if up + down == 0:
        print("    no discordant pairs -- the two runs classified every specimen identically.")
    else:
        print(f"    net {up - down:+d} discordant pair(s) out of {len(common)} -- "
              f"{'consistent one-directional shift, but' if down == 0 or up == 0 else 'MIXED direction,'} "
              f"far too few discordant pairs for this delta to be distinguishable from noise on its "
              f"own (a McNemar test needs several discordant pairs to have any power; this has "
              f"{up + down}). Read the direction, not the magnitude, as the informative part.")

    import json
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out_p = os.path.join(HERE, "out", "genus_scale_cap_check.json")
    with open(out_p, "w") as fh:
        json.dump(dict(baseline=rb, scale_cap=rs, direction=direction,
                        paired_flips_up=up, paired_flips_down=down),
                  fh, indent=1, default=float)
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
