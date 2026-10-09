"""V2 -- is `deform_verts` what produces the corpus-wide biologically impossible traits, and does
removing it cost biological signal?

See PREREGISTRATION_V2_corpus_deform.md. trainer.py:352 adds deformation as the final operation,
so `verts - deform_verts` exactly recovers the undeformed fit: ablation, not re-optimisation.

Also reconstructs `traits_Z8.npz`, whose builder is absent from the repo -- reproducing T1's
69/757 exactly is the voiding control.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
sys.path.insert(0, HERE)

import trait_extract as TX  # noqa: E402

BOUNDS = {"HW/HL": (0.40, 2.00), "SL/HL": (0.20, 3.00), "PetL/WL": (0.02, 0.80),
          "ML/HL": (0.10, 2.00), "GL/WL": (0.50, 4.00)}
RATIOS = list(BOUNDS) + ["HL/WL", "HW/WL"]


def ratios_of(t):
    return {r: t[r.split("/")[0]] / np.maximum(t[r.split("/")[1]], 1e-12) for r in RATIOS}


def flagged_mask(R):
    """Unique specimens with ANY impossible trait -- T1's convention."""
    n = len(next(iter(R.values())))
    f = np.zeros(n, bool)
    for k, (lo, hi) in BOUNDS.items():
        f |= (R[k] < lo) | (R[k] > hi)
    return f


def f_ratio(X, genus):
    """Between-genus / within-genus variance, per feature. T4's statistic."""
    out = []
    for j in range(X.shape[1]):
        x = X[:, j]
        gm = defaultdict(list)
        for g, v in zip(genus, x):
            gm[g].append(v)
        gm = {g: np.array(v) for g, v in gm.items() if len(v) >= 2}
        if len(gm) < 2:
            out.append(np.nan); continue
        grand = np.mean([v.mean() for v in gm.values()])
        between = np.mean([(v.mean() - grand) ** 2 for v in gm.values()])
        within = np.mean([v.var(ddof=1) for v in gm.values()])
        out.append(between / max(within, 1e-12))
    return np.array(out)


def main():
    M = TX.load_model()
    lm = TX.load_landmarks(M)
    print(f"landmark set {TX.landmark_version()}", flush=True)

    labels, V, D = [], [], []
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        labels += [str(x) for x in d["labels"]]
        V.append(np.asarray(d["verts"], dtype=np.float64))
        D.append(np.asarray(d["deform_verts"], dtype=np.float64))
    V = np.concatenate(V); D = np.concatenate(D)
    n = len(labels)
    print(f"{n} fits, verts {V.shape}", flush=True)

    t_with = TX.traits(V, M, lm)
    t_without = TX.traits(V - D, M, lm)
    R_with, R_without = ratios_of(t_with), ratios_of(t_without)

    # ---------------- VOIDING CONTROL: reproduce traits_Z8.npz and T1's 69 --------------
    ref = np.load(os.path.join(HERE, "traits_Z8.npz"))
    ref_lab = [str(x) for x in ref["labels"]]
    order = [ref_lab.index(l) for l in labels] if ref_lab != labels else list(range(n))
    worst, worst_k = 0.0, ""
    for k in ref.files:
        if k == "labels" or k not in t_with:
            continue
        a, b = np.asarray(t_with[k]), np.asarray(ref[k])[order]
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
        if rel > worst:
            worst, worst_k = rel, k
    f_with = flagged_mask(R_with)
    f_without = flagged_mask(R_without)
    ctrl = worst < 1e-6 and int(f_with.sum()) == 69

    print("\n" + "=" * 76)
    print("V2 -- does deform_verts produce the corpus-wide impossible traits?")
    print("=" * 76)
    print(f"[VOIDING CONTROL] max relative deviation from traits_Z8.npz: {worst:.3e} ({worst_k}); "
          f"flagged {int(f_with.sum())}/757 (T1: 69)  ->  "
          f"{'PASS' if ctrl else 'FAIL -- NOTHING BELOW IS COMPARABLE TO T1'}")

    # ---------------- H1: flagged fraction ---------------------------------------------
    print(f"\n{'ratio':>9}{'bounds':>16}{'flagged WITH':>14}{'flagged WITHOUT':>17}")
    for k, (lo, hi) in BOUNDS.items():
        a = int(((R_with[k] < lo) | (R_with[k] > hi)).sum())
        b = int(((R_without[k] < lo) | (R_without[k] > hi)).sum())
        print(f"{k:>9}{f'[{lo:.2f},{hi:.2f}]':>16}{a:>14d}{b:>17d}")
    print(f"{'ANY':>9}{'':>16}{int(f_with.sum()):>14d}{int(f_without.sum()):>17d}")
    print(f"{'':>9}{'':>16}{100*f_with.mean():>13.1f}%{100*f_without.mean():>16.1f}%")
    rescued = int((f_with & ~f_without).sum()); created = int((~f_with & f_without).sum())
    print(f"\ntransitions: flagged -> valid {rescued}   valid -> flagged {created}")
    h1 = f_without.mean() < 0.03
    print(f"[H1 deformation makes the violations] {100*f_with.mean():.1f}% -> "
          f"{100*f_without.mean():.1f}% (bar <3.0%)  ->  {'PASS' if h1 else 'FAIL'}")

    # ---------------- H2: is it trivial? spread retention -------------------------------
    print(f"\n{'ratio':>9}{'sd(log) WITH':>14}{'sd(log) WITHOUT':>17}{'retained':>10}")
    ret = []
    for k in RATIOS:
        a = float(np.std(np.log(np.maximum(R_with[k], 1e-12))))
        b = float(np.std(np.log(np.maximum(R_without[k], 1e-12))))
        ret.append(b / max(a, 1e-12))
        print(f"{k:>9}{a:>14.4f}{b:>17.4f}{ret[-1]:>9.2f}x")
    h2 = min(ret) >= 0.60
    print(f"[H2 not trivial -- variation retained] worst ratio retains {min(ret):.2f}x "
          f"(bar >=0.60x)  ->  {'PASS' if h2 else 'FAIL -- H1 IS AN ARTEFACT OF BLANDNESS'}")

    # ---------------- H3: genus signal ---------------------------------------------------
    genus = np.array([l.split("_")[0] for l in labels])
    # T4 reported 27 genera / 386 specimens, so its per-genus threshold was stricter than >=2.
    # Sweep it: the WITH-vs-WITHOUT ratio is the claim, absolute F depends on the threshold.
    Fsweep = {}
    for MIN in (2, 3, 5, 8):
        keep = np.array([np.sum(genus == g) >= MIN for g in genus])
        if keep.sum() < 10:
            continue
        Xw = np.column_stack([np.log(np.maximum(R_with[k], 1e-12)) for k in RATIOS])[keep]
        Xo = np.column_stack([np.log(np.maximum(R_without[k], 1e-12)) for k in RATIOS])[keep]
        fw, fo = f_ratio(Xw, genus[keep]), f_ratio(Xo, genus[keep])
        Fsweep[MIN] = {"n_genera": int(len(set(genus[keep]))), "n": int(keep.sum()),
                       "mean_F_with": float(np.nanmean(fw)),
                       "mean_F_without": float(np.nanmean(fo)),
                       "per_trait_with": {k: float(v) for k, v in zip(RATIOS, fw)},
                       "per_trait_without": {k: float(v) for k, v in zip(RATIOS, fo)}}
        if MIN == 5:
            F_w, F_o = fw, fo
    print(f"\n{'min/genus':>10}{'genera':>8}{'n':>6}{'mean F WITH':>13}{'F WITHOUT':>11}{'ratio':>8}")
    for MIN, v in Fsweep.items():
        print(f"{MIN:>10}{v['n_genera']:>8}{v['n']:>6}{v['mean_F_with']:>13.2f}"
              f"{v['mean_F_without']:>11.2f}{v['mean_F_without']/v['mean_F_with']:>7.2f}x")
    print(f"\nper-trait at min/genus=5:\n{'ratio':>10}{'F WITH':>10}{'F WITHOUT':>12}")
    for i, k in enumerate(RATIOS):
        print(f"{k:>10}{F_w[i]:>10.2f}{F_o[i]:>12.2f}")
    mw, mo = Fsweep[5]["mean_F_with"], Fsweep[5]["mean_F_without"]
    h3 = mo >= 0.8 * mw
    print(f"{'mean':>9}{mw:>10.2f}{mo:>12.2f}   ({mo/mw:.2f}x)")
    print(f"[H3 no biological signal lost] mean F {mw:.2f} -> {mo:.2f} = {mo/mw:.2f}x "
          f"(bar >=0.80x)  ->  {'PASS' if h3 else 'FAIL'}")

    # ---------------- asymmetry ----------------------------------------------------------
    a_w, a_o = TX.asymmetry(t_with), TX.asymmetry(t_without)
    print("\nbilateral asymmetry (median):")
    for k in sorted(a_w):
        print(f"  {k:>4}  {np.median(a_w[k]):.4f} -> {np.median(a_o[k]):.4f}")

    json.dump({
        "n": n, "control": {"max_rel_dev": worst, "flagged_with": int(f_with.sum()), "pass": ctrl},
        "flagged_with": int(f_with.sum()), "flagged_without": int(f_without.sum()),
        "rescued": rescued, "created": created,
        "spread_retained": {k: r for k, r in zip(RATIOS, ret)},
        "F_sweep": Fsweep,
        "mean_F_with": mw, "mean_F_without": mo,
        "asym_with": {k: float(np.median(v)) for k, v in a_w.items()},
        "asym_without": {k: float(np.median(v)) for k, v in a_o.items()},
        "bars": {"H1": bool(h1), "H2": bool(h2), "H3": bool(h3)},
        "flagged_labels_with": [labels[i] for i in np.where(f_with)[0]],
        "flagged_labels_without": [labels[i] for i in np.where(f_without)[0]],
    }, open(os.path.join(HERE, "v2_results.json"), "w"), indent=2)
    print("=" * 76)


if __name__ == "__main__":
    main()
