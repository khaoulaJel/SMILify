"""P0 readout (PREREGISTRATION.md measurements 1-4).

1. Articulation per specimen = A7 statistic (median |leg bend - rest|) on the fit's FK joints
   (angles, so the fitter's normalised frame does not matter). Fits reload-verified (JAB loader).
2. Validity gate on JAB: A_prod fitted bend (seed mean) vs expert bend (Q3a A7), Spearman rho >= 0.5.
3. Stability: within-specimen SD over seeds 0-2 on stab48 vs between-specimen SD; usable if <= 0.5.
4. Where the 11 JAB specimens fall in the P0 distribution.
Writes out/P0_articulation.json.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "joint_alignment_benchmark", "tools"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import make_shift_sets as mss  # noqa: E402  (bend statistic, same code as S2)
import model_joints as MJ  # noqa: E402

O = "/hpcwork/nao48500/review_methods/P0"


def bends_of(npz):
    z = np.load(npz, allow_pickle=True)
    labs = [MJ.clean_label(x) for x in z["labels"]]
    fits = MJ.load_fit(npz, set(labs))
    return {s: mss.bend_stat([fits[s]["FK"]]) for s in labs}


def main():
    split = json.load(open(os.path.join(HERE, "out", "split.json")))
    art = {}
    for p in sorted(glob.glob(os.path.join(O, "runs", "c??_s0", "Stage_3_deform_fine.npz"))):
        art.update(bends_of(p))
    missing = set(split["pose_train"] + split["pose_eval"]) - set(art)
    print(f"[P0] fitted {len(art)} / {split['n']} specimens; missing {len(missing)}")
    a = np.array(list(art.values()))
    print(f"[P0] fitted articulation (median leg bend vs rest): pct 5/25/50/75/95 = "
          f"{np.percentile(a, [5, 25, 50, 75, 95]).round(1)}")
    tr = np.array([art[s] for s in split["pose_train"] if s in art])
    ev = np.array([art[s] for s in split["pose_eval"] if s in art])
    ks = stats.ks_2samp(tr, ev)
    print(f"[P0] pose_train median {np.median(tr):.1f} (n={len(tr)}) vs pose_eval {np.median(ev):.1f} "
          f"(n={len(ev)}), KS p = {ks.pvalue:.3f}")

    # 2. validity gate on JAB
    a7 = json.load(open(os.path.join(HERE, "..", "Q3a_cse_transfer_case_study", "out", "A7_articulation.json")))
    jab = {}
    for s in (0, 1, 2):
        for sid, b in bends_of(f"/hpcwork/nao48500/jab_runs/A_prod_s{s}/Stage_3_deform_fine.npz").items():
            jab.setdefault(sid, []).append(b)
    sids = sorted(jab)
    fitted = np.array([np.mean(jab[s]) for s in sids])
    expert = np.array([a7[s]["median"] for s in sids])
    rho = stats.spearmanr(fitted, expert)
    gate = rho.correlation >= 0.5
    print(f"[P0] validity gate (JAB, n=11): Spearman fitted vs expert = {rho.correlation:.2f} (p {rho.pvalue:.3f}); "
          f"median signed diff fitted - expert = {np.median(fitted - expert):+.1f} deg -> {'PASS' if gate else 'FAIL'}")

    # 3. stability
    st = {}
    for s in (1, 2):
        st[s] = bends_of(os.path.join(O, "runs", f"stab48_s{s}", "Stage_3_deform_fine.npz"))
    trip = np.array([[art[k], st[1][k], st[2][k]] for k in split["stab48"] if k in art and k in st[1] and k in st[2]])
    within = float(np.mean(trip.std(1)))
    between = float(np.std(trip.mean(1)))
    print(f"[P0] stability on {len(trip)} specimens: within-specimen SD {within:.2f} deg, between {between:.2f} "
          f"-> ratio {within / between:.2f} ({'usable' if within / between <= 0.5 else 'NOT usable'})")

    # 4. JAB in the P0 distribution
    pct = {s: float(stats.percentileofscore(a, f)) for s, f in zip(sids, fitted)}
    print("[P0] JAB fitted articulation as a percentile of P0:", {k[:10]: round(v) for k, v in pct.items()})
    json.dump(dict(per_specimen=art, pct=np.percentile(a, [5, 25, 50, 75, 95]).tolist(),
                   train_eval_ks_p=float(ks.pvalue),
                   gate=dict(spearman=float(rho.correlation), p=float(rho.pvalue),
                             median_signed_diff=float(np.median(fitted - expert)), passed=bool(gate),
                             fitted=dict(zip(sids, fitted.tolist())), expert=dict(zip(sids, expert.tolist()))),
                   stability=dict(n=len(trip), within_sd=within, between_sd=between, ratio=within / between),
                   jab_percentiles=pct),
              open(os.path.join(HERE, "out", "P0_articulation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
