"""A3 analysis -- morphometric recovery + correspondence guard, arm B (no offset/normal) vs arm A.

Bars fixed in PREREGISTRATION_A3_offset_normal_downstream.md before the fits ran.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "groundtruth"))

import trait_extract as TX        # noqa: E402
from measure import load_model    # noqa: E402

GT = os.path.join(REPO, "diagnostics/moonshot/synth_power48/ground_truth.npz")
ARMS = {"A_prod": "diagnostics/moonshot/runs/A3_prod/Stage_3_deform_fine.npz",
        "B_nooffnorm": "diagnostics/moonshot/runs/A3_nooffnorm/Stage_3_deform_fine.npz"}
TRAITS = ("HW", "HL", "ML", "SL", "WL", "PetL", "GL", "FL", "TBL")


def arm_metrics(path, M, V_gt, gt_names):
    z = np.load(os.path.join(REPO, path), allow_pickle=True)
    V = z["verts"].astype(np.float64)
    names = [str(x).replace(".obj", "") for x in z["labels"]]
    if names != gt_names:                       # align by name, never assume order
        order = [names.index(n) for n in gt_names]
        V = V[order]; dv = z["deform_verts"].astype(np.float64)[order]
    else:
        dv = z["deform_verts"].astype(np.float64)
    ext_gt = (V_gt.max(1) - V_gt.min(1)).mean(-1)
    ext = (V.max(1) - V.min(1)).mean(-1)
    corr = np.linalg.norm(V - V_gt, axis=-1).mean(-1) / ext_gt      # TRUE correspondence error
    deform = np.linalg.norm(dv, axis=-1).mean(-1) / ext_gt
    tr_gt, tr = TX.traits(V_gt, M=M), TX.traits(V, M=M)
    err = {}
    for t in TRAITS:
        if t not in tr_gt:
            continue
        g, f_ = np.asarray(tr_gt[t], float) / ext_gt, np.asarray(tr[t], float) / ext
        err[t] = np.abs(f_ - g) / g * 100.0     # scale-corrected, as in M2
    return dict(corr=corr, deform=deform, trait_err=err)


def main():
    M = load_model()
    gt = np.load(GT, allow_pickle=True)
    V_gt = gt["verts"].astype(np.float64)
    gt_names = [str(x) for x in gt["names"]]

    res = {}
    for arm, path in ARMS.items():
        if not os.path.exists(os.path.join(REPO, path)):
            print(f"[a3] MISSING {arm}: {path}"); return
        res[arm] = arm_metrics(path, M, V_gt, gt_names)

    A, B = res["A_prod"], res["B_nooffnorm"]
    # primary: per-specimen median-across-traits trait error
    trs = [t for t in TRAITS if t in A["trait_err"]]
    a_spec = np.median(np.stack([A["trait_err"][t] for t in trs]), axis=0)
    b_spec = np.median(np.stack([B["trait_err"][t] for t in trs]), axis=0)
    rel = (np.median(b_spec) - np.median(a_spec)) / np.median(a_spec) * 100.0
    sgn = stats.wilcoxon(a_spec, b_spec)
    better = int((b_spec < a_spec).sum())

    corr_rel = (B["corr"].mean() - A["corr"].mean()) / A["corr"].mean() * 100.0
    guard_ok = bool(corr_rel <= 5.0)
    primary_ok = bool(rel <= -10.0 and sgn.pvalue < 0.05)
    verdict = ("PASS" if (primary_ok and guard_ok)
               else "PARTIAL" if (rel < 0 and sgn.pvalue < 0.05 and guard_ok)
               else "FAIL")

    print(f"{'trait':7s}{'A prod':>10s}{'B no-off/norm':>16s}{'rel %':>9s}")
    for t in trs:
        ma, mb = np.median(A["trait_err"][t]), np.median(B["trait_err"][t])
        print(f"{t:7s}{ma:>9.2f}%{mb:>15.2f}%{(mb-ma)/ma*100:>8.1f}%")
    print(f"\nPRIMARY  median trait err: A {np.median(a_spec):.2f}%  B {np.median(b_spec):.2f}%  "
          f"rel {rel:+.1f}%  wilcoxon p={sgn.pvalue:.4g}  B better on {better}/48")
    print(f"GUARD    correspondence err: A {A['corr'].mean():.4f}  B {B['corr'].mean():.4f}  "
          f"rel {corr_rel:+.1f}%  (bar <=+5%)  OK={guard_ok}")
    print(f"CONTEXT  deform mag: A {A['deform'].mean():.4f}  B {B['deform'].mean():.4f}  "
          f"({(B['deform'].mean()-A['deform'].mean())/A['deform'].mean()*100:+.1f}%)")
    print(f"\nVERDICT: {verdict}")

    json.dump(dict(verdict=verdict, primary_rel_pct=float(rel),
                   wilcoxon_p=float(sgn.pvalue), b_better_n=better,
                   corr_rel_pct=float(corr_rel), guard_ok=guard_ok,
                   median_trait_err=dict(A=float(np.median(a_spec)), B=float(np.median(b_spec))),
                   per_trait={t: dict(A=float(np.median(A["trait_err"][t])),
                                      B=float(np.median(B["trait_err"][t]))) for t in trs},
                   corr=dict(A=float(A["corr"].mean()), B=float(B["corr"].mean())),
                   deform=dict(A=float(A["deform"].mean()), B=float(B["deform"].mean()))),
              open(os.path.join(OUT, "a3_results.json"), "w"), indent=2)
    print("wrote", os.path.join(OUT, "a3_results.json"))


if __name__ == "__main__":
    main()
