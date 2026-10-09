"""Z11 stage 1 -- feature-block search. Which representation actually carries genus?

Z10 established the only axis with headroom left: 1-NN -> L2 logistic bought +44% relative on the
same features. This asks the prior question -- whether the FEATURES are right at all.

THE OBSERVATION THIS IS BUILT ON
`analyse.features()` uses measured lengths and part dimensions only. It has never used `betas`,
even though `measure_run` stores them on every row and Z4 measured them as by far the most
transferable channel the pipeline produces:

    betas          gen@20/spread 0.2936   <- transfers
    deform_verts   gen@20/spread 0.9885   <- does not
    composed       gen@20/spread 0.6147

The traits are derived from the fitted mesh, i.e. downstream of both channels, so they inherit the
free-form noise. `betas` are the shape-space coordinates directly. If genus lives in shape, the
betas should carry it better than measurements taken off a noisy surface.

BLOCKS
  T   traits, R >= 0.1                (Z10's best, the baseline to beat)
  B   betas, 25 shape-space coords    never tried
  S   log_beta_scales, per-joint      the channel Z3 showed is unmodelled but non-empty
  combinations of the above

Size: the traits go through the shipped Mosimann log-shape-ratio step, so they are proportions.
`betas` and `log_beta_scales` are standardised here but NOT size-corrected -- absolute size is not
recoverable from worker scans (measure.py's header), so any size signal in them is scan-arbitrary
rather than biological. Whether that helps or hurts is exactly what the block comparison measures.

Stage 1 is a search: fixed C, no permutation null, ranked by lot-blind top-1. Stage 2 (z12) takes
the winners, tunes C inside the training folds, and computes the null. Splitting it this way keeps
the search cheap and keeps the reported number honest -- nothing here is a final figure.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

import analyse as AN  # noqa: E402
import measure as ms  # noqa: E402


def topk_lot_blind(make_model, F, y, groups, ks=(1, 3, 5)):
    """Leave-one-LOT-out top-k accuracy. A fresh model per held-out lot.

    Top-k matters here and is not padding. With 83 genera the practical use of this tool is a
    SHORTLIST -- 'which few genera should a specialist consider' -- and a method can be useful at
    that job while being useless at top-1. Reporting only top-1 would hide it.
    """
    hits = {k: 0 for k in ks}
    total = 0
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        cls = np.unique(y[tr])
        if len(cls) < 2:
            continue
        m = make_model()
        m.fit(F[tr], y[tr])
        if hasattr(m, "predict_proba"):
            sc = m.predict_proba(F[te])
        elif hasattr(m, "decision_function"):
            sc = np.atleast_2d(m.decision_function(F[te]))
        else:
            sc = None
        order = np.argsort(-sc, axis=1) if sc is not None else None
        names = m.classes_ if hasattr(m, "classes_") else cls
        for i, truth in enumerate(y[te]):
            total += 1
            if order is None:
                continue
            ranked = names[order[i]]
            for k in ks:
                if truth in ranked[:k]:
                    hits[k] += 1
    return {k: hits[k] / total for k in ks} if total else {k: float("nan") for k in ks}


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
    S = np.array([r["log_beta_scales"] for r in rows], dtype=np.float64).reshape(len(rows), -1)
    print(f"traits {Ftr.shape}  betas {B.shape}  log_beta_scales {S.shape}")

    genus = np.array([str(r.get("genus") or str(r["label"]).split("_")[0]) for r in rows])
    spec = np.array([str(r.get("specimen") or r["label"]) for r in rows])
    lot = np.array([AN.accession_lot(s) for s in spec])
    keep = np.isin(genus, [g for g in set(genus) if len(set(lot[genus == g])) >= 2])
    genus, lot = genus[keep], lot[keep]
    Ftr, B, S = Ftr[keep], B[keep], S[keep]
    print(f"lot-blind classifiable: {len(genus)} specimens, {len(set(genus))} genera, "
          f"{len(set(lot))} lots\n")

    R = {t["trait"]: t["R"] for t in json.load(open(
        os.path.join(REPO, "diagnostics/full_corpus/out_Z8/z8_repeatability.json")))["traits"]}
    sel = [i for i, n in enumerate(names) if R.get(n, -9) >= 0.1]
    T = Ftr[:, sel]

    BLOCKS = {
        "T  traits R>=0.1": T,
        "B  betas": B,
        "S  log_beta_scales": S,
        "T+B": np.hstack([T, B]),
        "B+S": np.hstack([B, S]),
        "T+B+S": np.hstack([T, B, S]),
    }
    MODELS = {
        "logistic L2": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=1.0,
                                                 class_weight="balanced")),
        "LDA shrink": lambda: make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "linear SVC": lambda: make_pipeline(
            StandardScaler(), LinearSVC(C=0.1, class_weight="balanced", max_iter=5000)),
    }

    print("=" * 84)
    print("STAGE 1 -- lot-blind top-k genus accuracy by feature block  (C fixed, no null yet)")
    print("Z10 reference: traits R>=0.1 + logistic L2 = 0.163 top-1")
    print("=" * 84)
    print(f"{'block':<22}{'dims':>6}{'model':<14}{'top-1':>9}{'top-3':>9}{'top-5':>9}")
    res = {}
    for bname, X in BLOCKS.items():
        for mname, mk in MODELS.items():
            acc = topk_lot_blind(mk, X, genus, lot)
            res[f"{bname}|{mname}"] = {"dims": X.shape[1], **{f"top{k}": v
                                                              for k, v in acc.items()}}
            print(f"{bname:<22}{X.shape[1]:>6}{mname:<14}"
                  f"{acc[1]:>9.3f}{acc[3]:>9.3f}{acc[5]:>9.3f}", flush=True)

    best = max(res.items(), key=lambda kv: kv[1]["top1"])
    print("\n" + "=" * 84)
    print(f"best: {best[0]}  top-1 {best[1]['top1']:.3f}  top-3 {best[1]['top3']:.3f}  "
          f"top-5 {best[1]['top5']:.3f}")
    print(f"Z10 baseline (traits R>=0.1, logistic L2, C=1): 0.163 top-1")
    print("=" * 84)
    od = os.path.join(HERE, "out_Z11")
    os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "z11_blocks.json"), "w"), indent=2)
    print(f"wrote {od}/z11_blocks.json")


if __name__ == "__main__":
    main()
