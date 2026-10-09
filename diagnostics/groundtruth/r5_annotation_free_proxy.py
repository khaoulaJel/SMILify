"""R5 -- can we detect the named-part misplacement WITHOUT annotations?

R3/R4 established, on 12 annotated specimens, that the fitted model's named parts sit 27-81% of a
body length from the real anatomy while some surface is always nearby. That is the MECHANISM. What
12 specimens cannot establish is PREVALENCE across the 757-fit corpus -- and annotating hundreds of
specimens is not feasible.

So: is there a signal computable from a fit ALONE that tracks the named-part error? If a proxy
correlates on the 12 where truth exists, it can be applied to all 757 where truth does not, giving
corpus-scale prevalence without corpus-scale annotation.

CANDIDATE PROXIES, all annotation-free and all computed from the fitted mesh:
  bilateral    | centroid of part X_r vs the MIRROR of part X_l. An ant is bilaterally symmetric,
               | so a correct fit has these coincide; a misplaced part breaks it.
  buried       | fraction of a part's vertices that are INTERIOR -- further from the mesh's outer
               | hull than the part's own median. A mandible swallowed by the head reads high.
  penetration  | overlap with NON-ADJACENT parts, using part_groups' own adjacency derivation.
               | A mandible inside the head is exactly this.
  extent_ratio | the part's spatial extent relative to the same part on the template. A part
               | collapsed or stretched to satisfy a surface term reads far from 1.

Validation is on the 12 annotated specimens against the R4 per-specimen named-part error. A proxy
is only used at corpus scale if it correlates there.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
sys.path.insert(0, HERE)

from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import trait_extract as TX  # noqa: E402

ANTERIOR = ["ma_r", "ma_l", "an_1_r", "an_1_l", "an_2_r", "an_2_l"]


def main():
    M = TX.load_model(); dom = M["dominant"]; jn = M["jnames"]; T = M["v_template"]
    labels, V = [], []
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        labels += [str(x).replace("_processed.obj", "") for x in d["labels"]]
        V.append(np.asarray(d["verts"], dtype=np.float64))
    V = np.concatenate(V)
    n = len(labels)
    print(f"{n} fits", flush=True)

    PIDX = {p: np.where(dom == jn.index(p))[0] for p in ANTERIOR if p in jn}
    mir = TX.mirror_map(T)                      # exact bilateral vertex pairing on the template
    scale = np.linalg.norm(np.ptp(V, axis=1), axis=1)      # per-fit body diagonal

    # ---- proxy 1: bilateral centroid disagreement of the named anterior parts ----------
    bil = np.zeros(n)
    for a, b in (("ma_r", "ma_l"), ("an_1_r", "an_1_l"), ("an_2_r", "an_2_l")):
        if a not in PIDX or b not in PIDX:
            continue
        ca = V[:, PIDX[a]].mean(1)
        cb = V[:, mir[PIDX[b]]].mean(1)          # mirror-mapped, so it should coincide with ca
        bil += np.linalg.norm(ca - cb, axis=1)
    bil = 100 * bil / (3 * scale)

    # ---- proxy 2: how buried the anterior parts are inside the mesh --------------------
    bur = np.zeros(n)
    for i in range(n):
        tree = cKDTree(V[i])
        c = V[i].mean(0)
        rad = np.linalg.norm(V[i] - c, axis=1)
        out = np.percentile(rad, 90)             # a crude outer shell
        f = []
        for p in ANTERIOR:
            if p not in PIDX:
                continue
            f.append(float((np.linalg.norm(V[i][PIDX[p]] - c, axis=1) < 0.5 * out).mean()))
        bur[i] = float(np.mean(f))
        if i % 200 == 0:
            print(f"  buried {i}/{n}", flush=True)

    # ---- proxy 3: extent of each anterior part relative to the template ----------------
    ext = np.zeros(n)
    for p in ANTERIOR:
        if p not in PIDX:
            continue
        t0 = np.linalg.norm(np.ptp(T[PIDX[p]], 0))
        e = np.linalg.norm(np.ptp(V[:, PIDX[p]], axis=1), axis=1) / max(t0, 1e-9)
        ext += np.abs(np.log(np.maximum(e / np.median(e), 1e-9)))
    ext /= len(PIDX)

    PROX = {"bilateral": bil, "buried": bur, "extent": ext}

    # ---- validate on the 12 where truth exists ----------------------------------------
    r4 = json.load(open(os.path.join(HERE, "r4_results.json")))
    truth = {r["arm"]: r for r in r4}["FREE"]["per_specimen"]
    idx = [i for i, l in enumerate(labels) if l in truth]
    y = np.array([truth[labels[i]] for i in idx])
    print(f"\nvalidation on {len(idx)} annotated specimens "
          f"(named-part error {y.min():.1f}-{y.max():.1f}%)\n")
    print(f"{'proxy':14s}{'Spearman rho':>14}{'p':>10}   usable at corpus scale?")
    keep = {}
    for k, v in PROX.items():
        rho, pv = spearmanr(v[idx], y)
        ok = pv < 0.05 and abs(rho) > 0.5
        keep[k] = ok
        print(f"{k:14s}{rho:>+14.2f}{pv:>10.3f}   {'YES' if ok else 'no'}")

    out = {"n_fits": n, "validation": {k: {"rho": float(spearmanr(PROX[k][idx], y).correlation),
                                           "p": float(spearmanr(PROX[k][idx], y).pvalue),
                                           "usable": bool(keep[k])} for k in PROX}}
    if any(keep.values()):
        best = max([k for k in keep if keep[k]],
                   key=lambda k: abs(spearmanr(PROX[k][idx], y).correlation))
        v = PROX[best]
        thr = float(np.percentile(v[idx], 100 * (y > 20).mean())) if (y > 20).any() else None
        print(f"\ncorpus-scale estimate using '{best}':")
        for q in (10, 25, 50, 75, 90):
            print(f"   p{q:<3d} {np.percentile(v, q):.2f}")
        if thr is not None:
            print(f"   fits above the annotated >20%-error threshold: "
                  f"{100*(v > thr).mean():.0f}% of the corpus")
            out["corpus_fraction_above_threshold"] = float((v > thr).mean())
        out["best_proxy"] = best
    else:
        print("\nNo proxy validates on the annotated set. Corpus-scale prevalence CANNOT be "
              "estimated without more annotation -- that is itself the answer.")
    json.dump(out, open(os.path.join(HERE, "r5_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
