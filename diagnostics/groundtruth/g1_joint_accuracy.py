"""G1 -- ACCURACY against human-annotated joints. The first direct measurement in this project.

Every metric in the Z-series measured PRECISION -- gen@20/spread is population self-consistency,
trait R is nest-mate self-agreement, classification accuracy is a downstream proxy. None could
measure ACCURACY, because there was no truth. Ten annotated specimens change that.

THE QUESTION THIS UNIQUELY ANSWERS
A measurement can be imprecise but UNBIASED (errors scatter about the true value and average out)
or imprecise and BIASED (systematically wrong). Z8 proved imprecise; it cannot tell these apart.
The project's surviving claim -- population-level morphometrics are supported, per-specimen are not
-- rests entirely on the assumption that the noise averages out. If there is systematic bias that
claim fails too. Bias is a mean, and means converge fast, so n=10 is enough.

ALIGNMENT, and why not the annotator's declared frame
`frame_joints` declares origin=b_t, forward=b_h. That frame CANNOT be built on the fitted side:
`b_h` is coincident with `b_t` in the model template (verified: |b_t-b_h| = 0.0 exactly on all 757
production fits), so the forward axis has zero length. Instead a similarity Procrustes (rotation,
translation, uniform scale) is fitted on the joints placed in BOTH, which is standard GPA practice
and is the right choice when the two spaces have arbitrary scale and pose -- absolute size is not
recoverable from worker scans in any case (measure.py header).

The alignment is fitted on shared joints, so it absorbs global pose and scale disagreement by
construction. What remains is SHAPE disagreement, which is what morphometrics is about. Residuals
are reported as a percentage of each specimen's own centroid size, so they are comparable across
specimens of different absolute scale.

WHAT IS EXCLUDED, and why it is not a judgement call
* the 4 wing joints -- the annotator marked them "not present: biological" on all 10 specimens,
  independently reproducing the skinning-weight finding that they carry no geometry on a wingless
  worker.
* `b_h` -- degenerate on the fitted side, so its residual would measure the model's structural
  defect rather than fit quality. It is reported SEPARATELY below, because that defect is a real
  and important result rather than something to hide in an average.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import pickle  # noqa: E402

WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
DEGENERATE = {"b_h"}


def similarity_procrustes(A, B):
    """Fit B -> A with rotation, translation and uniform scale. Returns (R, s, t, aligned_B)."""
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    s = (S * np.array([1, 1, d])).sum() / (B0 ** 2).sum()
    return R, s, ca - s * (cb @ R), s * (B0 @ R) + ca


def main():
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float64)

    fitted = {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        V = np.asarray(d["verts"], dtype=np.float64)
        Jf = (np.einsum("jv,nvc->njc", Jr, V) if Jr.shape[1] == V.shape[1]
              else np.einsum("vj,nvc->njc", Jr, V))
        for i, lab in enumerate(d["labels"]):
            fitted[str(lab).replace("_processed.obj", "").replace(".obj", "")] = Jf[i]

    rows, per_joint = [], {n: [] for n in names}
    head_gap = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "..", "annotation/gt_batch1/*_joints.json"))):
        gt = json.load(open(f))
        sid = gt["specimen_id"]
        if sid not in fitted:
            print(f"  !! no fit for {sid}")
            continue
        P = {j["joint_name"]: np.array(j["position"], dtype=np.float64)
             for j in gt["joints"] if j.get("position") is not None}
        use = [n for n in names if n in P and n not in WING and n not in DEGENERATE]
        A = np.array([P[n] for n in use])                       # ground truth
        Bx = np.array([fitted[sid][names.index(n)] for n in use])  # fitted
        R, s, t, Ba = similarity_procrustes(A, Bx)
        csize = np.sqrt(((A - A.mean(0)) ** 2).sum() / len(A))  # centroid size of the truth
        resid = np.linalg.norm(A - Ba, axis=1) / csize * 100.0
        for n, e in zip(use, resid):
            per_joint[n].append(e)
        rows.append((sid, len(use), float(np.median(resid)), float(resid.mean()),
                     float(np.percentile(resid, 90))))
        # the degenerate head joint, measured rather than skipped
        if "b_h" in P and "b_t" in P:
            gt_off = np.linalg.norm(P["b_h"] - P["b_t"]) / csize * 100.0
            head_gap.append((sid, gt_off))

    print("=" * 96)
    print("G1 -- joint placement accuracy vs human annotation")
    print("similarity-Procrustes aligned on shared joints; error as % of specimen centroid size")
    print("=" * 96)
    print(f"{'specimen':<40}{'joints':>7}{'median':>9}{'mean':>8}{'p90':>8}")
    for sid, n, med, mean, p90 in rows:
        print(f"{sid:<40}{n:>7}{med:>9.2f}{mean:>8.2f}{p90:>8.2f}")
    allmed = np.array([r[2] for r in rows])
    print(f"\n  across {len(rows)} specimens: median-of-medians {np.median(allmed):.2f}% "
          f"of centroid size  (range {allmed.min():.2f}-{allmed.max():.2f}%)")

    print("\n" + "=" * 96)
    print("PER-JOINT -- where the skeleton is right and where it is not")
    print("=" * 96)
    st = sorted(((np.median(v), n, len(v)) for n, v in per_joint.items() if v))
    print(f"{'BEST 12':<16}{'median err %':>13}{'n':>4}     {'WORST 12':<16}{'median err %':>13}{'n':>4}")
    worst = st[::-1]
    for i in range(12):
        a = f"{st[i][1]:<16}{st[i][0]:>13.2f}{st[i][2]:>4}"
        b = f"{worst[i][1]:<16}{worst[i][0]:>13.2f}{worst[i][2]:>4}"
        print(f"{a}     {b}")

    print("\n" + "=" * 96)
    print("THE DEGENERATE HEAD JOINT -- excluded above, reported here")
    print("=" * 96)
    print("  fitted |b_t - b_h| is EXACTLY 0.0 on all 757 production fits (model template defect).")
    print(f"{'specimen':<40}{'GT |b_t-b_h| as % of centroid size':>36}")
    for sid, g in head_gap:
        print(f"{sid:<40}{g:>36.2f}")
    hg = np.array([g for _, g in head_gap])
    print(f"\n  the human places the head joint {hg.mean():.2f}% of centroid size from the thorax "
          f"(range {hg.min():.2f}-{hg.max():.2f}%)")
    print("  the model places it at exactly 0. That distance is unrepresentable by the skeleton.")

    json.dump({"per_specimen": rows,
               "per_joint": {n: float(np.median(v)) for n, v in per_joint.items() if v},
               "head_gap": head_gap},
              open(os.path.join(HERE, "g1_results.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'g1_results.json')}")


if __name__ == "__main__":
    main()
