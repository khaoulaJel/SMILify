"""Is coxal placement error a RIGID OFFSET or a DEFORMATION?

interleg_tolerance_20260828.py showed the coxa fits at risk = placement_error/inter-leg_tolerance
= 0.995, against ~0.21 for every other segment -- which is why co carries 57.8% of the addressable
leg-level residual despite being the best-placed segment in absolute terms.

That says the coxa needs ~5x more PRECISION than anywhere else, not more correspondence. Two
sub-cases, with different fixes:
  (i)  RIGID: the whole coxa is displaced as a unit -> the coxa's articulation point is in the
       wrong place, which is set by the joint regressor / betas, not by the dense term. Fix is
       shape/joint-location, and re-weighting the loss will not help.
  (ii) DEFORM: the coxa centroid is right and the error is shape/spread -> the dense term can
       reach it, and it is currently under-weighted because the L2 gradient is proportional to
       the residual, which is SMALLEST exactly where tolerance is tightest.

Decomposes per-(specimen, leg) coxal error into the centroid offset and the residual about it.

Run: python diagnostics/correspondence_accuracy/coxa_error_decomposition_20260828.py
"""

import json
import os
import sys

import numpy as np
from pytorch3d.io import load_obj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
import labels as lb  # noqa: E402

CORPUS = "synth_power48"


def main():
    M = ms.load_model()
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    vseg, vleg = vlabels["leg_seg"], vlabels["leg_id"]
    segs = list(lb.LEG_SEGMENTS)

    out = {}
    for arm in ["P48_zero", "P48_cse_all"]:
        d = np.load(os.path.join(MOON, "runs", arm, "Stage_3_deform_fine.npz"))
        labs = [str(x) for x in d["labels"]]
        fitted = d["verts"].astype(np.float64)
        acc = {s: {"total": [], "rigid": [], "deform": []} for s in segs}
        for i, lab in enumerate(labs):
            stem = lab[:-4] if lab.endswith(".obj") else lab
            ov, _, _ = load_obj(os.path.join(MOON, CORPUS, f"{stem}.obj"), load_textures=False)
            gt = ov.numpy().astype(np.float64)
            bs = float(np.linalg.norm(gt.max(0) - gt.min(0)))
            for s in segs:
                for leg in lb.LEGS:
                    m = (vseg == s) & (vleg == leg)
                    if not m.any():
                        continue
                    diff = (fitted[i][m] - gt[m]) / bs
                    off = diff.mean(0)
                    acc[s]["total"].append(float(np.mean(np.linalg.norm(diff, axis=1))))
                    acc[s]["rigid"].append(float(np.linalg.norm(off)))
                    acc[s]["deform"].append(float(np.mean(np.linalg.norm(diff - off, axis=1))))
        out[arm] = acc

    print(f"\nper-(specimen,leg) error decomposition, fraction of body bbox diagonal")
    print(f"{'segment':<8}{'arm':<14}{'total':>9}{'rigid':>9}{'deform':>9}{'rigid_share':>13}")
    rows = []
    for s in segs:
        for arm in ["P48_zero", "P48_cse_all"]:
            a = out[arm][s]
            t, r, df = (float(np.mean(a[k])) for k in ("total", "rigid", "deform"))
            rows.append(dict(segment=s, arm=arm, total=t, rigid=r, deform=df, rigid_share=r / t))
            print(f"{s:<8}{arm:<14}{t:>9.4f}{r:>9.4f}{df:>9.4f}{r / t:>13.1%}")

    op = os.path.join(HERE, "out", "coxa_error_decomposition_20260828.json")
    with open(op, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
