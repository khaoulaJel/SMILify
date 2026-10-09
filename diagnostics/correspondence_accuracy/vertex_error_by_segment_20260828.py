"""Is the coxal leg-error a CORRESPONDENCE failure or a MODEL-EXPRESSIVENESS failure?

leg_error_by_segment_20260828.py located the addressable leg-level residual: co 57.8% + tr 27.7%
= 85.5% of it, against a MEASURED paired ceiling (GT mesh substituted for the fit) of co 0.095 --
so it is addressable, not irreducible adjacency overlap.

Two explanations remain, and they call for opposite fixes:
  (A) correspondence: the fitter is given the wrong/weak target on coxal vertices.
  (B) expressiveness: the fitter is given the right target and CANNOT REACH IT, because coxa
      placement is fixed by the joint-regressor / thorax shape rather than by anything pose or
      the dense term can move.

The synthetic corpus is topology-identical to the template, so fitted vertex i and ground-truth
vertex i are the SAME anatomical point: ||fitted_i - gt_i|| is an exact per-vertex fitting error,
no correspondence assumption involved. Under (B) coxal vertex error stays high while distal
vertex error falls; under (A) coxal vertex error falls like everything else and the leg-level
error must come from somewhere other than placement.

Run: python diagnostics/correspondence_accuracy/vertex_error_by_segment_20260828.py
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

ARMS = ["P48_zero", "P48_cse_all", "C13_uniform", "C13_invtol", "C13_invtol2"]
CORPUS = "synth_power48"


def main():
    M = ms.load_model()
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    vseg = vlabels["leg_seg"]
    segs = list(lb.LEG_SEGMENTS)
    seg_masks = {s: (vseg == s) for s in segs}
    for s in segs:
        print(f"  template vertices in {s}: {int(seg_masks[s].sum())}")

    err = {a: {s: [] for s in segs} for a in ARMS}
    scale = []
    for a in ARMS:
        d = np.load(os.path.join(MOON, "runs", a, "Stage_3_deform_fine.npz"))
        labs = [str(x) for x in d["labels"]]
        fitted = d["verts"].astype(np.float64)
        for i, lab in enumerate(labs):
            stem = lab[:-4] if lab.endswith(".obj") else lab
            ov, _, _ = load_obj(os.path.join(MOON, CORPUS, f"{stem}.obj"), load_textures=False)
            gt = ov.numpy().astype(np.float64)
            dist = np.linalg.norm(fitted[i] - gt, axis=1)
            # body scale for this specimen: bounding-box diagonal of the GT mesh, so errors are
            # comparable across specimens and reported as a fraction of body size
            bs = float(np.linalg.norm(gt.max(0) - gt.min(0)))
            if a == ARMS[0]:
                scale.append(bs)
            for s in segs:
                err[a][s].append(float(np.mean(dist[seg_masks[s]])) / bs)

    print(f"\nper-vertex fitted-vs-truth error, fraction of body bbox diagonal (n=48)")
    print(f"{'segment':<8}" + "".join(f"{a:>14}" for a in ARMS))
    rows = []
    for s in segs:
        vals = {a: float(np.mean(err[a][s])) for a in ARMS}
        rows.append(dict(segment=s, **vals))
        print(f"{s:<8}" + "".join(f"{vals[a]:>14.4f}" for a in ARMS))

    op = os.path.join(HERE, "out", "vertex_error_by_segment_20260828.json")
    with open(op, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
