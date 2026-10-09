"""Does INTER-LEG TOLERANCE explain the leg-error pattern better than placement error does?

vertex_error_by_segment_20260828.py found the coxa is the BEST-placed leg segment
(0.0201 of body diagonal vs pt's 0.0397) and yet carries a 40% leg-level error rate, 57.8% of
the whole addressable residual. So coxal leg error is not "the fitter cannot reach the target"
and not "the target is wrong".

Proposed mechanism: leg_acc is decided by nearest-fitted-vertex, so what matters is placement
error RELATIVE TO THE DISTANCE TO THE NEAREST OTHER LEG, not relative to body size. Adjacent
coxae are packed on the thorax; tarsi are far apart. A displacement that is trivial against
body scale can cross the midpoint to a neighbouring coxa.

Measured here on the POSED ground-truth meshes (not the rest-pose template -- legs move):
for every leg vertex, the distance to the nearest vertex belonging to a DIFFERENT leg. The
prediction is a risk ratio (placement error / tolerance) that tracks the leg-error ranking,
where raw placement error anti-tracks it.

Run: python diagnostics/correspondence_accuracy/interleg_tolerance_20260828.py
"""

import json
import os
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import knn_points

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
N_SPEC = 48


def main():
    M = ms.load_model()
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    vseg, vleg = vlabels["leg_seg"], vlabels["leg_id"]
    segs = list(lb.LEG_SEGMENTS)
    is_leg = vleg != None  # noqa: E711

    d = np.load(os.path.join(MOON, "runs", "P48_zero", "Stage_3_deform_fine.npz"))
    labs = [str(x) for x in d["labels"]][:N_SPEC]

    tol = {s: [] for s in segs}
    for lab in labs:
        stem = lab[:-4] if lab.endswith(".obj") else lab
        ov, _, _ = load_obj(os.path.join(MOON, CORPUS, f"{stem}.obj"), load_textures=False)
        gt = ov.numpy().astype(np.float64)
        bs = float(np.linalg.norm(gt.max(0) - gt.min(0)))
        gt_t = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)
        for leg in lb.LEGS:
            src = is_leg & (vleg == leg)          # this leg's vertices
            other = is_leg & (vleg != leg)        # every OTHER leg's vertices
            if not src.any() or not other.any():
                continue
            q = gt_t[:, src, :]
            k = gt_t[:, other, :]
            dist = knn_points(q, k, K=1).dists[0, :, 0].sqrt().numpy()
            seg_here = vseg[src]
            for s in segs:
                m = seg_here == s
                if m.any():
                    tol[s].append(float(np.mean(dist[m])) / bs)

    with open(os.path.join(HERE, "out", "vertex_error_by_segment_20260828.json")) as f:
        verr = {r["segment"]: r for r in json.load(f)}
    with open(os.path.join(HERE, "out", "leg_error_by_segment_20260828.json")) as f:
        lerr = {r["segment"]: r for r in json.load(f)}

    print(f"\nall quantities as a fraction of body bbox diagonal (n={len(labs)} specimens)")
    print(f"{'segment':<8}{'tolerance':>11}{'place_err':>11}{'risk':>8}"
          f"{'leg_err':>10}{'ceiling':>10}")
    rows = []
    for s in segs:
        t = float(np.mean(tol[s]))
        p = verr[s]["cse"]
        risk = p / t
        rows.append(dict(segment=s, interleg_tolerance=t, placement_error=p, risk_ratio=risk,
                         leg_err_cse=lerr[s]["err_P48_cse_all"], leg_err_ceiling=lerr[s]["err_CEILING"]))
        print(f"{s:<8}{t:>11.4f}{p:>11.4f}{risk:>8.3f}"
              f"{lerr[s]['err_P48_cse_all']:>10.4f}{lerr[s]['err_CEILING']:>10.4f}")

    from scipy import stats
    y = np.array([r["leg_err_cse"] for r in rows])
    for name, x in [("risk ratio", np.array([r["risk_ratio"] for r in rows])),
                    ("placement error alone", np.array([r["placement_error"] for r in rows])),
                    ("tolerance alone", np.array([r["interleg_tolerance"] for r in rows]))]:
        rho, p = stats.spearmanr(x, y)
        print(f"  spearman(leg_err, {name:<22}) = {rho:+.3f}  (p={p:.3g}, n=6)")

    op = os.path.join(HERE, "out", "interleg_tolerance_20260828.json")
    with open(op, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
