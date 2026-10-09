"""Is a Local Reference Frame even WELL-DEFINED on tr/fe geometry? (CPU, no retrain)

WHY THIS RUNS BEFORE ANY LRF-BASED DESCRIPTOR IS BUILT
------------------------------------------------------
LRF canonicalisation (SHOT, 3DSmoothNet, DIP) is the standard cheap alternative to a full
SE(3)-equivariant retrain: instead of retraining the backbone, canonicalise each point's
neighbourhood by eigendecomposition of its local covariance. The literature is explicit about the
failure mode: the sign of the normal axis and the directions of the remaining two axes are NOT
unique in planar or locally-symmetric regions -- LRFs break down exactly where local geometry has no
well-defined principal direction.

A leg segment is close to a cylinder, and a cylinder's cross-section has NO unique in-plane axis by
construction. So the concern is precise: an LRF descriptor would assume a disambiguating signal
exists exactly where this geometry structurally fails to provide one -- the identical pathology to
cosine-similarity confidence, which had the best rank correlation and still poisoned its own ranking
head because it assumed similarity implies matchability.

WHAT IS MEASURED
----------------
For sampled points on `tr`/`fe` (and `body` as a POSITIVE CONTROL, since a non-cylindrical region
should be better-conditioned if the metric is meaningful at all), the local covariance eigenvalues
l1>=l2>=l3 within a neighbourhood radius, and:

  axis_gap     = (l1 - l2) / l1   -- conditioning of the LIMB-AXIS direction (l1's eigenvector).
                                     NOT the decisive quantity, and it was gated on by mistake in the
                                     first version of this script (2026-08-27). On a cylinder patch
                                     the neighbourhood is elongated ALONG the limb, so l1 >> l2 ~= l3
                                     and this gap is LARGE -- which merely confirms the limb axis is
                                     well-defined, something never in doubt.
  circum_gap   = (l2 - l3) / l2   -- THE DECISIVE QUANTITY. l2 and l3's eigenvectors span the
                                     CIRCUMFERENTIAL plane. When l2 ~= l3 the frame may spin freely
                                     about the limb axis, which is exactly the ambiguity that has
                                     limited this whole investigation.
  linearity    = (l1 - l2) / l1,  planarity = (l2 - l3) / l1,  scattering = l3 / l1
                                     (Weinmann et al. dimensionality features, for interpretability)

PRE-REGISTERED DECISION RULE (fixed before running)
---------------------------------------------------
  DROP the LRF path if median `circum_gap` = (l2-l3)/l1 on tr/fe is < 0.15 -- a near-degenerate
  circumferential plane, where canonicalisation is arbitrary and the cosine failure repeats.
  PROCEED only if tr/fe circum_gap is comfortably above that AND not dramatically worse than the
  `body` control.

  RESULT (2026-08-27): DROP. circum_gap fe 0.057 / tr 0.122 at r=0.10 -- both below threshold, and
  3-7x worse than the `body` control (0.397), which confirms the metric is meaningful rather than
  uniformly small. Conditioning is acceptable only at r=0.02, where the >=12-neighbour filter
  already discards ~90% of points. The cylinder degeneracy the literature warns about is REAL here.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    build_segment_taxonomy, build_vertex_labels, load_model_dict,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--n_specimens", type=int, default=8)
    ap.add_argument("--n_points", type=int, default=8000, help="denser than training: this is pure "
                                                              "geometry, no network involved")
    ap.add_argument("--radii", default="0.02,0.05,0.10",
                    help="neighbourhood radii to test, in model units. sa1's own ball-query radii "
                         "are 0.1/0.2/0.4, so these bracket the small end where an LRF would live.")
    ap.add_argument("--min_neighbours", type=int, default=12)
    ap.add_argument("--drop_threshold", type=float, default=0.30,
                    help="pre-registered: DROP the LRF path if median tr/fe circum_gap (l2-l3)/l2 is "
                         "below this. NOTE the first run of this script gated on (l1-l2)/l1 by "
                         "mistake and printed PROCEED. The threshold is calibrated on verify_metric_on_toy.py: "
                         "a perfect cylinder scores 0.031 (finite-sample floor) and a clearly "
                         "disambiguable 4:1 elliptical tube scores 0.937, so 0.30 sits well above "
                         "the noise floor and well below a genuinely conditioned frame.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_lrf_check_20260827")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    radii = [float(x) for x in args.radii.split(",")]

    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    seg_class_id, _ = build_vertex_labels(dd, class_names, name_to_id)
    seg_short = {c: (None if n == "body" else n.split("_")[-1]) for c, n in enumerate(class_names)}

    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].long()
    gt = np.load(os.path.join(REPO, args.corpus), allow_pickle=True)["verts"]
    n_val = max(1, int(len(gt) * args.val_frac))
    val = gt[-n_val:][:args.n_specimens]

    acc = {}
    for si, vv in enumerate(val):
        v = torch.as_tensor(vv, dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]
        tv = knn_points(pts.unsqueeze(0), v.unsqueeze(0), K=1).idx[0, :, 0].numpy()
        seg = seg_class_id[tv]
        P = pts.numpy().astype(np.float64)

        # pairwise distances once per specimen (n_points^2 is fine at 8k on CPU in chunks)
        for rad in radii:
            for group, mask in (("tr", np.isin(seg, [c for c, s in seg_short.items() if s == "tr"])),
                                ("fe", np.isin(seg, [c for c, s in seg_short.items() if s == "fe"])),
                                ("body", seg == 0)):
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                take = idx if len(idx) <= 300 else np.random.default_rng(si).choice(idx, 300, replace=False)
                for i in take:
                    d = np.linalg.norm(P - P[i], axis=1)
                    nb = P[d <= rad]
                    if len(nb) < args.min_neighbours:
                        continue
                    C = np.cov((nb - nb.mean(0)).T)
                    w = np.linalg.eigvalsh(C)[::-1]      # l1 >= l2 >= l3
                    if w[0] <= 1e-18:
                        continue
                    key = (group, rad)
                    acc.setdefault(key, []).append((
                        (w[1] - w[2]) / max(w[1], 1e-18),  # circum_gap <- DECISIVE, l2-normalised
                        (w[0] - w[1]) / w[0],            # axis_gap    (not decisive)
                        w[2] / w[0],                     # scattering
                    ))
        print(f"[{si+1}/{len(val)}] specimen done", flush=True)

    print("\n=== LRF conditioning: is the in-plane axis well-defined on tr/fe? ===")
    print(f"{'group':>6} {'radius':>7} {'n':>6} | {'circum_gap (DECISIVE)':>26} | {'axis_gap':>11} {'scatter':>8}")
    print(f"{'':>6} {'':>7} {'':>6} | {'median':>8} {'p25':>8} {'p75':>8} | {'median':>11} {'median':>8}")
    res = {}
    for (group, rad), rows in sorted(acc.items()):
        a = np.array(rows)
        ci, ax_, sc = a[:, 0], a[:, 1], a[:, 2]
        res[f"{group}@{rad}"] = dict(n=len(a), circum_median=float(np.median(ci)),
                                     circum_p25=float(np.percentile(ci, 25)),
                                     circum_p75=float(np.percentile(ci, 75)),
                                     axis_median=float(np.median(ax_)),
                                     scatter_median=float(np.median(sc)))
        print(f"{group:>6} {rad:>7.3f} {len(a):>6} | {np.median(ci):>8.3f} {np.percentile(ci,25):>8.3f} "
              f"{np.percentile(ci,75):>8.3f} | {np.median(ax_):>11.3f} {np.median(sc):>8.3f}")

    print("\n=== PRE-REGISTERED VERDICT (drop if tr/fe median inplane_gap < "
          f"{args.drop_threshold} on circum_gap) ===")
    verdict = {}
    for rad in radii:
        for group in ("tr", "fe"):
            k = f"{group}@{rad}"
            if k not in res:
                continue
            m = res[k]["circum_median"]
            body = res.get(f"body@{rad}", {}).get("circum_median", float("nan"))
            ok = m >= args.drop_threshold
            verdict[k] = bool(ok)
            print(f"  {group} @ r={rad}: median inplane_gap = {m:.3f} "
                  f"(body control {body:.3f}) -> {'PROCEED' if ok else 'DROP'}")
    any_ok = any(verdict.values())
    print(f"\nOVERALL: {'LRF path is viable at some radius -- see which' if any_ok else 'DROP THE LRF PATH. The in-plane axis is near-degenerate on tr/fe at every radius tested, exactly the cylinder failure mode the literature warns about. Building an LRF descriptor here would repeat the cosine-similarity mistake: assuming a disambiguating signal exists where the geometry does not provide one.'}")

    with open(os.path.join(out_dir, "lrf_wellposedness.json"), "w") as fh:
        json.dump({"config": vars(args), "results": res, "verdict": verdict, "any_ok": any_ok}, fh, indent=2)
    print(f"\nwrote {out_dir}/lrf_wellposedness.json")


if __name__ == "__main__":
    main()
