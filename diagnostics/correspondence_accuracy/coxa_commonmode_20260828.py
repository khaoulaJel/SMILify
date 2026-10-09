"""C13 FAILED: coxal placement is inert to the dense term at ANY weight (0.0201 -> 0.0200 across
a 91x sweep, while every other segment moved monotonically). So coxal position is set by
parameters the correspondence term cannot steer. Which ones?

The coxa articulation is placed by the joint regressor from the shape parameters, so a coxal
offset is a SHAPE error, not a pose error. Two sub-cases with very different fixes:

  COMMON-MODE: all six coxae are displaced in a consistent direction in the body frame (e.g. all
    outward -> thorax too narrow, all forward/back -> thorax too short). That is a handful of
    global shape degrees of freedom and is very tractable: it means the thorax betas are
    mis-estimated in a specific, nameable way.
  PER-LEG: each coxa is off independently. That is 6x3 unconstrained degrees of freedom and
    implies the joint regressor itself cannot place coxae for these specimens.

Offsets are expressed in each specimen's own PCA body frame (axis 0 = long axis, axis 1 = lateral,
axis 2 = dorsoventral), and the LATERAL component of left-side legs is sign-flipped so a
symmetric "all coxae too far out" bias ADDS across sides instead of cancelling.

Run: python diagnostics/correspondence_accuracy/coxa_commonmode_20260828.py
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

ARM = "C13_uniform"
CORPUS = "synth_power48"


def main():
    M = ms.load_model()
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    vseg, vleg = vlabels["leg_seg"], vlabels["leg_id"]
    legs = list(lb.LEGS)

    d = np.load(os.path.join(MOON, "runs", ARM, "Stage_3_deform_fine.npz"))
    labs = [str(x) for x in d["labels"]]
    fitted = d["verts"].astype(np.float64)

    per_leg_off = {lg: [] for lg in legs}   # offset in body frame, lateral sign-normalised
    # CONTROL: is the coxal shift merely inherited from a global body-placement bias? If the
    # thorax/body is itself shifted along the long axis by the same amount, the coxa is not
    # independently misplaced at all and the finding is about global registration, not shape.
    body_off, coxa_raw, spec_common, spec_perleg = [], [], [], []
    for i, lab in enumerate(labs):
        stem = lab[:-4] if lab.endswith(".obj") else lab
        ov, _, _ = load_obj(os.path.join(MOON, CORPUS, f"{stem}.obj"), load_textures=False)
        gt = ov.numpy().astype(np.float64)
        bs = float(np.linalg.norm(gt.max(0) - gt.min(0)))

        # body frame from the GT mesh's own principal axes
        c = gt.mean(0)
        _, _, vt = np.linalg.svd(gt - c, full_matrices=False)
        F = vt[:3]                                    # rows = axes, long/lateral/dorsoventral

        bm = (vseg == None) & (vleg == None)  # noqa: E711  -- non-leg (body) vertices
        body_off.append(F @ (((fitted[i][bm] - gt[bm]).mean(0)) / bs))

        this = []
        for lg in legs:
            m = (vseg == "co") & (vleg == lg)
            if not m.any():
                continue
            off = ((fitted[i][m] - gt[m]).mean(0)) / bs
            o = F @ off                               # into body frame
            if lg.endswith("_l"):
                o = o * np.array([1.0, -1.0, 1.0])    # mirror lateral so sides are comparable
            per_leg_off[lg].append(o)
            this.append(o)
        this = np.array(this)
        mu_s = this.mean(0)
        spec_common.append(float(np.sum(mu_s ** 2)))
        spec_perleg.append(float(np.mean(np.sum((this - mu_s) ** 2, axis=1))))

    print(f"\nmean coxa offset in body frame, fraction of body diagonal ({ARM}, n={len(labs)})")
    print(f"{'leg':<8}{'long':>9}{'lateral':>9}{'dorsov':>9}{'|mean|':>9}{'mean|.|':>9}")
    means = {}
    for lg in legs:
        A = np.array(per_leg_off[lg])
        mu = A.mean(0)
        means[lg] = mu
        print(f"{lg:<8}{mu[0]:>+9.4f}{mu[1]:>+9.4f}{mu[2]:>+9.4f}"
              f"{np.linalg.norm(mu):>9.4f}{np.mean(np.linalg.norm(A, axis=1)):>9.4f}")

    allA = np.concatenate([np.array(per_leg_off[lg]) for lg in legs])
    gmu = allA.mean(0)
    tot = float(np.mean(np.sum(allA ** 2, axis=1)))
    common = float(np.sum(gmu ** 2))
    # variance attributable to a per-leg (but specimen-independent) mean, beyond the global mean
    perleg = float(np.mean([np.sum((means[lg] - gmu) ** 2) for lg in legs]))
    print(f"\n  global mean offset (all coxae): long {gmu[0]:+.4f}  lateral {gmu[1]:+.4f}  "
          f"dorsoventral {gmu[2]:+.4f}   |{np.linalg.norm(gmu):.4f}|")
    print(f"  mean squared offset            {tot:.6f}  (100%)")
    print(f"   .. common-mode (global mean)  {common:.6f}  ({common / tot:.1%})")
    print(f"   .. per-leg systematic         {perleg:.6f}  ({perleg / tot:.1%})")
    print(f"   .. residual (per-specimen)    {tot - common - perleg:.6f} "
          f"({(tot - common - perleg) / tot:.1%})")

    B = np.array(body_off)
    bmu = B.mean(0)
    print(f"\n  CONTROL -- body (non-leg) mean offset, same frame:")
    print(f"    long {bmu[0]:+.4f}  lateral {bmu[1]:+.4f}  dorsoventral {bmu[2]:+.4f}"
          f"   |{np.linalg.norm(bmu):.4f}|   mean|.| {np.mean(np.linalg.norm(B, axis=1)):.4f}")
    print(f"    coxa long-axis shift MINUS body long-axis shift = "
          f"{gmu[0] - bmu[0]:+.4f}  (0 => wholly inherited from global body placement)")
    print(f"\n  WITHIN-SPECIMEN split of the coxal offset:")
    print(f"    common to all 6 coxae of a specimen  {np.mean(spec_common):.6f} "
          f"({np.mean(spec_common) / tot:.1%})")
    print(f"    scatter between coxae of a specimen  {np.mean(spec_perleg):.6f} "
          f"({np.mean(spec_perleg) / tot:.1%})")

    op = os.path.join(HERE, "out", "coxa_commonmode_20260828.json")
    with open(op, "w") as f:
        json.dump(dict(arm=ARM, per_leg_mean={k: v.tolist() for k, v in means.items()},
                       global_mean=gmu.tolist(), mean_sq=tot, common_mode=common,
                       per_leg_systematic=perleg), f, indent=2)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
