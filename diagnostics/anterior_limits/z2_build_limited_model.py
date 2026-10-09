"""Z2 step 1 -- author rotation limits for the 12 unconstrained non-leg joints.

THE GAP (measured, diagnostics/anchor_restage/):
`OmniAnt_25PCs_joint_limited.pkl` constrains 99 of 165 rotation axes. The 66 free ones are the
4 wing joints (unbindable, 0 skin mass -- left free here) and these 12:

    b_a_1 b_a_2 b_a_3 b_a_4 b_a_5   b_h   an_1_r an_2_r an_3_r an_1_l an_2_l an_3_l

`b_h` carries 2180.69 skin mass -- the HIGHEST of any joint in the model -- and all three of its
axes are free. On bench50 the fitted rotations run to a max of 101.1 deg at the head, 174.0 deg
at an_2_l, 135.9 deg at b_a_5. Medians are 2-6 deg on the body axis: this is purely a TAIL, which
is what a hinge is for and what no surface metric can see. Where limits ARE authored the hinge
binds exactly (l_1_fe_r max 70.1 deg against a 70 deg limit).

WHY SYMMETRIC BANDS
docs/joint_limits_user_guide.md is emphatic that per-axis signs depend on each bone's own local
frame and must never be guessed. A symmetric band [-t, +t] is sign-agnostic, so it is the only
form that can be authored without opening Blender. The values below are domain judgement and
deliberately GENEROUS -- each clips roughly the top 1-2% of the fitted distribution and leaves
every median untouched. That makes the test conservative: if a band this loose moves the
mechanism, a properly authored one would move it more; if it does not, the representation gap is
not the mechanism.

These are NOT proposed as the final limits. Final values should be authored per-axis in Blender
by someone who can see the bone frames (docs/joint_limits_user_guide.md).
"""
import argparse
import os
import pickle

import numpy as np

# joint -> symmetric half-width in degrees
BANDS = {
    1: 45.0,    # b_a_1  petiole
    2: 45.0,    # b_a_2  postpetiole
    3: 40.0,    # b_a_3  gaster
    4: 40.0,    # b_a_4  gaster
    5: 45.0,    # b_a_5  gaster tip
    46: 60.0,   # b_h    head
    48: 90.0, 52: 90.0,   # an_1  scape -- genuinely very mobile
    49: 90.0, 53: 90.0,   # an_2  funiculus
    50: 75.0, 54: 75.0,   # an_3
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
    ap.add_argument("--dst", default="3D_model_prep/OmniAnt_25PCs_anterior_limited.pkl")
    args = ap.parse_args()

    with open(args.src, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()

    lim = np.asarray(dd["joint_limits"], dtype=np.float64).copy()
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    free = (lim[:, :, 1] - lim[:, :, 0]) >= 2 * np.pi - 1e-6

    print(f"{'j':>3} {'name':<10}{'was':>12}{'now':>18}")
    n_changed = 0
    for j, half in BANDS.items():
        if not free[j].all():
            raise SystemExit(f"j{j} {names[j]} is already constrained -- refusing to overwrite "
                             "authored limits")
        r = np.radians(half)
        lim[j, :, 0] = -r
        lim[j, :, 1] = +r
        n_changed += 3
        print(f"{j:>3} {names[j]:<10}{'free':>12}{f'+/- {half:.0f} deg':>18}")

    # every other axis must be byte-identical to the source
    lim0 = np.asarray(dd["joint_limits"], dtype=np.float64)
    mask = np.ones(lim.shape[:2], dtype=bool)
    mask[list(BANDS.keys())] = False
    assert np.array_equal(lim[mask], lim0[mask]), "untouched axes changed -- aborting"

    still_free = ((lim[:, :, 1] - lim[:, :, 0]) >= 2 * np.pi - 1e-6).sum()
    print(f"\nconstrained axes: {(~free).sum()} -> {165 - still_free}   (+{n_changed})")
    # 30 remain free: 12 = the 4 wing joints x 3 axes (unbindable, 0 skin mass), and 18 leg
    # axes that were never authored. Legs are deliberately OUT OF SCOPE for Z2 -- the leg axis
    # is the one this project has already closed (W1-W5, C14p), and mixing it in would make a
    # positive result unattributable.
    print(f"still free: {still_free} (expected 30 = 12 wing + 18 never-authored leg axes)")
    assert still_free == 30, f"expected 30 free axes left, got {still_free}"

    dd["joint_limits"] = lim
    with open(args.dst, "wb") as fh:
        pickle.dump(dd, fh, protocol=2)
    print(f"\nwrote {args.dst}")

    # readback: the file on disk must carry what we intended
    with open(args.dst, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        back = u.load()
    lb = np.asarray(back["joint_limits"], dtype=np.float64)
    assert np.allclose(lb, lim), "readback mismatch"
    for k in dd:
        if k == "joint_limits":
            continue
        a, b = np.asarray(dd[k], dtype=object), np.asarray(back[k], dtype=object)
        assert a.shape == b.shape, f"{k} shape changed on write"
    print("readback OK: joint_limits match, every other key shape-identical")


if __name__ == "__main__":
    main()
