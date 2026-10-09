"""Verify a candidate metric on toy geometry with a KNOWN answer, before pre-registering a threshold.

WHY THIS EXISTS
---------------
Twice in one session (2026-08-27) a pre-registered gate measured a real quantity that was not the
decision-relevant one:

  1. `S1` for confidence signals used global Spearman correlation. Cosine similarity scored the BEST
     rank correlation (-0.361) and yet filtering by it made retrieval error 44% WORSE -- because the
     relationship is non-monotonic and only the HEAD of the ranking matters for filtering.
  2. The LRF check gated on `(l1-l2)/l1`, the LIMB-AXIS gap. On a cylinder that is large precisely
     because the neighbourhood is elongated along the limb -- it confirms something never in doubt.
     The decisive quantity spans the circumferential plane -- and this toy check then caught a
     THIRD error before it reached real data: `(l2-l3)/l1` scored a clearly disambiguable 4:1
     elliptical tube at 0.056, BELOW the 0.15 gate, because dividing by l1 crushes the scale for any
     elongated neighbourhood. The correct form compares the two circumferential eigenvalues to EACH
     OTHER: `(l2-l3)/l2`, which scores that tube 0.937 vs a cylinder's 0.031.

Both are the same failure: choosing a metric that is ABOUT the right phenomenon without checking it
DISCRIMINATES the cases the decision depends on. The general fix is cheap -- construct toy cases
whose correct answer is known by construction, and confirm the metric separates them in the expected
direction and by a useful margin, BEFORE applying it to real data.

Run this, or an equivalent, for every new threshold. It costs seconds.
"""
import numpy as np


def toy_clouds(n=4000, seed=0):
    """Shapes whose eigenstructure is known analytically."""
    rng = np.random.default_rng(seed)
    out = {}

    # perfect cylinder: circumferential plane is EXACTLY degenerate (l2 == l3 by symmetry)
    t = rng.uniform(-1, 1, n)
    a = rng.uniform(0, 2 * np.pi, n)
    out["cylinder_r0.15"] = np.stack([t, 0.15 * np.cos(a), 0.15 * np.sin(a)], 1)
    out["cylinder_r0.05"] = np.stack([t, 0.05 * np.cos(a), 0.05 * np.sin(a)], 1)

    # flat plane: normal is well-defined, in-plane axes are degenerate with each other
    out["plane"] = np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), np.zeros(n)], 1)

    # elliptical tube: circumferential plane is NOT degenerate -- the metric MUST see this,
    # otherwise it is just detecting "is elongated" rather than "is rotationally ambiguous"
    out["elliptic_tube"] = np.stack([t, 0.20 * np.cos(a), 0.05 * np.sin(a)], 1)

    # isotropic blob: nothing is degenerate, nothing is elongated
    out["blob"] = rng.normal(0, 0.3, (n, 3))
    return out


def gaps(P):
    w = np.linalg.eigvalsh(np.cov((P - P.mean(0)).T))[::-1]
    return dict(axis_gap=(w[0] - w[1]) / w[0],
                circum_gap_L1=(w[1] - w[2]) / w[0],   # REJECTED by this toy check, see below
                circum_gap=(w[1] - w[2]) / max(w[1], 1e-18),
                scatter=w[2] / w[0])


def main():
    print("Toy verification of the LRF conditioning metrics.\n")
    print("Known answers by construction:")
    print("  cylinder      -> circumferential plane EXACTLY degenerate  => circum_gap ~ 0")
    print("  elliptic_tube -> elongated BUT not rotationally ambiguous  => circum_gap clearly > 0")
    print("  plane         -> normal well-defined, in-plane degenerate  => circum_gap large")
    print("  blob          -> nothing degenerate                        => both gaps small\n")
    print(f"{'shape':>16} | {'axis_gap':>9} {'/l1 (bad)':>10} {'circum_gap':>11} {'scatter':>8}")
    res = {}
    for name, P in toy_clouds().items():
        g = gaps(P)
        res[name] = g
        print(f"{name:>16} | {g['axis_gap']:>9.3f} {g['circum_gap_L1']:>10.3f} "
              f"{g['circum_gap']:>11.3f} {g['scatter']:>8.3f}")

    print("\nCHECKS")
    ok = True

    c = res["cylinder_r0.15"]["circum_gap"]
    t = res["elliptic_tube"]["circum_gap"]
    a_cyl = res["cylinder_r0.15"]["axis_gap"]

    c1 = c < 0.10
    print(f"  [{'PASS' if c1 else 'FAIL'}] cylinder circum_gap ~ 0            : {c:.4f} < 0.10")
    print("         (0.031 is the FINITE-SAMPLE floor at n=4000, not 0 -- an earlier 0.02 tolerance\n"
          "          was unrealistic. This is the empirical noise floor the real-data threshold\n"
          "          must sit well above.)")
    ok &= c1

    c2 = t > 0.50 and t > 5 * c
    print(f"  [{'PASS' if c2 else 'FAIL'}] elliptic tube separates from cyl  : {t:.4f} vs {c:.4f}")
    print(f"         (with the REJECTED /l1 form these were "
          f"{res['elliptic_tube']['circum_gap_L1']:.4f} vs {res['cylinder_r0.15']['circum_gap_L1']:.4f}"
          f" -- a clearly disambiguable 4:1 tube fell BELOW the 0.15 gate, because dividing by l1"
          f" crushes the scale for any elongated neighbourhood)")
    ok &= c2

    c3 = a_cyl > 0.5
    print(f"  [{'PASS' if c3 else 'FAIL'}] axis_gap LARGE on a cylinder      : {a_cyl:.4f} > 0.5")
    print("         ^ this is exactly why axis_gap was the WRONG gate: it is large on the very")
    print("           shape whose circumferential frame is degenerate.")
    ok &= c3

    c4 = res["plane"]["circum_gap"] > 0.5
    print(f"  [{'PASS' if c4 else 'FAIL'}] plane circum_gap large            : {res['plane']['circum_gap']:.4f} > 0.5")
    ok &= c4

    print(f"\nMETRIC VERIFIED: {ok}. "
          f"circum_gap=(l2-l3)/l2 discriminates rotational ambiguity; axis_gap=(l1-l2)/l1 does not.")
    print("Only after this should the threshold be applied to real geometry.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
