"""Z4 -- is `deform_verts` anatomy the shape space cannot express, or per-specimen noise?

WHY THIS IS THE QUESTION
After Z1-Z3 the whole remaining gap on WORKERS is the free-form channel. On the shipped pipeline
(bench50_clean, 50 real workers), leave-one-out gen@20/spread by channel:

    betas alone         0.2641   <- transfers well
    deform_verts alone  0.9881   <- does not transfer at all
    composed            0.6147

Four parameterisation hypotheses are closed (X1 scale magnitude, Z2 rotation freedom, Z3 joint
blendshapes, Z1 the stale worker anchor). So the question is no longer HOW shape is stored, it is
WHAT `deform_verts` contains:

  ANATOMY -> the 25-D space is under-capacity for workers; rebuilding it with more modes from
             worker registrations is licensed, and this measures how much is recoverable.
  NOISE   -> it cannot transfer by construction, gen/spread's 0.00 floor is unreachable on real
             scans, and the target must be re-derived rather than chased.

THE TEST, AND ITS BUILT-IN NULL
Ants are bilaterally symmetric, so each specimen is its own replicate. The template's mirror
involution is exact, so `deform_verts` can be split into

    S = (D + M(D)) / 2   symmetric      A = (D - M(D)) / 2   antisymmetric

where M mirrors in y and re-indexes through the involution. The two subspaces have EQUAL
dimension, so an isotropic random field lands exactly half its energy in each: **the noise null is
0.500, by construction, not by estimation.** Anatomy is bilateral and lands in S.

NOTHING IMPOSES THIS. `symmetry_penalty` (trainer_moonshot.py:150) acts on `log_beta_scales` and
`betas_trans` only; `midline_penalty` acts on the declared midsagittal vertices' y coordinate.
Neither touches `deform_verts`, so bilateral agreement here is measured, not enforced.

Then the part that decides the next project: does S TRANSFER? gen@20/spread is computed on S and
on A separately. A symmetric part that transfers is anatomy the shape space is missing.
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "pose_causality"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from y1_shape_space_analysis import KS, loo_gen_over_spread, rest_space_shaped  # noqa: E402

# every local run whose specimens are real workers and whose shape space is open
RUNS = ["Z3_A_off", "Z2_A_stock", "bench50_G1_learned", "bench50_G3_zero"]


def involution(v_template):
    """Template's own mirror map: mirror in y, match back by nearest neighbour.

    Taken from build_shape_space.py, which is explicit that the map must come from the TEMPLATE
    and not from a fitted specimen (the latter conflates the mirror with fit error, 7.68e-02 vs
    the template's true 0). cKDTree here instead of pytorch3d.knn_points -- CPU, no GPU needed.
    """
    vm = v_template.copy()
    vm[:, 1] *= -1
    idx = cKDTree(v_template).query(vm, k=1)[1]
    md = float(np.linalg.norm(v_template[idx] - vm, axis=1).max())
    exact = float((idx[idx] == np.arange(len(idx))).mean())
    return idx, md, exact


def main():
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    vt = np.asarray(dd["v_template"], dtype=np.float64)
    shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)

    idx, md, exact = involution(vt)
    print("=" * 100)
    print("[VOIDING] the template's mirror involution")
    print(f"  max match distance {md:.2e}   involution exact on {100*exact:.1f}% of vertices")
    ok = md < 1e-6 and exact > 0.99
    print(f"  -> {'PASS' if ok else 'FAIL -- VOID: the mirror map is not clean enough to split on'}")
    assert ok

    mirror_sign = np.array([1.0, -1.0, 1.0])
    out = {}

    print("\n" + "=" * 100)
    print("SYMMETRIC SHARE OF THE FREE-FORM FIELD    (noise null = 0.500 exactly, by construction)")
    print("=" * 100)
    print(f"{'run':<22}{'n':>4}{'sym share':>12}{'|S| rms':>11}{'|A| rms':>11}{'vs null':>10}")
    for run in RUNS:
        p = os.path.join(REPO, "diagnostics/moonshot/runs", run, "Stage_3_deform_fine.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        D = np.asarray(d["deform_verts"], dtype=np.float64)      # (n, V, 3)
        MD = D[:, idx, :] * mirror_sign                          # mirrored counterpart
        S, A = 0.5 * (D + MD), 0.5 * (D - MD)
        es, ea = float((S ** 2).sum()), float((A ** 2).sum())
        share = es / (es + ea)
        out[run] = {"sym_share": share, "n": int(len(D))}
        print(f"{run:<22}{len(D):>4}{share:>12.4f}"
              f"{np.sqrt((S**2).sum(2)).mean():>11.5f}{np.sqrt((A**2).sum(2)).mean():>11.5f}"
              f"{share-0.5:>+10.4f}")

    print("\n" + "=" * 100)
    print("DOES THE SYMMETRIC PART TRANSFER?   gen@20/spread, leave-one-out")
    print("floor under exact correspondence = 0.00 at k=20; 1.0 = no better than the mean")
    print("=" * 100)
    run = "Z3_A_off" if os.path.exists(
        os.path.join(REPO, "diagnostics/moonshot/runs/Z3_A_off/Stage_3_deform_fine.npz")) else RUNS[1]
    d = np.load(os.path.join(REPO, "diagnostics/moonshot/runs", run, "Stage_3_deform_fine.npz"))
    D = np.asarray(d["deform_verts"], dtype=np.float64)
    betas = np.asarray(d["betas"], dtype=np.float64)
    n = len(D)
    MD = D[:, idx, :] * mirror_sign
    S, A = 0.5 * (D + MD), 0.5 * (D - MD)
    P = rest_space_shaped(vt, shapedirs, betas).reshape(n, -1)

    print(f"({run}, n={n} real workers)\n")
    print(f"{'component':<34}{'gen@20/spread':>15}{'spread':>12}")
    for nm, X in [("betas only (parametric)", P),
                  ("deform_verts, full", D.reshape(n, -1)),
                  ("  -> symmetric part S", S.reshape(n, -1)),
                  ("  -> antisymmetric part A", A.reshape(n, -1)),
                  ("betas + S (drop A)", P + S.reshape(n, -1)),
                  ("betas + deform (as shipped)", P + D.reshape(n, -1))]:
        sp, gen, rat = loo_gen_over_spread(X, KS)
        print(f"{nm:<34}{rat[20]:>15.4f}{sp:>12.5f}")
        out.setdefault("gen20", {})[nm.strip()] = rat[20]

    od = os.path.join(REPO, "diagnostics/deform_nature/out_Z4")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "z4_results.json"), "w"), indent=2)
    print(f"\nwrote {od}/z4_results.json")


if __name__ == "__main__":
    main()
