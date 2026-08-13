"""Execution plan §4.5 — landmark-restricted correspondence scoring.

within_part_signal.py scores correctness over the WHOLE mesh (~10k vertices) or over whole
anatomical part groups. That is a materially different, harder question than what
morphometrics actually needs: measure.py never reads a whole part's dense correspondence, it
reads JOINT POSITIONS (`J_regressor @ verts`, a weighted average over each joint's own
skinning-dominant vertex set) and whole-part rigid-aligned extents. This probe asks the
narrower, cheaper question the plan names directly: is correspondence recovered well enough on
the specific vertices morphometric measurements actually depend on, independent of whether it
is recovered on the mesh as a whole (already answered, and closed, in FINAL_REPORT.md/E7).

LANDMARK DEFINITION, stated explicitly because it is a modelling choice, not a given: for each
joint j (55 joints in the template's J_regressor), the landmark vertex is
`argmax_v weights[v, j]` -- the single vertex most strongly skinned to that joint. This is a
discrete proxy for the joint-CENTER read-out measure.py/probe_23b/24d actually use (a weighted
average over many vertices, not one point) -- a genuine simplification, made because
within_part_signal.py's scoring metric (`nearest-descriptor-neighbour equals the SAME vertex`)
is only defined for discrete points, not weighted read-outs. Restricted further to leg-tip and
antenna-tip joints specifically (per the plan's own phrasing) by name-matching the template's
J_names: `l_*_pt`/`l_*_ta` (pretarsus/tarsus, the leg's distal-most joints) and `an_3` (the
antenna's terminal segment), plus the FULL 55-joint set for the "joint centers" reading.

This is a fork, not an edit, of within_part_signal.py -- imports its hks()/cotangent_laplacian()
unchanged, reuses its exact scoring protocol (`score()`), and adds one thing: masking the
comparison down to the landmark vertex set instead of all vertices / whole part groups.
"""

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from scipy.spatial import cKDTree  # noqa: E402

from within_part_signal import cotangent_laplacian, hks  # noqa: E402,F401 (re-exported for reuse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--n_eig", type=int, default=120)
    ap.add_argument("--run", default="AB_A_SYN", help="fitted run to score as the 'fitted' baseline; skipped if absent")
    args = ap.parse_args()

    import config

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [str(x) for x in dd["J_names"]]
    weights = np.asarray(dd["weights"])
    if hasattr(weights, "toarray"):
        weights = weights.toarray()
    v_tpl = np.asarray(dd["v_template"], dtype=np.float64)
    faces = np.asarray(dd["f"]).astype(np.int64)
    V = len(v_tpl)

    # one landmark vertex per joint: the vertex most strongly skinned to it
    joint_landmark = weights.argmax(0)  # (n_joints,) vertex index per joint
    landmark_all = np.unique(joint_landmark)

    # joint names carry a bilateral _r/_l suffix (l_1_pt_r, an_3_l, ...), not a bare suffix --
    # confirmed by reading dd["J_names"] directly (2026-08-13), not assumed from other scripts'
    # naming conventions, which is exactly what the first version of this filter got wrong
    # (matched zero joints, crashed the job on an empty landmark set).
    is_tip = np.array([("_pt_" in n or "_ta_" in n or n.startswith("an_3")) for n in jn])
    landmark_tips = np.unique(joint_landmark[is_tip])

    print(f"template {V} verts, {len(jn)} joints")
    print(f"landmark set (all joint centers): {len(landmark_all)} vertices")
    print(f"landmark set (leg-tip + antenna-tip joints only): {len(landmark_tips)} vertices"
          f" ({[jn[j] for j in np.nonzero(is_tip)[0]]})")
    print(f"computing HKS with {args.n_eig} eigenpairs...\n", flush=True)
    d_tpl = hks(v_tpl, faces, n_eig=args.n_eig)

    gt = np.load(os.path.join(MOON, "synth_clean", "ground_truth.npz"))
    gtv = gt["verts"]
    names = [str(x) for x in gt["names"]]

    fitp = os.path.join(MOON, "runs", args.run, "Stage_3_deform_fine.npz")
    fitd = np.load(fitp) if os.path.isfile(fitp) else None
    fit_labels = [str(x) for x in fitd["labels"]] if fitd is not None else []
    if fitd is None:
        print(f"[note] no fit at {fitp} yet -- 'fitted' row will be skipped, HKS/spatial/random rows unaffected")

    landmark_sets = {"joint_centers_55": landmark_all, "leg_antenna_tips": landmark_tips}

    ident = np.arange(V)
    rng = np.random.default_rng(0)
    results = {lname: {k: {"acc": [], "err": []} for k in ("hks_global", "spatial", "random_landmark", "fitted")}
               for lname in landmark_sets}

    for si in range(min(args.n, len(names))):
        tgt = gtv[si].astype(np.float64)
        d_tgt = hks(tgt, faces, n_eig=args.n_eig)

        def norm(x):
            c = x.mean(0)
            return (x - c) / np.abs(x - c).max()

        tn = norm(tgt)

        # global HKS NN (same as within_part_signal.py's hks_global) -- computed once,
        # then masked per landmark set below
        m_hks = cKDTree(d_tgt).query(d_tpl, k=1)[1]
        m_spatial = cKDTree(tn).query(norm(v_tpl), k=1)[1]
        m_fit = None
        stems = [x[:-4] if x.endswith(".obj") else x for x in fit_labels]
        if fitd is not None and names[si] in stems:
            bi = stems.index(names[si])
            fv = fitd["verts"][bi].astype(np.float64)
            m_fit = cKDTree(tn).query(fv, k=1)[1]

        for lname, lset in landmark_sets.items():
            r = results[lname]
            r["hks_global"]["acc"].append(float((m_hks[lset] == ident[lset]).mean()))
            r["hks_global"]["err"].append(np.linalg.norm(tn[m_hks[lset]] - tn[lset], axis=1))
            r["spatial"]["acc"].append(float((m_spatial[lset] == ident[lset]).mean()))
            r["spatial"]["err"].append(np.linalg.norm(tn[m_spatial[lset]] - tn[lset], axis=1))
            m_rand = lset[rng.integers(0, len(lset), len(lset))]
            r["random_landmark"]["acc"].append(float((m_rand == lset).mean()))
            r["random_landmark"]["err"].append(np.linalg.norm(tn[m_rand] - tn[lset], axis=1))
            if m_fit is not None:
                r["fitted"]["acc"].append(float((m_fit[lset] == ident[lset]).mean()))
                r["fitted"]["err"].append(np.linalg.norm(tn[m_fit[lset]] - tn[lset], axis=1))

        print(f"  {names[si]}  done", flush=True)

    for lname, r in results.items():
        print(f"\n=== landmark set: {lname} ({len(landmark_sets[lname])} verts) ===")
        print(f"{'method':<18}{'correct %':>11}{'median err':>13}{'p90 err':>10}")
        print("-" * 55)
        for k, v in r.items():
            if not v["acc"]:
                continue
            e = np.concatenate(v["err"])
            print(
                f"{k:<18}{100 * float(np.mean(v['acc'])):>10.2f}%{100 * float(np.median(e)):>12.2f}%"
                f"{100 * float(np.percentile(e, 90)):>9.2f}%"
            )

    import json

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out = {
        lname: {
            k: dict(
                correct=float(np.mean(v["acc"])) if v["acc"] else None,
                median_err=float(np.median(np.concatenate(v["err"]))) if v["acc"] else None,
                p90_err=float(np.percentile(np.concatenate(v["err"]), 90)) if v["acc"] else None,
            )
            for k, v in r.items()
        }
        for lname, r in results.items()
    }
    outp = os.path.join(HERE, "out", "within_part_signal_landmarks.json")
    with open(outp, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
