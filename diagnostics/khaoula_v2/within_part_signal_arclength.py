"""Does the rig's own kinematic chain (arc-length position, no external model, no training)
recover within-part correspondence better than HKS or DINO?

WHY THIS EXISTS (a fundamentally different lever than anything closed so far)
Everything closed to this point (SDF/geodesic/hull partitioning, HKS, raw and refined DINO)
attacked the within-part problem by giving each vertex a better FINGERPRINT so independent
per-point nearest-neighbour search picks the right match. None of them touch the SEARCH itself.
This is a different axis: the kinematic chain already gives a known 1D order along each limb
(coxa -> trochanter -> femur -> tibia -> tarsus -> pretarsus), sitting unused in
`weights`/`kintree_table`. This constrains matching to respect that known order -- not full
permutation search, just "position along the limb can't jump backward" -- which targets the
APERTURE PROBLEM directly: ambiguity along a tube's own axis, not around its circumference.
Arc-length position genuinely cannot resolve circumferential ambiguity (many vertices share the
same arc-length around a leg's ring), so this is expected to help median ERROR (a wrong-but-
same-ring match is spatially close) more than it helps exact-index "correct %" -- read both,
but weight median error as the practically meaningful number, same convention used throughout.

METHOD
For vertex v with dominant joint j (skinning-weight argmax) and parent p = kintree_table[0][j]:
  - joint positions J = J_regressor @ verts, computed independently on EACH mesh (template and
    target), NOT looked up from a template table by index -- using the target's own vertex
    index to look up a template-precomputed value would trivially recover ground truth by
    construction and prove nothing. This mirrors measure.py's own gauge-invariance argument for
    bone lengths (rotating a joint doesn't change parent-distance) extended to a continuous
    position along the bone.
  - segment lengths are this MESH's own ||J[j] - J[parent(j)]||, so chain arc-length is
    independently computed per specimen, cumulative from each chain's proximal root.
  - vertex arc position = cumulative distance to the start of its segment + its own projection
    fraction onto that segment, normalised by TOTAL chain length (0 = chain root, 1 = chain tip).
  - matching: 1D nearest-neighbour on this scalar, restricted per anatomical group (same gt16
    grouping `within_part_signal_dino.py`/`within_part_signal.py` already use, for direct
    comparability -- NOT a new grouping invented for this test).

Fully defined everywhere (no rendering, no occlusion, no "seen" mask) -- unlike DINO's 77%
coverage, this is 100% coverage like HKS, and needs no GPU: computed once from vertex positions
+ the model file, runs on CPU, same corpus (synth_clean) and same score_synth_roundtrip.py-style
exact-index metric everything else in this investigation uses.

PRE-REGISTERED GATE, fixed before running (same "half of current best" convention used for the
refinement layer): arc-length within-part median error must land AT OR BELOW HALF of raw DINO's
10.34% (the current best surviving within-part candidate), i.e. <= 5.17%. Secondary target: below
the fitted pipeline's reference. Read separately for {leg, antenna} groups (the hypothesis's
actual target -- a 1D chain order) vs {head, mandible, thorax/gaster-ish "body"} groups (where
the hypothesis does not obviously apply -- these are not tube-like single chains in the same
way) so a result isn't averaged across a case the mechanism was never expected to fix.
"""

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, "..", "moonshot"))
sys.path.insert(0, REPO)
sys.path.insert(0, MOON)
sys.path.insert(0, os.path.join(MOON, "hull"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from scipy.spatial import cKDTree  # noqa: E402
from within_part_signal import hks  # noqa: E402  -- unmodified, for the addendum comparison


# ======================================================================================
def arc_length_descriptor(verts, J_regressor, parents, dom_joint):
    """Per-vertex scalar arc-length position along its dominant joint's kinematic chain,
    normalised to [0,1] per chain, computed entirely from THIS mesh's own geometry.
    """
    V = len(verts)
    J = J_regressor @ verts  # (55, 3), this mesh's own joint positions

    # per-joint segment length (this mesh's own), and cumulative distance from each
    # joint's own chain root to the START of its segment
    n_j = len(parents)
    seg_len = np.zeros(n_j)
    for j in range(n_j):
        p = parents[j]
        if p >= 0:
            seg_len[j] = np.linalg.norm(J[j] - J[p])
    cum_start = np.zeros(n_j)
    # process in topological order (parents always have lower... NOT guaranteed here, so do a
    # simple fixed-point walk -- 55 joints, trivially cheap)
    order = []
    resolved = np.zeros(n_j, dtype=bool)
    resolved[parents == -1] = True
    cum_start[parents == -1] = 0.0
    remaining = list(np.where(~resolved)[0])
    while remaining:
        nxt = []
        for j in remaining:
            p = parents[j]
            if resolved[p]:
                cum_start[j] = cum_start[p] + seg_len[p]
                resolved[j] = True
            else:
                nxt.append(j)
        remaining = nxt

    # chain total length: walk each joint forward to find its chain's max cumulative extent.
    # Simplest robust approach: total length per joint's chain = max over all joints sharing
    # the same chain ROOT (first non-root ancestor) of (cum_start + seg_len).
    def chain_root(j):
        while parents[j] != -1 and parents[parents[j]] != -1:
            # walk up until parent is the skeleton root (b_t, index 0) or a body segment
            j = parents[j]
        return j

    roots = np.array([chain_root(j) for j in range(n_j)])
    chain_total = np.zeros(n_j)
    for r in np.unique(roots):
        m = roots == r
        chain_total[m] = (cum_start[m] + seg_len[m]).max()
    chain_total = np.maximum(chain_total, 1e-9)

    arc = np.zeros(V)
    for v in range(V):
        j = dom_joint[v]
        p = parents[j]
        if p < 0:
            arc[v] = 0.0
            continue
        seg = J[j] - J[p]
        L2 = float(seg @ seg)
        frac = 0.5 if L2 < 1e-12 else float(np.clip((verts[v] - J[p]) @ seg / L2, 0.0, 1.0))
        arc[v] = (cum_start[j] + frac * seg_len[j]) / chain_total[j]
    return arc


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--run", default="T04_baseline")
    ap.add_argument("--gt_groups", default="16", choices=["7", "13", "16"])
    args = ap.parse_args()

    import config
    from fitter_3d.trainer_hierarchical import anatomical_groups

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [str(x) for x in dd["J_names"]]
    sd, sa = args.gt_groups in ("13", "16"), args.gt_groups == "16"
    jg = anatomical_groups(jn, split_distal=sd, split_anterior=sa)
    gnames = sorted(set(jg.values()))
    n2i = {nm: i for i, nm in enumerate(gnames)}
    dom_joint = np.asarray(dd["weights"]).argmax(1)
    vpart = np.array([n2i[jg[int(d)]] for d in dom_joint], dtype=np.int64)
    parents = np.asarray(dd["kintree_table"])[0]
    Jr = np.asarray(dd["J_regressor"])

    # which groups are the hypothesis's actual target (tube-like chains) vs not
    leg_antenna_groups = {i for i, nm in enumerate(gnames) if nm.startswith("l") and (nm[1].isdigit()) or nm == "antenna"}
    # robust re-derivation: leg groups look like 'l{1,2,3}{p,d}_{r,l}' or 'l{1,2,3}_{r,l}';
    # antenna groups are literally named 'antenna' when split_anterior; else folded into 'body'
    leg_antenna_groups = {i for i, nm in enumerate(gnames) if nm.startswith("l") or nm == "antenna"}

    v_tpl = np.asarray(dd["v_template"], dtype=np.float64)
    faces = np.asarray(dd["f"]).astype(np.int64)
    V = len(v_tpl)

    print(f"template {V} verts, {len(faces)} faces, {len(gnames)} anatomical groups")
    print(f"hypothesis-target groups (leg/antenna chains): {sorted(gnames[i] for i in leg_antenna_groups)}")
    print("computing arc-length descriptor on the template ...\n", flush=True)
    d_tpl = arc_length_descriptor(v_tpl, Jr, parents, dom_joint)
    print("computing HKS descriptor on the template (comparison reference) ...", flush=True)
    d_tpl_hks = hks(v_tpl, faces, n_eig=120)

    gt = np.load(os.path.join(MOON, "synth_clean", "ground_truth.npz"))
    gtv = gt["verts"]
    names = [str(x) for x in gt["names"]]

    fitp = os.path.join(HERE, "runs", args.run, "Stage_3_deform_fine.npz")
    if not os.path.isfile(fitp):
        fitp = os.path.join(MOON, "runs", args.run, "Stage_3_deform_fine.npz")
    fitd = np.load(fitp) if os.path.isfile(fitp) else None
    print(f"  fitted-control source: {fitp if fitd is not None else '(none found)'}")
    fit_labels = [str(x) for x in fitd["labels"]] if fitd is not None else []

    rng = np.random.default_rng(0)
    acc = {k: [] for k in ("arclen_within", "arclen_within_legant", "arclen_within_other",
                            "hks_within", "random_within", "spatial", "fitted")}
    err = {k: [] for k in acc}
    ident = np.arange(V)

    for si in range(min(args.n, len(names))):
        tgt = gtv[si].astype(np.float64)
        print(f"  {names[si]}: computing arc-length + HKS ...", flush=True)
        d_tgt = arc_length_descriptor(tgt, Jr, parents, dom_joint)
        d_tgt_hks = hks(tgt, faces, n_eig=120)

        def norm(x):
            c = x.mean(0)
            return (x - c) / np.abs(x - c).max()

        tn = norm(tgt)
        extent = 1.0

        def score(match, key, mask=None):
            m = mask if mask is not None else np.ones(V, dtype=bool)
            acc[key].append(float((match[m] == ident[m]).mean()))
            err[key].append(np.linalg.norm(tn[match[m]] - tn[ident[m]], axis=1) / extent)

        m_arc = np.empty(V, dtype=np.int64)
        m_hks = np.empty(V, dtype=np.int64)
        m_rand = np.empty(V, dtype=np.int64)
        legant_mask = np.zeros(V, dtype=bool)
        other_mask = np.zeros(V, dtype=bool)
        for g in range(len(gnames)):
            idx = np.nonzero(vpart == g)[0]
            if len(idx) == 0:
                continue
            m_arc[idx] = idx[cKDTree(d_tgt[idx].reshape(-1, 1)).query(d_tpl[idx].reshape(-1, 1), k=1)[1]]
            m_hks[idx] = idx[cKDTree(d_tgt_hks[idx]).query(d_tpl_hks[idx], k=1)[1]]
            m_rand[idx] = idx[rng.integers(0, len(idx), len(idx))]
            if g in leg_antenna_groups:
                legant_mask[idx] = True
            else:
                other_mask[idx] = True
        score(m_arc, "arclen_within")
        score(m_arc, "arclen_within_legant", legant_mask)
        score(m_arc, "arclen_within_other", other_mask)
        score(m_hks, "hks_within")
        score(m_rand, "random_within")
        score(cKDTree(tn).query(norm(v_tpl), k=1)[1], "spatial")

        if fitd is not None and names[si] in [x[:-4] if x.endswith(".obj") else x for x in fit_labels]:
            bi = [x[:-4] if x.endswith(".obj") else x for x in fit_labels].index(names[si])
            fv = fitd["verts"][bi].astype(np.float64)
            score(cKDTree(tn).query(fv, k=1)[1], "fitted")
        print(f"  {names[si]}  done", flush=True)

    print(f"\n{'method':<24}{'correct %':>11}{'median err':>13}{'p90 err':>10}   what it is")
    print("-" * 96)
    desc = {
        "fitted": "the pipeline's own result",
        "spatial": "nearest target vertex to the TEMPLATE rest pose",
        "random_within": "chance level inside the correct part",
        "hks_within": "intrinsic descriptor NN, correct part given (100% coverage, reproduction)",
        "arclen_within": "arc-length NN, correct part given, ALL groups",
        "arclen_within_legant": "arc-length NN, LEG/ANTENNA groups only (the hypothesis target)",
        "arclen_within_other": "arc-length NN, head/mandible/body groups (not the hypothesis target)",
    }
    for k in ("fitted", "spatial", "random_within", "hks_within",
              "arclen_within", "arclen_within_legant", "arclen_within_other"):
        if not acc[k]:
            continue
        e = np.concatenate(err[k])
        print(
            f"{k:<24}{100 * float(np.mean(acc[k])):>10.2f}%{100 * float(np.median(e)):>12.2f}%"
            f"{100 * float(np.percentile(e, 90)):>9.2f}%   {desc[k]}"
        )

    print("\nREADING")
    RAW_DINO_REFERENCE = 0.1034  # T1.1, job 15520392, this exact E6 protocol
    result = {}
    if acc["arclen_within_legant"]:
        me_all = np.median(np.concatenate(err["arclen_within"]))
        me_la = np.median(np.concatenate(err["arclen_within_legant"]))
        me_other = np.median(np.concatenate(err["arclen_within_other"])) if acc["arclen_within_other"] else float("nan")
        mf = np.median(np.concatenate(err["fitted"])) if acc["fitted"] else float("nan")
        mh = np.median(np.concatenate(err["hks_within"]))
        print(f"  arc-length within-part median error: ALL groups {100*me_all:.2f}%, "
              f"leg/antenna (hypothesis target) {100*me_la:.2f}%, other groups {100*me_other:.2f}%")
        print(f"  reference: HKS within-part {100*mh:.2f}%, raw-DINO reference {100*RAW_DINO_REFERENCE:.2f}%, "
              f"fitted pipeline {100*mf:.2f}%")
        print(f"\n  PRE-REGISTERED GATE (leg/antenna groups, the hypothesis's actual target): "
              f"must land <= half of raw DINO's {100*RAW_DINO_REFERENCE:.2f}%, i.e. <= {50*RAW_DINO_REFERENCE:.2f}%.")
        clears = me_la <= 0.5 * RAW_DINO_REFERENCE
        print(f"  arc-length leg/antenna median err {100*me_la:.2f}% vs gate {50*RAW_DINO_REFERENCE:.2f}% -> "
              f"{'CLEARS the gate' if clears else 'does NOT clear the gate'}")
        print(f"  (secondary target, would justify T1.2: fitted pipeline {100*mf:.2f}% -- "
              f"{'also cleared' if me_la < mf else 'not cleared'})")
        result = dict(median_err_all=float(me_all), median_err_legant=float(me_la),
                      median_err_other=float(me_other), hks_within_this_run=float(mh),
                      raw_dino_reference=RAW_DINO_REFERENCE, fitted_reference=float(mf),
                      clears_gate=bool(clears))

    import json

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out = dict(
        scores={k: dict(correct=float(np.mean(v)) if v else None,
                         median_err=float(np.median(np.concatenate(err[k]))) if v else None)
                for k, v in acc.items()},
        n_specimens=args.n, gt_groups=args.gt_groups, **result,
    )
    out_p = os.path.join(HERE, "out", "within_part_signal_arclength.json")
    with open(out_p, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
