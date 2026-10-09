"""W4 -- which normalisation destroyed the signal? A 2x2 ablation of W1's PART_FRAME.

Bar and predictions fixed in PREREGISTRATION_W4_normalisation_ablation.md BEFORE this ran.

W1 showed anatomy-conditioned normalised part coordinates lose to plain rigid proximity on every
segment, and concluded "normalising a part to canonical shape destroys correspondence-bearing
signal". But PART_FRAME divided out TWO things at once -- axial extent L and radial scale R -- so
W1 cannot say which did the damage. W4 varies one factor at a time, everything else identical.

THE PROOF-BACKED NO-OP
----------------------
`PF_none` (absolute axial, absolute radial) is a rigid change of basis of the part into its own
frame. Nearest neighbour is invariant under a rigid transform, so PF_none MUST equal XYZ_RIGID to
floating point. That is a provable identity, not a plausibility check, which is why it is the
voiding one: if it fails, the frame construction or the retrieval indexing is wrong and nothing
here may be read.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    build_segment_taxonomy, build_vertex_labels, load_model_dict,
)
from w1_within_part_retrieval import (  # noqa: E402
    LEG_SEGMENTS, MIN_VERTS, BAR_SEGMENTS, part_frame, rigid_into, three_tests, nn_retrieve,
)

# W1's published PART_FRAME medians -- mechanism check 2 must reproduce these.
W1_PART_FRAME = {"co": 0.2871, "tr": 0.1020, "fe": 0.1108, "ti": 0.1329, "ta": 0.1901}


def frame_coords(seg_xyz, fr, norm_axial, norm_radial, drop_radial=False, offset="centroid"):
    """Point positions in the part frame, with each scale factor divided out or not.

    norm_axial  : axial coordinate divided by the segment's length L
    norm_radial : radial coordinate divided by the segment's radius R
    offset      : "centroid" -> axial measured from the part's own centroid (X . axis)
                  "tipmin"   -> axial measured from its proximal extreme (X . axis - t_min), the
                                convention W1's `canonical_coords` used.

    WHY THE OFFSET IS AN EXPLICIT PARAMETER (run 1 was VOIDED over exactly this). The registered
    identity PF_none == XYZ_RIGID holds only under "centroid": `rigid_into` maps A's points into
    B's frame WITHOUT any t_min shift, so it compares both parts referenced to their own centroids.
    Subtracting t_min applies a DIFFERENT per-specimen translation to A and to B, which is not a
    rigid change of basis of the comparison -- it is a genuine re-alignment, and it broke the
    identity by 0.24. The ladder therefore runs under "centroid" so PF_none is a true no-op, and
    W1's convention is kept as its own arm so the W1 reproduction check still has something to
    check.
    """
    X = seg_xyz - fr["origin"]
    t = X @ fr["axis"]
    if offset == "tipmin":
        t = t - fr["t_min"]
    a = X @ fr["e1"]
    b = X @ fr["e2"]
    if norm_axial == "uniform":
        # isotropic: every coordinate divided by the SAME scalar -> relative weighting untouched
        t, a, b = t / fr["L"], a / fr["L"], b / fr["L"]
    else:
        if norm_axial:
            t = t / fr["L"]
        if norm_radial:
            a, b = a / fr["R"], b / fr["R"]
    if drop_radial:
        return t[:, None]
    return np.stack([t, a, b], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--n_specimens", type=int, default=12)
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="diagnostics/within_part_correspondence/out_W4")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    labels, _ = build_vertex_labels(dd, class_names, name_to_id)
    body_idx = np.where(labels == name_to_id["body"])[0]
    parts = {}
    for cls, cid in name_to_id.items():
        if cls == "body":
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) >= MIN_VERTS:
            parts[cls] = idx

    data = np.load(os.path.join(REPO, args.corpus))
    verts_all = data["verts"]
    spec_ids = np.arange(verts_all.shape[0] - args.n_specimens, verts_all.shape[0])
    V = verts_all[spec_ids].astype(np.float64)
    body_centroids = V[:, body_idx, :].mean(axis=1)
    print(f"held-out specimens {spec_ids[0]}..{spec_ids[-1]}; parts {len(parts)}")

    frames = {}
    for cls, idx in parts.items():
        leg, side, seg = cls.split("_")
        pi = LEG_SEGMENTS.index(seg)
        par = (body_idx if pi == 0
               else np.where(labels == name_to_id[f"{leg}_{side}_{LEG_SEGMENTS[pi-1]}"])[0])
        for s in range(len(V)):
            frames[(cls, s)] = part_frame(V[s][idx], V[s][par].mean(0), body_centroids[s])

    pairs = []
    while len(pairs) < args.n_pairs:
        a, b = rng.integers(0, len(V), 2)
        if a != b:
            pairs.append((int(a), int(b)))

    ARMS = ["RANDOM", "XYZ_RIGID", "PF_none", "PF_axial", "PF_radial", "PF_both",
            "PF_axial_only_coord", "PF_uniform", "PF_both_W1CONV"]
    rows, identity_dev = [], []

    for (a, b) in pairs:
        for cls, idx in sorted(parts.items()):
            fa, fb = frames[(cls, a)], frames[(cls, b)]
            if fa is None or fb is None:
                continue
            VA, VB = V[a][idx], V[b][idx]
            seglen, n = fb["L"], len(idx)

            def coords(na, nr, drop=False, off="centroid"):
                return (frame_coords(VA, fa, na, nr, drop, off),
                        frame_coords(VB, fb, na, nr, drop, off))

            ret = {"RANDOM": rng.integers(0, n, n),
                   "XYZ_RIGID": nn_retrieve(rigid_into(VA, fa, fb), VB)}
            for name, (na, nr, dr, off) in {
                "PF_none": (False, False, False, "centroid"),
                "PF_axial": (True, False, False, "centroid"),
                "PF_radial": (False, True, False, "centroid"),
                "PF_both": (True, True, False, "centroid"),
                "PF_axial_only_coord": (True, False, True, "centroid"),
                # PF_uniform is added because PF_axial/PF_radial/PF_both are CONFOUNDED: dividing
                # only one axis by its own scale changes not just what information is kept but the
                # relative WEIGHTING of axial vs radial inside the Euclidean NN metric, and those
                # two effects are not separable in an anisotropic arm. PF_uniform divides ALL THREE
                # coordinates by the SAME scalar (L), so the metric's relative weighting is exactly
                # preserved and only overall size is removed -- the biologically meaningful
                # normalisation, and the only one of these arms that isolates information loss from
                # re-weighting.
                "PF_uniform": ("uniform", False, False, "centroid"),
                # W1's exact convention, retained solely so check 2 has something to reproduce
                "PF_both_W1CONV": (True, True, False, "tipmin"),
            }.items():
                qa, qb = coords(na, nr, dr, off)
                ret[name] = nn_retrieve(qa, qb)

            errs = {}
            for arm, j in ret.items():
                e = np.linalg.norm(VB[j] - VB[np.arange(n)], axis=1) / seglen
                errs[arm] = float(np.median(e))
                rows.append({"pair": f"{a}->{b}", "part": cls, "seg": cls.split("_")[2],
                             "arm": arm, "median_err": errs[arm], "n": n})
            identity_dev.append(abs(errs["PF_none"] - errs["XYZ_RIGID"]))

    def per(arm, seg):
        return np.array([r["median_err"] for r in rows if r["arm"] == arm and r["seg"] == seg])

    segs = sorted({r["seg"] for r in rows}, key=LEG_SEGMENTS.index)

    # ---------------------------------------------------------------- check 1 (voiding)
    max_dev = float(np.max(identity_dev))
    id_ok = max_dev < 1e-9
    print(f"\n[CHECK 1 -- VOIDING] PF_none vs XYZ_RIGID identity: max |diff| over "
          f"{len(identity_dev)} part-pairs = {max_dev:.3e}   {'PASS' if id_ok else 'FAIL -- VOID'}")

    # ---------------------------------------------------------------- check 2
    print("\n[CHECK 2] PF_both_W1CONV reproduces W1's PART_FRAME (bar: within 1%)")
    rep_ok = True
    for seg in segs:
        got, want = float(np.median(per("PF_both_W1CONV", seg))), W1_PART_FRAME.get(seg)
        if want:
            dev = abs(got - want) / want
            rep_ok &= dev <= 0.01
            print(f"  {seg:<4} got {got:.4f}  W1 {want:.4f}  dev {dev:.2%}")

    # ---------------------------------------------------------------- results
    print("\n" + "=" * 100)
    print("median normalised 3D within-part retrieval error; ceiling = 0")
    print("=" * 100)
    print(f"{'seg':<5}" + "".join(f"{a:>21}" for a in ARMS))
    summary = {}
    for seg in segs:
        summary[seg] = {a: float(np.median(per(a, seg))) for a in ARMS}
        print(f"{seg:<5}" + "".join(f"{summary[seg][a]:>21.4f}" for a in ARMS))

    print("\n" + "=" * 100)
    print("COST OF EACH NORMALISATION, relative to PF_none (= no normalisation)")
    print("negative = normalising made it WORSE")
    print("=" * 100)
    print(f"{'seg':<5}{'PF_none':>10}{'÷L only':>12}{'÷R only':>12}{'÷both':>12}"
          f"{'axial-only':>13}{'ISO ÷L all':>13}")
    cost = {}
    for seg in segs:
        base = summary[seg]["PF_none"]
        cost[seg] = {k: (base - summary[seg][k]) / base
                     for k in ("PF_axial", "PF_radial", "PF_both", "PF_axial_only_coord",
                               "PF_uniform")}
        print(f"{seg:<5}{base:>10.4f}{cost[seg]['PF_axial']:>11.1%}"
              f"{cost[seg]['PF_radial']:>12.1%}{cost[seg]['PF_both']:>12.1%}"
              f"{cost[seg]['PF_axial_only_coord']:>13.1%}{cost[seg]['PF_uniform']:>13.1%}")

    # ---------------------------------------------------------------- P1 monotonicity
    print("\n" + "=" * 100)
    print("[P1] monotonicity: PF_none <= {PF_axial, PF_radial} <= PF_both, per segment")
    print("=" * 100)
    p1 = {}
    for seg in segs:
        s = summary[seg]
        ok = (s["PF_none"] <= s["PF_axial"] + 1e-12 and s["PF_none"] <= s["PF_radial"] + 1e-12
              and s["PF_axial"] <= s["PF_both"] + 1e-12 and s["PF_radial"] <= s["PF_both"] + 1e-12)
        p1[seg] = bool(ok)
        viol = []
        if s["PF_axial"] < s["PF_none"]:
            viol.append("÷L HELPS")
        if s["PF_radial"] < s["PF_none"]:
            viol.append("÷R HELPS")
        if s["PF_both"] < max(s["PF_axial"], s["PF_radial"]):
            viol.append("÷both < single")
        print(f"  {seg:<4} {'holds' if ok else 'VIOLATED: ' + ', '.join(viol)}")
    p1_all = all(p1.values())

    print("\n" + "=" * 100)
    print("[P2] is axial normalisation more harmful than radial? (per-segment sign test)")
    print("=" * 100)
    p2 = {}
    for seg in segs:
        ax, ra = per("PF_axial", seg), per("PF_radial", seg)
        t = three_tests(ra, ax)          # ra < ax  =>  radial cheaper  =>  axial more harmful
        p2[seg] = {"axial": float(np.median(ax)), "radial": float(np.median(ra)),
                   "sign_p": t["sign_p"], "axial_worse": bool(np.median(ax) > np.median(ra))}
        print(f"  {seg:<4} ÷L {np.median(ax):.4f}   ÷R {np.median(ra):.4f}   "
              f"axial more harmful: {p2[seg]['axial_worse']}   sign p {t['sign_p']:.2e}")

    print("\n" + "=" * 100)
    print(f"[CHECK 1] PF_none == XYZ_RIGID : {'PASS' if id_ok else 'VOID'} (max dev {max_dev:.1e})")
    print(f"[CHECK 2] PF_both == W1        : {'PASS' if rep_ok else 'FAIL'}")
    print(f"[CHECK 3] RANDOM worst on all  : "
          f"{all(summary[s]['RANDOM'] >= max(summary[s][a] for a in ARMS if a != 'RANDOM') for s in segs)}")
    print(f"[P1] monotone on all segments  : {p1_all}"
          + ("" if p1_all else "  <-- W1's stated lesson is TOO STRONG; see RESULTS"))
    print("=" * 100)

    payload = {"config": vars(args), "summary": summary, "cost_vs_PF_none": cost,
               "P1_monotone": p1, "P1_all": bool(p1_all), "P2_axial_vs_radial": p2,
               "identity_max_dev": max_dev, "identity_ok": bool(id_ok),
               "w1_reproduction_ok": bool(rep_ok), "spec_ids": spec_ids.tolist()}
    with open(os.path.join(out_dir, "w4_results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(os.path.join(out_dir, "w4_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_dir}/w4_results.json and w4_rows.csv")


if __name__ == "__main__":
    main()
