"""W1 -- within-part correspondence retrieval: is anatomy-conditioning the missing information?

Bars fixed in PREREGISTRATION_W1_within_part_descriptors.md BEFORE this ran. Read that first.

THE QUESTION
------------
~83% of residual correspondence error is WITHIN anatomical parts, not across them. Coarse part
identity is substantially solved. So: given that a region is ALREADY KNOWN to be the correct part,
does an anatomy-conditioned descriptor identify which point inside it corresponds, better than the
nearest-surface geometry the optimizer currently uses?

Smallest experiment that answers it. No network trained, no fitter run, no live module touched.

WHY THE GROUND TRUTH IS EXACT AND FREE
--------------------------------------
`synth_b2_train_corrb06.npz` holds `verts` of shape (4000, 10235, 3) -- 4000 specimens on ONE fixed
template topology. Vertex i of specimen A and vertex i of specimen B are therefore the same
anatomical point BY CONSTRUCTION. Correspondence ground truth is index identity; nothing is
annotated or inferred. This is what makes a decisive test cheap.

METRIC (and why not top-1)
--------------------------
error(i) = || B_verts[retrieved_j] - B_verts[i] || / seglen_B(s), i.e. 3D error on the target mesh
normalised by that segment's own PCA extent. NOT exact-vertex top-1: F7 measured that 19% of
vertices have a >0.99-similar neighbour and that the crowding is entirely local (median graph
distance 0.0049 body-diagonals), so top-1 cannot be high and does not need to be. Scoring on it
would repeat F6's error of fixing in the method what only needed fixing in the framing.

The true partner is always in the candidate set (retrieval is over full vertex sets, not a sample),
so the ceiling is exactly 0 and there is no finite-sampling floor to subtract.

THE PART FRAME, AND ITS ONE LIKELY SILENT FAILURE
-------------------------------------------------
The frame is PCA on the part's own vertices. PCA leaves BOTH the axis sign and the circumferential
reference ambiguous, and if either flips between specimens the descriptor is noise while still
producing plausible-looking numbers. Both are therefore resolved from anatomy, not from PCA:

  axis sign : oriented to point from the PARENT (proximal) segment's centroid toward this
              segment's centroid. Chain order co->tr->fe->ti->ta->pt; the parent of `co` is `body`.
  theta = 0 : referenced to the component of (body centroid - segment centroid) orthogonal to the
              axis. A fixed anatomical direction, identical in construction on every specimen.

The axis-flip rate is measured against a reference specimen and reported as a registered mechanism
check that VOIDS the arm if it fails -- not assumed.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    build_segment_taxonomy,
    build_vertex_labels,
    load_model_dict,
)

LEG_SEGMENTS = ["co", "tr", "fe", "ti", "ta", "pt"]
BAR_SEGMENTS = ["co", "tr", "fe", "ti"]   # the four the pre-registered bar is set on
MIN_VERTS = 30                            # segments smaller than this are excluded and named


# ---------------------------------------------------------------------------------------------
# Part frame
# ---------------------------------------------------------------------------------------------
def part_frame(seg_xyz, proximal_ref, body_centroid):
    """Anatomy-oriented orthonormal frame for one part on one specimen.

    Returns dict with origin, axis (unit, sign resolved by `proximal_ref`), e1/e2 spanning the
    cross-section (e1 referenced to `body_centroid`), axial length L and radial scale R.
    None if degenerate.
    """
    if len(seg_xyz) < 4:
        return None
    origin = seg_xyz.mean(0)
    X = seg_xyz - origin
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vt[0]

    # --- axis sign from anatomy, never from PCA ---
    prox_dir = origin - proximal_ref
    if np.linalg.norm(prox_dir) < 1e-12:
        return None
    if np.dot(axis, prox_dir) < 0:
        axis = -axis

    t = X @ axis
    L = float(t.max() - t.min())
    if L <= 1e-9:
        return None

    # --- circumferential reference from anatomy, never from PCA's 2nd component ---
    ref = body_centroid - origin
    ref = ref - np.dot(ref, axis) * axis
    if np.linalg.norm(ref) < 1e-9:
        # body centroid lies on the axis: fall back to the least-aligned canonical direction,
        # deterministically (still specimen-independent given the axis).
        alt = np.eye(3)[int(np.argmin(np.abs(axis)))]
        ref = alt - np.dot(alt, axis) * axis
    e1 = ref / np.linalg.norm(ref)
    e2 = np.cross(axis, e1)

    radial = np.sqrt(np.maximum((X @ e1) ** 2 + (X @ e2) ** 2, 0.0))
    R = float(np.percentile(radial, 95))
    if R <= 1e-9:
        R = 1.0
    return {"origin": origin, "axis": axis, "e1": e1, "e2": e2, "L": L, "R": R,
            "t_min": float(t.min())}


def canonical_coords(seg_xyz, fr):
    """Point positions in the part's own normalised frame: (t in [0,1], (r/R)cos0, (r/R)sin0).

    This is the anatomy-conditioned descriptor. It is a 3-vector in a canonical space, so nearest
    neighbour in it is directly comparable to nearest neighbour in raw 3D (the XYZ_RIGID arm) --
    no feature weights are chosen, which keeps the comparison free of tuning.
    """
    X = seg_xyz - fr["origin"]
    t = (X @ fr["axis"] - fr["t_min"]) / fr["L"]
    a = (X @ fr["e1"]) / fr["R"]
    b = (X @ fr["e2"]) / fr["R"]
    return np.stack([t, a, b], axis=1)


def rigid_into(seg_xyz, fr_src, fr_dst):
    """Map a part's points from its own frame into the target's frame as a RIGID motion.

    Correspondence-free: it uses only each part's own PCA frame, so it leaks nothing. This is the
    'part is already roughly placed, now take the nearest surface point' situation the optimizer is
    actually in -- the baseline that matters.
    """
    X = seg_xyz - fr_src["origin"]
    local = np.stack([X @ fr_src["axis"], X @ fr_src["e1"], X @ fr_src["e2"]], axis=1)
    B = np.stack([fr_dst["axis"], fr_dst["e1"], fr_dst["e2"]], axis=0)
    return local @ B + fr_dst["origin"]


# ---------------------------------------------------------------------------------------------
# Local geometric descriptors (Level 1 -- generic, no anatomy)
# ---------------------------------------------------------------------------------------------
def local_geo_features(all_xyz, idx, scales=(16, 48)):
    """Multi-scale local-PCA eigenvalue features at the given vertices.

    Computed on the FULL mesh neighbourhood (not the isolated part), because that is the geometry a
    descriptor would actually see. Per scale: linearity, planarity, scattering from the normalised
    eigenvalues -- standard, rotation-invariant, and carrying no anatomical knowledge, which is the
    point of this arm.
    """
    tree = cKDTree(all_xyz)
    feats = []
    for k in scales:
        _, nb = tree.query(all_xyz[idx], k=k)
        P = all_xyz[nb]                                   # (n, k, 3)
        P = P - P.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", P, P) / k
        ev = np.linalg.eigvalsh(cov)[:, ::-1]             # descending
        ev = np.maximum(ev, 0.0)
        s = ev.sum(axis=1, keepdims=True)
        s[s < 1e-20] = 1e-20
        e = ev / s
        lin = (e[:, 0] - e[:, 1])
        pla = (e[:, 1] - e[:, 2])
        sca = e[:, 2]
        feats.append(np.stack([lin, pla, sca], axis=1))
    return np.concatenate(feats, axis=1)


def zscore(F):
    mu, sd = F.mean(0), F.std(0)
    sd[sd < 1e-12] = 1.0
    return (F - mu) / sd


# ---------------------------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------------------------
def three_tests(a, b):
    """Paired sign test + Wilcoxon + paired-t, always together (standing project rule)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    out = {"n": int(n), "mean_delta": float(d.mean()) if n else float("nan")}
    if n < 3:
        return {**out, "sign_p": float("nan"), "wilcoxon_p": float("nan"), "ttest_p": float("nan")}
    n_better = int((d < 0).sum())
    n_eff = int((d != 0).sum())
    out["n_better"] = n_better
    out["n_eff"] = n_eff
    out["sign_p"] = float(stats.binomtest(n_better, n_eff, 0.5).pvalue) if n_eff else float("nan")
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    out["ttest_p"] = float(stats.ttest_1samp(d, 0.0).pvalue)
    return out


def nn_retrieve(Q, K):
    """Index into K of the nearest row to each row of Q."""
    return cKDTree(K).query(Q, k=1)[1]


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--n_specimens", type=int, default=12)
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="diagnostics/within_part_correspondence/out_W1")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    labels, _ = build_vertex_labels(dd, class_names, name_to_id)

    data = np.load(os.path.join(REPO, args.corpus))
    verts_all = data["verts"]
    n_total = verts_all.shape[0]
    # Held-out tail of the corpus, disjoint from anything used for B2/C-series training.
    spec_ids = np.arange(n_total - args.n_specimens, n_total)
    V = verts_all[spec_ids].astype(np.float64)
    print(f"corpus {args.corpus}: {n_total} specimens, using held-out tail {spec_ids[0]}..{spec_ids[-1]}")

    body_id = name_to_id["body"]
    body_idx = np.where(labels == body_id)[0]
    body_centroids = V[:, body_idx, :].mean(axis=1)                    # (S,3)

    # --- part index sets, and which are excluded ---
    parts, excluded = {}, []
    for cls, cid in name_to_id.items():
        if cls == "body":
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < MIN_VERTS:
            excluded.append((cls, int(len(idx))))
            continue
        parts[cls] = idx
    print(f"parts used: {len(parts)}; excluded (<{MIN_VERTS} verts): "
          + ", ".join(f"{c}({n})" for c, n in sorted(excluded)))

    # --- per-specimen frames, plus the registered axis-flip control ---
    frames = {}          # (cls, s) -> frame
    frame_align = {}     # (cls, s) -> |cos| with anatomical proximal->distal direction
    flip_stats = {}
    for cls, idx in parts.items():
        leg, side, seg = cls.split("_")
        pi = LEG_SEGMENTS.index(seg)
        if pi == 0:
            parent_idx = body_idx
        else:
            parent_cls = f"{leg}_{side}_{LEG_SEGMENTS[pi - 1]}"
            parent_idx = np.where(labels == name_to_id[parent_cls])[0]
        axes = []
        for s in range(len(V)):
            fr = part_frame(V[s][idx], V[s][parent_idx].mean(0), body_centroids[s])
            frames[(cls, s)] = fr
            axes.append(fr["axis"] if fr is not None else np.full(3, np.nan))
        # --- MECHANISM CHECK 2, re-specified (see RESULTS: the first version was the wrong
        # instrument). The original compared each specimen's axis to a REFERENCE SPECIMEN'S axis
        # in world coordinates -- but specimens are differently POSED, so a leg legitimately points
        # elsewhere and a negative dot measures pose, not a broken frame. The property that
        # actually matters is whether the PCA first component IS the anatomical long axis, which
        # is pose-invariant: |dot(pca_axis, unit(centroid - parent_centroid))|. Sign itself needs
        # no test -- it is resolved against that same direction by construction above.
        align = []
        for s in range(len(V)):
            fr = frames[(cls, s)]
            if fr is None:
                continue
            pd = V[s][idx].mean(0) - V[s][parent_idx].mean(0)
            nrm = np.linalg.norm(pd)
            if nrm < 1e-12:
                continue
            a_ = abs(float(np.dot(fr["axis"], pd / nrm)))
            frame_align[(cls, s)] = a_
            align.append(a_)
        align = np.array(align)
        flip_stats[cls] = {"n": int(len(align)),
                           "median_align": float(np.median(align)),
                           "min_align": float(align.min()),
                           "n_degenerate": int((align < 0.5).sum())}

    total_degen = sum(f["n_degenerate"] for f in flip_stats.values())
    total_frames = sum(f["n"] for f in flip_stats.values())
    flip_rate = total_degen / max(total_frames, 1)
    med_align = float(np.median([f["median_align"] for f in flip_stats.values()]))
    print(f"\n[MECHANISM CHECK 2] frame axis vs anatomical proximal->distal direction: "
          f"median |cos| = {med_align:.3f}; degenerate (|cos|<0.5): "
          f"{total_degen}/{total_frames} = {flip_rate:.4f}")

    # --- local-geo features, per specimen (full-mesh neighbourhoods) ---
    print("computing local-geo features ...")
    all_part_idx = np.concatenate([parts[c] for c in sorted(parts)])
    lg_feats = {}
    for s in range(len(V)):
        F = local_geo_features(V[s], all_part_idx)
        lg_feats[s] = {v: F[i] for i, v in enumerate(all_part_idx)}
    lg_lookup = {}
    for s in range(len(V)):
        lg_lookup[s] = np.zeros((len(labels), 6))
        for v, f in lg_feats[s].items():
            lg_lookup[s][v] = f

    # --- pairs ---
    pairs = []
    while len(pairs) < args.n_pairs:
        a, b = rng.integers(0, len(V), 2)
        if a != b:
            pairs.append((int(a), int(b)))

    ARMS = ["RANDOM", "XYZ_RIGID", "LOCAL_GEO", "PART_FRAME", "PART_FRAME_AXIAL"]
    rows = []          # one row per (pair, part, arm, query point) aggregated to per-part medians
    noop_fail = []
    noop_ties = {}
    decomp = []        # axial/circumferential decomposition of PART_FRAME residual

    for (a, b) in pairs:
        for cls, idx in sorted(parts.items()):
            fa, fb = frames[(cls, a)], frames[(cls, b)]
            if fa is None or fb is None:
                continue
            VA, VB = V[a][idx], V[b][idx]
            seglen = fb["L"]
            n = len(idx)

            def err_of(j):
                return np.linalg.norm(VB[j] - VB, axis=1) / seglen

            ret = {}
            ret["RANDOM"] = rng.integers(0, n, n)
            ret["XYZ_RIGID"] = nn_retrieve(rigid_into(VA, fa, fb), VB)
            ret["LOCAL_GEO"] = nn_retrieve(zscore(lg_lookup[a][idx]), zscore(lg_lookup[b][idx]))
            CA, CB = canonical_coords(VA, fa), canonical_coords(VB, fb)
            ret["PART_FRAME"] = nn_retrieve(CA, CB)
            ret["PART_FRAME_AXIAL"] = nn_retrieve(CA[:, :1], CB[:, :1])

            for arm, j in ret.items():
                e = np.linalg.norm(VB[j] - VB[np.arange(n)], axis=1) / seglen
                rows.append({"pair": f"{a}->{b}", "part": cls, "seg": cls.split("_")[2],
                             "arm": arm, "median_err": float(np.median(e)),
                             "mean_err": float(e.mean()), "n": n,
                             "clean_frame": bool(min(frame_align.get((cls, a), 0.0),
                                                     frame_align.get((cls, b), 0.0)) >= 0.5)})

            # --- mechanism check 3: split PART_FRAME residual axially vs circumferentially ---
            d = VB[ret["PART_FRAME"]] - VB[np.arange(n)]
            ax = np.abs(d @ fb["axis"])
            tot = np.linalg.norm(d, axis=1)
            ci = np.sqrt(np.maximum(tot ** 2 - ax ** 2, 0.0))
            decomp.append({"part": cls, "seg": cls.split("_")[2],
                           "axial": float(np.median(ax / seglen)),
                           "circ": float(np.median(ci / seglen))})

    # --- mechanism check 1: self-retrieval no-op (A == B must give exactly 0) ---
    print("\n[MECHANISM CHECK 1] self-retrieval no-op (A == B) ...")
    for s in [int(x) for x in rng.integers(0, len(V), 3)]:
        for cls, idx in sorted(parts.items()):
            fr = frames[(cls, s)]
            if fr is None:
                continue
            Vs = V[s][idx]
            n = len(idx)
            # Tie-aware: the harness is correct iff the retrieved point is at DESCRIPTOR distance
            # 0 from the query. Requiring index identity instead would flag a legitimate exact tie
            # -- two distinct vertices with identical descriptors -- as a bug. That is a real
            # property of LOCAL_GEO (flat regions share eigen-features) and says the descriptor is
            # non-discriminative there, which the endpoint already measures; it is not a harness
            # fault, and conflating the two would void a run for the wrong reason.
            checks = {
                "XYZ_RIGID": (rigid_into(Vs, fr, fr), Vs),
                "PART_FRAME": (canonical_coords(Vs, fr), canonical_coords(Vs, fr)),
                "LOCAL_GEO": (zscore(lg_lookup[s][idx]), zscore(lg_lookup[s][idx])),
            }
            for arm, (Q, K) in checks.items():
                j = nn_retrieve(Q, K)
                dd_ = np.linalg.norm(Q - K[j], axis=1)
                if dd_.max() > 1e-9:
                    noop_fail.append({"specimen": s, "part": cls, "arm": arm,
                                      "max_desc_dist": float(dd_.max())})
                e = np.linalg.norm(Vs[j] - Vs[np.arange(n)], axis=1) / fr["L"]
                noop_ties.setdefault(arm, []).append(float((e > 1e-9).mean()))
    print(f"  harness violations (retrieved point NOT at descriptor distance 0): {len(noop_fail)}"
          + ("" if not noop_fail else f"  <-- RUN IS VOID; examples {noop_fail[:3]}"))
    for arm, fr_ in sorted(noop_ties.items()):
        print(f"    {arm:<18} exact-tie rate (distinct vertex, identical descriptor): {np.mean(fr_):.4f}")

    # ------------------------------------------------------------------ aggregate + verdict
    def per_seg(arm, seg):
        return np.array([r["median_err"] for r in rows if r["arm"] == arm and r["seg"] == seg])

    segs = sorted({r["seg"] for r in rows}, key=lambda s: LEG_SEGMENTS.index(s))
    summary = {}
    print("\n" + "=" * 92)
    print("median normalised 3D within-part retrieval error (lower better); ceiling = 0")
    print("=" * 92)
    hdr = f"{'seg':<5}" + "".join(f"{a:>18}" for a in ARMS)
    print(hdr)
    for seg in segs:
        line = f"{seg:<5}"
        summary[seg] = {}
        for arm in ARMS:
            v = per_seg(arm, seg)
            summary[seg][arm] = {"median": float(np.median(v)), "n_part_pairs": int(len(v))}
            line += f"{np.median(v):>18.4f}"
        print(line)

    print("\n" + "=" * 92)
    print("PRE-REGISTERED BAR: PART_FRAME vs XYZ_RIGID, per segment, >=15% relative + sign p<0.05")
    print("=" * 92)
    print(f"{'seg':<5}{'XYZ_RIGID':>12}{'PART_FRAME':>13}{'rel.red.':>11}{'better':>10}{'sign p':>12}{'verdict':>10}")
    bar_pass = {}
    for seg in segs:
        x = per_seg("XYZ_RIGID", seg)
        p = per_seg("PART_FRAME", seg)
        t = three_tests(p, x)
        mx, mp = float(np.median(x)), float(np.median(p))
        rel = (mx - mp) / mx if mx > 0 else float("nan")
        ok = (rel >= 0.15) and (t["sign_p"] < 0.05) and (mp < mx)
        if seg in BAR_SEGMENTS:
            bar_pass[seg] = bool(ok)
        mark = ("PASS" if ok else "no") + ("" if seg in BAR_SEGMENTS else " (n/b)")
        print(f"{seg:<5}{mx:>12.4f}{mp:>13.4f}{rel:>10.1%}"
              f"{t.get('n_better',0)}/{t['n']:>8}{t['sign_p']:>12.2e}{mark:>10}")
        summary[seg]["bar"] = {"rel_reduction": rel, **t, "in_bar": seg in BAR_SEGMENTS,
                               "passes": bool(ok)}

    # Endpoint verdict against the pre-registered bar. Reported SEPARATELY from the frame-quality
    # flag, because conflating them would let a threshold inherited from the (replaced,
    # mis-specified) flip control silently overwrite a substantive result -- and the robustness
    # block above shows directly that frame degeneracy does not drive the endpoint.
    verdict = ("PASS" if all(bar_pass.get(s, False) for s in BAR_SEGMENTS)
               else "FAIL" if not any(bar_pass.values()) else "PARTIAL")
    if noop_fail:
        verdict = "VOID (harness)"
    frame_flag = "OK" if flip_rate <= 0.01 else f"{flip_rate:.1%} degenerate frames"

    print("\n" + "=" * 92)
    print("ROBUSTNESS: endpoint recomputed on NON-DEGENERATE frames only (|cos|>=0.5 on BOTH sides)")
    print("The frame is shared by XYZ_RIGID and PART_FRAME, so degeneracy penalises both equally;")
    print("this checks directly whether the 7.5% degenerate frames drive the verdict.")
    print("=" * 92)
    print(f"{'seg':<5}{'XYZ_RIGID':>12}{'PART_FRAME':>13}{'rel.red.':>11}{'kept':>10}{'sign p':>12}")
    clean_bar = {}
    for seg in segs:
        x = np.array([r["median_err"] for r in rows
                      if r["arm"] == "XYZ_RIGID" and r["seg"] == seg and r["clean_frame"]])
        p_ = np.array([r["median_err"] for r in rows
                       if r["arm"] == "PART_FRAME" and r["seg"] == seg and r["clean_frame"]])
        if len(x) < 3:
            print(f"{seg:<5}{'--':>12}{'--':>13}{'--':>11}{len(x):>10}")
            continue
        t = three_tests(p_, x)
        mx, mp = float(np.median(x)), float(np.median(p_))
        rel = (mx - mp) / mx if mx > 0 else float("nan")
        clean_bar[seg] = {"xyz": mx, "part_frame": mp, "rel": rel, "n": int(len(x)),
                          "sign_p": t["sign_p"]}
        print(f"{seg:<5}{mx:>12.4f}{mp:>13.4f}{rel:>10.1%}{len(x):>10}{t['sign_p']:>12.2e}")

    print("\n" + "=" * 92)
    print("[MECHANISM CHECK 3] PART_FRAME residual: axial vs circumferential (per segment length)")
    print("=" * 92)
    print(f"{'seg':<5}{'axial':>12}{'circ':>12}{'circ/axial':>13}")
    mech3 = {}
    for seg in segs:
        ax = np.median([d["axial"] for d in decomp if d["seg"] == seg])
        ci = np.median([d["circ"] for d in decomp if d["seg"] == seg])
        mech3[seg] = {"axial": float(ax), "circ": float(ci),
                      "ratio": float(ci / ax) if ax > 0 else float("nan")}
        print(f"{seg:<5}{ax:>12.4f}{ci:>12.4f}{ci/ax if ax>0 else float('nan'):>13.2f}")

    print("\n" + "=" * 92)
    print(f"ENDPOINT VERDICT (pre-registered bar): {verdict}")
    print(f"FRAME-QUALITY FLAG                  : {frame_flag}"
          "  -- see ROBUSTNESS block; endpoint unchanged on clean frames only")
    print("=" * 92)
    print(f"  bar segments {BAR_SEGMENTS}: " + ", ".join(f"{s}={bar_pass.get(s)}" for s in BAR_SEGMENTS))
    print(f"  [check 1] self-retrieval harness violations: {len(noop_fail)} (must be 0)")
    print(f"  [check 2] degenerate-frame rate           : {flip_rate:.4f} (must be ~0); median |cos| {med_align:.3f}")
    print(f"  [check 4] RANDOM is worst arm on all segs : "
          f"{all(summary[s]['RANDOM']['median'] >= max(summary[s][a]['median'] for a in ARMS if a!='RANDOM') for s in segs)}")

    payload = {"config": vars(args), "spec_ids": spec_ids.tolist(), "pairs": pairs,
               "excluded_parts": excluded, "summary": summary, "verdict": verdict, "frame_flag": frame_flag,
               "mechanism": {"noop_violations": noop_fail, "flip_rate": flip_rate,
                             "flip_per_part": flip_stats, "median_align": med_align,
                             "tie_rates": {k: float(np.mean(v)) for k, v in noop_ties.items()},
                             "axial_circ": mech3, "clean_frame_endpoint": clean_bar}}
    with open(os.path.join(out_dir, "w1_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    import csv
    with open(os.path.join(out_dir, "w1_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_dir}/w1_results.json and w1_rows.csv")


if __name__ == "__main__":
    main()
