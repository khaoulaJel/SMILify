"""CSE feasibility check on the FROZEN B2 backbone -- retrieval accuracy, axial vs circumferential.

WHY THIS EXISTS
---------------
PROGRESS_SYNTHESIS_20260826.md Part 5 reported an encouraging but narrowly-scoped signal:
Pearson r=0.547 between true 3D distance and frozen-`sa1`-feature distance, among 44 points in ONE
circumferential band of ONE segment (`fe`) of ONE leg (`l1_r`) of ONE specimen (`synth_000`), using
a COARSE PROXY for the per-point feature (nearest FPS centroid of `sa1`'s 512 centroids).

That doc itself pre-registered exactly what had to happen before any GPU-heavy CSE retrain:
  (a) cover `tr` (the worst-offending segment in every other check), not just `fe`;
  (b) measure actual NEAREST-NEIGHBOR RETRIEVAL ACCURACY -- the ranking property CSE is judged on
      at inference -- not a linear distance correlation;
  (c) replicate across specimens.

This script does all three, and fixes the approximation in (the r=0.547 probe) for free:
`SMILCorrespondenceNet.forward` decodes both heads from `l0_points` = `fp1(...)`, shape (B,128,N)
-- a TRUE PER-POINT feature, not a centroid proxy (smil_correspondence_net.py:137). We re-run that
exact propagation path here and read `l0_points` directly, so the probe measures the same tensor a
CSE embedding head would actually sit on top of. No retraining, no weight changes, no edits to any
live module -- the checkpoint is loaded read-only and run in eval mode.

LITERATURE FRAMING (why axial vs circumferential is the decisive split)
----------------------------------------------------------------------
SurfEmb (arXiv:2111.13489) documents that a deterministic per-point embedding degrades
specifically on geometrically ambiguous/symmetric surface regions, and that the fix is a
contrastive formulation representing a multi-modal distribution over the surface rather than a
single point. A leg segment is near-rotationally-symmetric in cross-section, so its circumferential
direction is exactly such an ambiguous axis, while its axial (along-chain) direction is not. CoE
(arXiv:2412.05557, 3DV 2025) is the point-cloud-native SOTA for retrieving dense correspondence by
nearest-neighbour search in a learned per-point embedding -- the architecture family this probe is
assessing feasibility for.

The falsifiable prediction this yields: retrieval error should be LOW axially and HIGH
circumferentially. If that holds, a CSE/CoE head is the right fix and its residual circumferential
error is EXPECTED (per SurfEmb), not a failure. If instead error is high in BOTH directions, the
frozen backbone does not carry correspondence-usable structure and a CSE head bolted onto it would
be a bad GPU bet -- that would be a genuine negative result and must be reported as such.

PRE-REGISTERED CRITERIA (fixed before the first run, per project discipline)
---------------------------------------------------------------------------
Two reference levels are computed for every segment, because a metric's floor/ceiling must be
MEASURED, never assumed (cf. the 0.974-not-1.0 correction earlier in this investigation):
  * RANDOM floor : partner drawn uniformly among same-segment points. Chance performance.
  * 3D-NN ceiling: partner = spatially nearest OTHER sampled point. The best any embedding could
    do given finite sampling -- this is NOT zero error, and the gap to it is the only real target.

  PASS (CSE worth the GPU spend) for a segment iff BOTH:
    P1. feature-NN retrieval error is below the RANDOM floor with a paired sign test p<0.05
        (reported alongside Wilcoxon and paired-t, per standing project rule), AND
    P2. feature-NN error closes >=25% of the RANDOM->3D-NN gap ("gap closed" statistic below).
  DIRECTIONAL PREDICTION (SurfEmb) recorded separately, and does NOT gate PASS:
    P3. axial gap-closed > circumferential gap-closed.
  A segment failing P1/P2 is reported as a negative result with the same prominence as a pass.
  `tr` is called out explicitly regardless of outcome.
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
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402
from scipy import stats  # noqa: E402

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    SMILCorrespondenceNet,
    build_segment_taxonomy,
    build_vertex_labels,
    load_model_dict,
)


def per_point_features(model, pts):
    """Re-run SMILCorrespondenceNet's own propagation path and return `l0_points` (B,128,N).

    This mirrors smil_correspondence_net.py:130-137 EXACTLY (verified by reading that file); it is
    duplicated here rather than achieved by editing the live module, because the live module feeds
    the queued D1 chain and must stay byte-identical during this investigation.
    """
    xyz = pts.transpose(2, 1)                                   # (B,3,N)
    l1_xyz, l1_points = model.sa1(xyz, None)
    l2_xyz, l2_points = model.sa2(l1_xyz, l1_points)
    l3_xyz, l3_points = model.sa3(l2_xyz, l2_points)
    l2_points = model.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
    l1_points = model.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
    l0_points = model.fp1(xyz, l1_xyz, None, l1_points)         # (B,128,N)
    return l0_points


def segment_frame(vert_xyz):
    """Principal axis + length of a segment, from its OWN POSED VERTEX GEOMETRY (PCA).

    Deliberately NOT derived from `chain_pos`. Verified live on this checkpoint's taxonomy
    (2026-08-26): `chain_pos` is CONSTANT within a class -- e.g. class 1 `l1_r_co` has
    min == max == 0.0 -- reproducing the degenerate-label finding recorded in
    PROGRESS_SYNTHESIS_20260826.md Part 5. Any axial coordinate built on it would be identically
    zero, so the axial/circumferential split is derived from geometry instead, which is independent
    of the broken label and therefore also valid for judging a future CSE head.

    Returns (unit axis (3,), length along that axis) or (None, None) if degenerate.
    """
    if len(vert_xyz) < 4:
        return None, None
    X = vert_xyz - vert_xyz.mean(0)
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    axis = vt[0]
    t = X @ axis
    L = float(t.max() - t.min())
    return (axis, L) if L > 1e-9 else (None, None)


def decompose(v_from, v_to, axis, seg_len):
    """Split the retrieval error between two TRUE vertices into axial + circumferential parts.

    axial          = |(v_from - v_to) . axis|            (displacement along the segment's axis)
    circumferential= |component orthogonal to axis|       (the near-rotationally-symmetric
                     direction SurfEmb predicts is the genuinely ambiguous one)
    All three returned as a FRACTION of the segment's own length, so short `tr` and long `fe` are
    directly comparable.
    """
    d = v_from - v_to
    total = np.linalg.norm(d, axis=-1)
    axial = np.abs(d @ axis)
    circ = np.sqrt(np.maximum(total ** 2 - axial ** 2, 0.0))
    return total / seg_len, axial / seg_len, circ / seg_len


def gap_closed(feat_err, rand_err, nn3d_err):
    """Fraction of the RANDOM->3D-NN headroom that the feature-NN actually closes.

    1.0 = matches the achievable ceiling, 0.0 = no better than chance, <0 = worse than chance.
    Guards the degenerate case where the floor and ceiling coincide (no headroom to close).
    """
    denom = rand_err - nn3d_err
    if denom <= 1e-12:
        return float("nan")
    return float((rand_err - feat_err) / denom)


def three_tests(a, b):
    """Paired sign test + Wilcoxon + paired-t, always reported together (standing project rule --
    this investigation has twice been misled by a single test or an aggregate mean)."""
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    n = len(d)
    out = {"n": int(n), "mean_delta": float(d.mean()) if n else float("nan")}
    if n < 3:
        return {**out, "sign_p": float("nan"), "wilcoxon_p": float("nan"), "ttest_p": float("nan")}
    n_neg = int((d < 0).sum())          # d<0 == feature beat the comparison
    n_eff = int((d != 0).sum())
    out["n_better"] = n_neg
    out["sign_p"] = float(stats.binomtest(n_neg, n_eff, 0.5).pvalue) if n_eff else float("nan")
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    out["ttest_p"] = float(stats.ttest_1samp(d, 0.0).pvalue)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="diagnostics/anatomical_pose_init/out_B2_correspondence_net_20260826/best_model.pt")
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--val_frac", type=float, default=0.05, help="must match B2 training so we probe HELD-OUT meshes")
    ap.add_argument("--n_specimens", type=int, default=12)
    ap.add_argument("--n_points", type=int, default=2048, help="match B2 training-time density")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_cse_feasibility_20260826")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = args.device

    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))

    # The checkpoint stores the EXACT class_names it was trained with. Prefer those over the
    # freshly-rebuilt taxonomy and assert they agree -- a silent class-id permutation between
    # training and probing would corrupt every per-segment number here without raising anything.
    ckpt = torch.load(os.path.join(REPO, args.ckpt), map_location=device)
    ck_names = ckpt.get("class_names") if isinstance(ckpt, dict) else None
    if ck_names is not None:
        ck_names = [str(c) for c in list(ck_names)]
        if ck_names != list(class_names):
            raise SystemExit(f"class taxonomy drift: checkpoint has {len(ck_names)} classes, "
                             f"rebuilt {len(class_names)}; first mismatch at "
                             f"{next(i for i, (a, b) in enumerate(zip(ck_names, class_names)) if a != b)}")
        class_names = ck_names
        name_to_id = {c: i for i, c in enumerate(class_names)}
        print(f"[setup] class taxonomy matches checkpoint exactly ({len(class_names)} classes)", flush=True)
    seg_class_id, chain_pos = build_vertex_labels(dd, class_names, name_to_id)

    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].long()

    model = SMILCorrespondenceNet(n_classes=len(class_names)).to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    if isinstance(ckpt, dict):
        print(f"[setup] checkpoint epoch={ckpt.get('epoch')} val_total={ckpt.get('val_total')} "
              f"trained n_points={ckpt.get('n_points')}", flush=True)

    gt = np.load(os.path.join(REPO, args.corpus), allow_pickle=True)
    verts_all = gt["verts"]
    n_val = max(1, int(len(verts_all) * args.val_frac))
    val_verts = verts_all[-n_val:]                       # HELD OUT from B2 training
    take = min(args.n_specimens, len(val_verts))
    print(f"[setup] {len(class_names)} classes | corpus {verts_all.shape} | "
          f"val pool {len(val_verts)} | probing {take} held-out specimens | device={device}", flush=True)

    # segment short-name ('fe','tr',...) per class id, so we can pool across the 6 legs
    seg_short = {}
    for cid, cname in enumerate(class_names):
        seg_short[cid] = None if cname == "body" else cname.split("_")[-1]

    # Canonical (rest) vertex positions: the frame in which CROSS-SPECIMEN retrieval error is
    # measured, so the number is pose-independent -- two points are "the same correspondence" iff
    # their TRUE template vertices are close on the template, regardless of how each was posed.
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    canon_frame = {}
    for cid in range(1, len(class_names)):
        ax, L = segment_frame(v_template[seg_class_id == cid])
        if ax is not None:
            canon_frame[cid] = (ax, L)

    records = []
    cross_store = []          # per-specimen payload reused for the cross-specimen pass
    for si in range(take):
        v = torch.as_tensor(val_verts[si], dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]
        nn_idx = knn_points(pts.unsqueeze(0), v.unsqueeze(0), K=1).idx[0, :, 0].numpy()

        with torch.no_grad():
            feats = per_point_features(model, pts.unsqueeze(0).to(device))[0].T.cpu().numpy()  # (N,128)

        pts_np = pts.numpy()
        v_np = v.numpy()
        true_seg = seg_class_id[nn_idx]
        true_cp = chain_pos[nn_idx]
        true_xyz = v_np[nn_idx]                          # 3D position of each point's TRUE vertex

        for cid in np.unique(true_seg):
            if cid == 0:
                continue                                  # 'body' has no chain position
            m = np.where(true_seg == cid)[0]
            if len(m) < 8:
                continue
            axis, seg_len = segment_frame(v_np[seg_class_id == cid])
            if axis is None:
                continue

            F = feats[m]
            P = pts_np[m]
            k = len(m)

            # --- partner selection: feature-NN, 3D-NN ceiling, random floor (self excluded) ---
            def nn_excl_self(M):
                D = ((M[:, None, :] - M[None, :, :]) ** 2).sum(-1)
                np.fill_diagonal(D, np.inf)
                return D.argmin(1)

            j_feat = nn_excl_self(F)
            j_3dnn = nn_excl_self(P)
            j_rand = np.array([rng.choice(np.delete(np.arange(k), i)) for i in range(k)])

            for tag, j in (("feat", j_feat), ("nn3d", j_3dnn), ("rand", j_rand)):
                tot, ax, ci = decompose(true_xyz[m], true_xyz[m][j], axis, seg_len)
                records.append(dict(specimen=int(si), class_id=int(cid), class_name=class_names[cid],
                                    seg=seg_short[cid], partner=tag, n=int(k),
                                    total=float(np.mean(tot)), axial=float(np.mean(ax)),
                                    circ=float(np.mean(ci))))
        cross_store.append({"feats": feats, "true_vert": nn_idx, "true_seg": true_seg})
        print(f"[{si+1}/{take}] specimen done ({len(records)} records)", flush=True)

    # ================= CROSS-SPECIMEN retrieval: the property CSE is actually judged on ==========
    # Within-specimen retrieval can be aced by features that merely encode absolute position in the
    # current pose -- that would be useless for correspondence. The real test is: given a query
    # point on specimen A, does its feature-nearest point on a DIFFERENT specimen (different pose
    # AND different shape) land on the same place of the template? Error is measured between the
    # two TRUE template vertices in the canonical rest frame, so it is pose-independent by
    # construction. Same measured floor/ceiling discipline: RANDOM floor, and a TEMPLATE-NN ceiling
    # (best achievable given that B only sampled finitely many points near A's true vertex).
    cross_records = []
    for a in range(len(cross_store)):
        b = (a + 1) % len(cross_store)
        if b == a:
            break
        A, B = cross_store[a], cross_store[b]
        for cid in np.unique(A["true_seg"]):
            if cid == 0 or cid not in canon_frame:
                continue
            ia = np.where(A["true_seg"] == cid)[0]
            ib = np.where(B["true_seg"] == cid)[0]
            if len(ia) < 8 or len(ib) < 8:
                continue
            axis, seg_len = canon_frame[cid]
            FA, FB = A["feats"][ia], B["feats"][ib]
            va, vb = A["true_vert"][ia], B["true_vert"][ib]

            D = ((FA[:, None, :] - FB[None, :, :]) ** 2).sum(-1)
            j_feat = D.argmin(1)
            # ceiling: the closest B point to A's true vertex, in canonical space
            Dt = ((v_template[va][:, None, :] - v_template[vb][None, :, :]) ** 2).sum(-1)
            j_tnn = Dt.argmin(1)
            j_rand = rng.integers(0, len(ib), size=len(ia))

            for tag, j in (("feat", j_feat), ("tnn", j_tnn), ("rand", j_rand)):
                tot, ax, ci = decompose(v_template[va], v_template[vb][j], axis, seg_len)
                cross_records.append(dict(pair=f"{a}->{b}", class_id=int(cid),
                                          class_name=class_names[cid], seg=seg_short[cid],
                                          partner=tag, n=int(len(ia)),
                                          total=float(np.mean(tot)), axial=float(np.mean(ax)),
                                          circ=float(np.mean(ci))))
    print(f"[cross] {len(cross_records)} cross-specimen records", flush=True)

    # ---------------- aggregate per segment type, pooled over legs+specimens ----------------
    def pick(seg, partner, field):
        """One value per (specimen, class) pair, ordered consistently so tests stay PAIRED."""
        rs = sorted([r for r in records if r["seg"] == seg and r["partner"] == partner],
                    key=lambda r: (r["specimen"], r["class_id"]))
        return np.array([r[field] for r in rs])

    segs = sorted({r["seg"] for r in records})
    summary = {}
    for seg in segs:
        entry = {}
        for field in ("total", "axial", "circ"):
            f, n3, rd = pick(seg, "feat", field), pick(seg, "nn3d", field), pick(seg, "rand", field)
            if not (len(f) == len(n3) == len(rd)) or len(f) == 0:
                continue
            entry[field] = {
                "feat_mean": float(f.mean()), "nn3d_mean": float(n3.mean()), "rand_mean": float(rd.mean()),
                "gap_closed": gap_closed(f.mean(), rd.mean(), n3.mean()),
                "vs_random": three_tests(f, rd),
            }
        # outlier-excluded companion (standing rule: always report next to the full sample)
        f, n3, rd = pick(seg, "feat", "total"), pick(seg, "nn3d", "total"), pick(seg, "rand", "total")
        if len(f) >= 5:
            keep = np.abs(f - np.median(f)) <= 3 * (np.median(np.abs(f - np.median(f))) + 1e-12)
            if keep.sum() >= 3:
                entry["total_outlier_excluded"] = {
                    "n_kept": int(keep.sum()), "n_dropped": int((~keep).sum()),
                    "feat_mean": float(f[keep].mean()),
                    "gap_closed": gap_closed(f[keep].mean(), rd[keep].mean(), n3[keep].mean()),
                }
        summary[seg] = entry

    # ---------------- pre-registered verdicts ----------------
    verdicts = {}
    for seg, e in summary.items():
        if "total" not in e:
            continue
        t = e["total"]
        p1 = np.isfinite(t["vs_random"]["sign_p"]) and t["vs_random"]["sign_p"] < 0.05 and t["feat_mean"] < t["rand_mean"]
        p2 = np.isfinite(t["gap_closed"]) and t["gap_closed"] >= 0.25
        p3 = (np.isfinite(e.get("axial", {}).get("gap_closed", np.nan))
              and np.isfinite(e.get("circ", {}).get("gap_closed", np.nan))
              and e["axial"]["gap_closed"] > e["circ"]["gap_closed"])
        verdicts[seg] = {"P1_beats_random": bool(p1), "P2_closes_25pct_gap": bool(p2),
                         "PASS": bool(p1 and p2), "P3_axial_beats_circumferential": bool(p3)}

    # ---------------- cross-specimen aggregate (same statistics, same discipline) ----------------
    def xpick(seg, partner, field):
        rs = sorted([r for r in cross_records if r["seg"] == seg and r["partner"] == partner],
                    key=lambda r: (r["pair"], r["class_id"]))
        return np.array([r[field] for r in rs])

    cross_summary, cross_verdicts = {}, {}
    for seg in sorted({r["seg"] for r in cross_records}):
        entry = {}
        for field in ("total", "axial", "circ"):
            f, tn, rd = xpick(seg, "feat", field), xpick(seg, "tnn", field), xpick(seg, "rand", field)
            if not (len(f) == len(tn) == len(rd)) or len(f) == 0:
                continue
            entry[field] = {"feat_mean": float(f.mean()), "tnn_mean": float(tn.mean()),
                            "rand_mean": float(rd.mean()),
                            "gap_closed": gap_closed(f.mean(), rd.mean(), tn.mean()),
                            "vs_random": three_tests(f, rd)}
        cross_summary[seg] = entry
        if "total" in entry:
            t = entry["total"]
            p1 = np.isfinite(t["vs_random"]["sign_p"]) and t["vs_random"]["sign_p"] < 0.05 and t["feat_mean"] < t["rand_mean"]
            p2 = np.isfinite(t["gap_closed"]) and t["gap_closed"] >= 0.25
            p3 = (np.isfinite(entry.get("axial", {}).get("gap_closed", np.nan))
                  and np.isfinite(entry.get("circ", {}).get("gap_closed", np.nan))
                  and entry["axial"]["gap_closed"] > entry["circ"]["gap_closed"])
            cross_verdicts[seg] = {"P1_beats_random": bool(p1), "P2_closes_25pct_gap": bool(p2),
                                   "PASS": bool(p1 and p2), "P3_axial_beats_circumferential": bool(p3)}

    payload = {"config": vars(args), "n_specimens_probed": take,
               "summary": summary, "verdicts": verdicts, "records": records,
               "cross_summary": cross_summary, "cross_verdicts": cross_verdicts,
               "cross_records": cross_records}
    out_json = os.path.join(out_dir, "cse_feasibility_retrieval.json")
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n=== CSE FEASIBILITY (frozen B2 backbone, per-point fp1 features, held-out specimens) ===")
    print(f"{'seg':>5} {'n':>4} | {'feat':>7} {'rand':>7} {'3dNN':>7} | {'gapClosed':>9} "
          f"{'ax_gap':>7} {'ci_gap':>7} | {'sign_p':>8} {'wilcox':>8} | verdict")
    for seg in sorted(summary):
        e, vd = summary[seg], verdicts.get(seg, {})
        if "total" not in e:
            continue
        t = e["total"]
        print(f"{seg:>5} {t['vs_random']['n']:>4} | {t['feat_mean']:>7.3f} {t['rand_mean']:>7.3f} "
              f"{t['nn3d_mean']:>7.3f} | {t['gap_closed']:>9.3f} "
              f"{e.get('axial',{}).get('gap_closed',float('nan')):>7.3f} "
              f"{e.get('circ',{}).get('gap_closed',float('nan')):>7.3f} | "
              f"{t['vs_random']['sign_p']:>8.2e} {t['vs_random']['wilcoxon_p']:>8.2e} | "
              f"{'PASS' if vd.get('PASS') else 'FAIL'}"
              f"{'  (axial>circ, as SurfEmb predicts)' if vd.get('P3_axial_beats_circumferential') else ''}")
    print("\nErrors are means over (specimen,leg-segment) pairs, as a FRACTION of that segment's own length.")
    print("gapClosed: 1.0 = matches the achievable 3D-NN ceiling, 0.0 = chance. Floor/ceiling MEASURED, not assumed.")

    print("\n=== CROSS-SPECIMEN retrieval (THE decisive test: query on A -> feature-NN on B,")
    print("    error between TRUE template vertices in the canonical rest frame, pose-independent) ===")
    print(f"{'seg':>5} {'n':>4} | {'feat':>7} {'rand':>7} {'tmplNN':>7} | {'gapClosed':>9} "
          f"{'ax_gap':>7} {'ci_gap':>7} | {'sign_p':>8} {'wilcox':>8} | verdict")
    for seg in sorted(cross_summary):
        e, vd = cross_summary[seg], cross_verdicts.get(seg, {})
        if "total" not in e:
            continue
        t = e["total"]
        print(f"{seg:>5} {t['vs_random']['n']:>4} | {t['feat_mean']:>7.3f} {t['rand_mean']:>7.3f} "
              f"{t['tnn_mean']:>7.3f} | {t['gap_closed']:>9.3f} "
              f"{e.get('axial',{}).get('gap_closed',float('nan')):>7.3f} "
              f"{e.get('circ',{}).get('gap_closed',float('nan')):>7.3f} | "
              f"{t['vs_random']['sign_p']:>8.2e} {t['vs_random']['wilcoxon_p']:>8.2e} | "
              f"{'PASS' if vd.get('PASS') else 'FAIL'}"
              f"{'  (axial>circ)' if vd.get('P3_axial_beats_circumferential') else ''}")
    print("\nWithin-specimen numbers can be inflated by features that merely encode absolute position")
    print("in the current pose; the CROSS-SPECIMEN block is the one that reflects what a CSE head")
    print("must actually do at inference. Weight the verdict on that block.")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
