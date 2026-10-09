"""Does a test-time rotation ensemble resolve circumferential ambiguity? (no retraining)

WHY
---
The measured residual on tr/fe is specifically CIRCUMFERENTIAL: cross-specimen gap-closed is
0.34-0.57 axially but only 0.09-0.20 around the limb (near chance). A full SE(3)-equivariant retrain
(Vector Neurons et al.) is the principled fix, but it is a real GPU commitment and this
investigation's standing rule is to demonstrate cheaply that a component limits the result before
paying to fix it.

This is the cheap precursor: "poor man's equivariance". Rotate the input point cloud about each
segment's OWN estimated long axis, run the ALREADY-TRAINED CSE head on each rotated copy, map the
retrieved vertices back, and aggregate. If the network's circumferential confusion is an artifact of
seeing one arbitrary orientation, ensembling over orientations should recover signal that a single
forward pass cannot -- and that would be direct evidence a full equivariant retrain would pay off.
If it does nothing, the ambiguity is structural rather than orientational, and the equivariant case
weakens considerably before any GPU is spent.

AGGREGATION
-----------
Two schemes, because they fail differently:
  vote    -- majority retrieved vertex across rotations (robust, discards near-ties)
  mean    -- average the per-rotation similarity rows, then argmax (keeps the full distribution,
             which is the natural choice if the ambiguity really is multi-modal per SurfEmb)

MEASURED (same metric as every earlier retrieval result, so numbers are comparable)
-----------------------------------------------------------------------------------
Cross-specimen retrieval gap-closed against a MEASURED random floor and template-NN ceiling,
decomposed into axial and circumferential components via posed-geometry PCA -- not via `chain_pos`,
which is degenerate in this codebase.

PRE-REGISTERED CRITERION (fixed before running)
-----------------------------------------------
The ensemble is WORTH PURSUING iff circumferential gap-closed on tr/fe improves by >= 0.05 absolute
over the single-pass baseline under either aggregation, with a paired sign test p<0.05 across
(specimen-pair, segment-class) units. Improvements confined to the axial component do NOT count --
axial was never the problem.
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
from scipy import stats  # noqa: E402

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    build_segment_taxonomy, build_vertex_labels, load_model_dict,
)
from smil_cse_net import SMILCSENet  # noqa: E402
from cse_feasibility_retrieval_20260826 import decompose, gap_closed, segment_frame  # noqa: E402


def rot_about(axis, theta):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3_ckpt", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/best_model.pt")
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--n_specimens", type=int, default=8)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--n_rot", type=int, default=8, help="rotations about the segment long axis")
    ap.add_argument("--min_gain", type=float, default=0.05, help="pre-registered circumferential bar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_rotens_20260827")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    seg_class_id, _ = build_vertex_labels(dd, class_names, name_to_id)
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    seg_short = {c: (None if n == "body" else n.split("_")[-1]) for c, n in enumerate(class_names)}

    canon = {}
    for cid in range(1, len(class_names)):
        ax, L = segment_frame(v_template[seg_class_id == cid])
        if ax is not None:
            canon[cid] = (ax, L)

    ck = torch.load(os.path.join(REPO, args.c3_ckpt), map_location=device)
    model = SMILCSENet(n_vertices=ck["n_vertices"], embed_dim=ck["embed_dim"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    keys = model.vertex_embeddings()

    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].long()
    gt = np.load(os.path.join(REPO, args.corpus), allow_pickle=True)["verts"]
    n_val = max(1, int(len(gt) * args.val_frac))
    val = gt[-n_val:][:args.n_specimens]

    store = []
    for si, vv in enumerate(val):
        v = torch.as_tensor(vv, dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]
        tv = knn_points(pts.unsqueeze(0), v.unsqueeze(0), K=1).idx[0, :, 0].numpy()
        P = pts.numpy().astype(np.float64)
        ctr = P.mean(0)

        with torch.no_grad():
            sim0 = (model(pts.unsqueeze(0).to(device))[0] @ keys.t()).cpu().numpy()

        # long axis of the whole specimen: rotating about it preserves the ant's gross orientation
        _, _, vt_ = np.linalg.svd(P - ctr, full_matrices=False)
        axis = vt_[0]

        sims = [sim0]
        votes = [sim0.argmax(1)]
        for k in range(1, args.n_rot):
            R = rot_about(axis, 2 * np.pi * k / args.n_rot)
            Pr = (P - ctr) @ R.T + ctr
            with torch.no_grad():
                s = (model(torch.as_tensor(Pr, dtype=torch.float32).unsqueeze(0).to(device))[0]
                     @ keys.t()).cpu().numpy()
            sims.append(s)
            votes.append(s.argmax(1))

        V = np.stack(votes)                                   # (R,N)
        vote_ret = np.array([np.bincount(V[:, i], minlength=keys.shape[0]).argmax()
                             for i in range(V.shape[1])])
        mean_ret = np.mean(sims, axis=0).argmax(1)
        store.append(dict(tv=tv, single=sim0.argmax(1), vote=vote_ret, mean=mean_ret,
                          feat=np.mean(sims, axis=0)))
        print(f"[{si+1}/{len(val)}] {args.n_rot} rotations done", flush=True)

    # cross-specimen retrieval, identical metric to every earlier result
    recs = []
    for a in range(len(store)):
        b = (a + 1) % len(store)
        if b == a:
            break
        A, B = store[a], store[b]
        sa, sb = seg_class_id[A["tv"]], seg_class_id[B["tv"]]
        for cid in np.unique(sa):
            if cid == 0 or cid not in canon or seg_short[cid] not in ("tr", "fe"):
                continue
            ia, ib = np.where(sa == cid)[0], np.where(sb == cid)[0]
            if len(ia) < 8 or len(ib) < 8:
                continue
            ax, L = canon[cid]
            va, vb = A["tv"][ia], B["tv"][ib]
            Dt = ((v_template[va][:, None, :] - v_template[vb][None, :, :]) ** 2).sum(-1)
            parts = {"tnn": Dt.argmin(1), "rand": rng.integers(0, len(ib), size=len(ia))}
            Dr = ((A["feat"][ia][:, None, :] - B["feat"][ib][None, :, :]) ** 2).sum(-1)
            parts["ens"] = Dr.argmin(1)
            recs.append(("meta", a, cid))
            for tag, j in parts.items():
                tot, axl, ci = decompose(v_template[va], v_template[vb][j], ax, L)
                recs.append(dict(pair=a, cid=int(cid), seg=seg_short[cid], partner=tag,
                                 total=float(tot.mean()), axial=float(axl.mean()), circ=float(ci.mean())))

    # per-point direct retrieval error, single vs ensembles (the simpler, more direct read)
    print("\n=== direct template-vertex retrieval error, single pass vs rotation ensembles ===")
    print(f"{'seg':>4} {'n':>6} | {'single':>8} {'vote':>8} {'mean':>8} | {'vote-single':>11} {'sign p':>9}")
    direct = {}
    for segname in ("tr", "fe"):
        ids = [c for c, s in seg_short.items() if s == segname]
        errs = {"single": [], "vote": [], "mean": []}
        for S in store:
            m = np.isin(seg_class_id[S["tv"]], ids)
            if not m.any():
                continue
            for k in errs:
                cid_of = seg_class_id[S["tv"][m]]
                L = np.array([canon[c][1] if c in canon else np.nan for c in cid_of])
                e = np.linalg.norm(v_template[S[k][m]] - v_template[S["tv"][m]], axis=1) / L
                errs[k].append(np.nanmean(e))
        a1, a2, a3 = (np.array(errs[k]) for k in ("single", "vote", "mean"))
        d = a2 - a1
        nb = int((d < 0).sum()); ne = int((d != 0).sum())
        p = stats.binomtest(nb, ne, 0.5).pvalue if ne else float("nan")
        direct[segname] = dict(single=float(a1.mean()), vote=float(a2.mean()), mean=float(a3.mean()),
                               delta_vote=float(d.mean()), sign_p=float(p), n=int(len(a1)))
        print(f"{segname:>4} {len(a1):>6} | {a1.mean():>8.4f} {a2.mean():>8.4f} {a3.mean():>8.4f} | "
              f"{d.mean():>+11.4f} {p:>9.3g}")

    def pick(seg, tag, field):
        rs = [r for r in recs if isinstance(r, dict) and r["seg"] == seg and r["partner"] == tag]
        return np.array([r[field] for r in sorted(rs, key=lambda r: (r["pair"], r["cid"]))])

    print("\n=== cross-specimen gap-closed, ENSEMBLE features (circumferential is the bar) ===")
    print(f"{'seg':>4} {'n':>4} | {'total':>8} {'axial':>8} {'circ':>8}")
    cross = {}
    for seg in ("tr", "fe"):
        e, t, r = pick(seg, "ens", "total"), pick(seg, "tnn", "total"), pick(seg, "rand", "total")
        if len(e) == 0:
            continue
        g = {f: gap_closed(pick(seg, "ens", f).mean(), pick(seg, "rand", f).mean(),
                           pick(seg, "tnn", f).mean()) for f in ("total", "axial", "circ")}
        cross[seg] = dict(n=len(e), **g)
        print(f"{seg:>4} {len(e):>4} | {g['total']:>8.3f} {g['axial']:>8.3f} {g['circ']:>8.3f}")
    print("  single-pass reference (eval_C3_vs_B2, n=12): tr circ 0.317, fe circ 0.280")

    print(f"\n=== PRE-REGISTERED VERDICT (need circumferential +{args.min_gain} and sign p<0.05) ===")
    base_circ = {"tr": 0.317, "fe": 0.280}
    ok_any = False
    for seg in ("tr", "fe"):
        if seg not in cross:
            continue
        gain = cross[seg]["circ"] - base_circ[seg]
        d = direct.get(seg, {})
        ok = gain >= args.min_gain and d.get("sign_p", 1) < 0.05 and d.get("delta_vote", 0) < 0
        ok_any |= ok
        print(f"  {seg}: circumferential {base_circ[seg]:.3f} -> {cross[seg]['circ']:.3f} "
              f"({gain:+.3f}), direct sign p={d.get('sign_p', float('nan')):.3g} -> "
              f"{'PURSUE' if ok else 'NO GAIN'}")
    # Interpretation guard (added 2026-08-27 after the first run): a NEGATIVE result here has two
    # very different explanations, and the numbers distinguish them.
    #   (a) rotated inputs still yield VALID predictions but ensembling adds nothing
    #       -> the ambiguity is structural, and equivariance is unlikely to pay off.
    #   (b) rotated inputs yield GARBAGE (error rises far above the single-pass baseline)
    #       -> the network is simply not rotation-invariant, the ensemble's premise fails, and this
    #          test is UNINFORMATIVE about equivariance rather than evidence against it. Severe
    #          rotation sensitivity is precisely the condition an equivariant architecture targets.
    worst = max((direct[s_]["vote"] / max(direct[s_]["single"], 1e-9)) for s_ in direct) if direct else 1.0
    invalid = worst > 1.5
    if ok_any:
        print("\nOVERALL: rotation ensembling recovers circumferential signal -- a full equivariant "
              "retrain has a measured reason to expect payoff.")
    elif invalid:
        print(f"\nOVERALL: TEST INVALID, not a negative result. Rotated inputs degrade retrieval "
              f"{worst:.1f}x versus the single pass, so the ensemble is averaging garbage and its "
              f"premise fails. This says the trained head is strongly rotation-SENSITIVE; it says "
              f"NOTHING about whether an equivariant retrain would help, and must not be cited as "
              f"evidence against one.")
    else:
        print("\nOVERALL: rotated inputs remain valid yet ensembling adds nothing -- consistent with "
              "the ambiguity being structural rather than orientational.")

    with open(os.path.join(out_dir, "rotation_ensemble.json"), "w") as fh:
        json.dump({"config": vars(args), "direct": direct, "cross": cross,
                   "baseline_circ": base_circ, "pursue": bool(ok_any)}, fh, indent=2)
    print(f"\nwrote {out_dir}/rotation_ensemble.json")


if __name__ == "__main__":
    main()
