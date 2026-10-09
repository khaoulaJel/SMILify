"""Task 6 -- is multi-hypothesis correspondence worth building? (retrain-free feasibility)

WHY THIS IS JUSTIFIED INDEPENDENTLY OF THE EQUIVARIANCE QUESTION
---------------------------------------------------------------
The circumferential ambiguity is a genuine STRUCTURAL multimodality: SurfEmb's whole thesis is that
in ambiguous regions there may be no single correct answer to commit to, which is exactly the
situation "force one confident top-1" handles worst and "emit several plausible candidates, then
select" handles best. This investigation has already run that pattern successfully one level up, at
pose initialisation (multi-start candidates + a selection criterion). This applies it one level
down, at correspondence.

Crucially it needs NO retraining for a feasibility pass: the trained CSE head already produces a
full similarity row per point, so top-k candidates are free.

STRUCTURE OF THE TEST -- HEADROOM FIRST, SELECTOR SECOND
-------------------------------------------------------
It is pointless to design a selector before knowing whether the candidate set even CONTAINS a better
answer. So this measures, in order:

  1. top1        -- what the head currently commits to. The baseline.
  2. oracle@k    -- error if an ORACLE picked the best of the top-k candidates. This is the CEILING,
                    and the entire headroom available to any selector. If oracle@k is barely better
                    than top1, multi-hypothesis is dead on arrival regardless of selector quality,
                    and that must be reported as a negative rather than papered over by inventing
                    cleverer selection.
  3. realisable GT-FREE selectors, scored against that ceiling:
       coherence -- pick the candidate whose template position best agrees with the choices of the
                    point's SPATIAL NEIGHBOURS on the scan. Neighbouring surface points must map to
                    neighbouring template vertices; this is a pure smoothness prior, needs no ground
                    truth, and is the natural analogue of the neighbourhood-consensus idea that
                    already underpins the working cycle-consistency filter.
       cycle     -- pick the candidate with the best forward-backward consistency, reusing the one
                    signal already MEASURED to work on this problem (0.726 -> 0.449 at top-50%).

PRE-REGISTERED CRITERIA (fixed before running)
----------------------------------------------
  H1 HEADROOM EXISTS: oracle@5 error is at least 25% below top1 error on tr/fe. If this fails, STOP
     -- report that the candidate set does not contain better answers and do not build a selector.
  H2 A SELECTOR CAPTURES IT: the better GT-free selector closes >= 30% of the top1 -> oracle@5 gap,
     with a paired sign test p<0.05 across (specimen, segment-class) units.
  H1 without H2 means the idea is sound but needs a better selector -- an honest and useful
  intermediate outcome, to be reported as such rather than as success.

NOTE ON METRIC DISCIPLINE
-------------------------
No new geometric threshold is introduced here: the error measure is the same canonical-space
template-vertex distance normalised by segment length used by every earlier retrieval result, so
these numbers are directly comparable. Where a NEW thresholded quantity is introduced in future
work, verify_metric_on_toy.py must be run on it first -- that check has now caught three metric
errors, including one in a correction to an earlier mistake.
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
from cse_feasibility_retrieval_20260826 import segment_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3_ckpt", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/best_model.pt")
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--n_specimens", type=int, default=10)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--knn", type=int, default=12, help="spatial neighbours for the coherence selector")
    ap.add_argument("--headroom_bar", type=float, default=0.25)
    ap.add_argument("--capture_bar", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_multihyp_20260827")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing result file that was written with a different config")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    seg_class_id, _ = build_vertex_labels(dd, class_names, name_to_id)
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    seg_short = {c: (None if n == "body" else n.split("_")[-1]) for c, n in enumerate(class_names)}
    seg_len = {}
    for cid in range(1, len(class_names)):
        _, L = segment_frame(v_template[seg_class_id == cid])
        if L:
            seg_len[cid] = L

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

    rows = []
    for si, vv in enumerate(val):
        v = torch.as_tensor(vv, dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]
        tv = knn_points(pts.unsqueeze(0), v.unsqueeze(0), K=1).idx[0, :, 0].numpy()
        P = pts.numpy().astype(np.float64)

        with torch.no_grad():
            sim = (model(pts.unsqueeze(0).to(device))[0] @ keys.t())
            topv, topi = sim.topk(args.topk, dim=1)
            back = sim.argmax(0)                       # best point for each vertex (for cycle)
        topi = topi.cpu().numpy(); topv = topv.cpu().numpy()
        backn = back.cpu().numpy()

        # spatial neighbours on the scan
        nb = knn_points(pts.unsqueeze(0), pts.unsqueeze(0), K=args.knn + 1).idx[0, :, 1:].numpy()

        # --- selectors -------------------------------------------------------------------
        top1 = topi[:, 0]
        # coherence: candidate closest (in template space) to the mean template position of the
        # neighbours' current top-1 picks -- a pure smoothness prior, no ground truth used
        nb_mean = v_template[top1[nb]].mean(1)                      # (N,3)
        cand_xyz = v_template[topi]                                 # (N,k,3)
        coh = topi[np.arange(len(topi)),
                   np.linalg.norm(cand_xyz - nb_mean[:, None, :], axis=2).argmin(1)]
        # cycle: candidate whose "best point" lands nearest the query point itself
        cyc_d = np.linalg.norm(P[backn[topi]] - P[:, None, :], axis=2)   # (N,k)
        cyc = topi[np.arange(len(topi)), cyc_d.argmin(1)]

        cid_of = seg_class_id[tv]
        for segname in ("tr", "fe"):
            ids = [c for c, s in seg_short.items() if s == segname]
            m = np.isin(cid_of, ids)
            if m.sum() < 8:
                continue
            L = np.array([seg_len.get(c, np.nan) for c in cid_of[m]])
            def err(pred):
                return np.nanmean(np.linalg.norm(v_template[pred[m]] - v_template[tv[m]], axis=1) / L)
            oracle = np.nanmean(np.min(
                np.linalg.norm(v_template[topi[m]] - v_template[tv[m]][:, None, :], axis=2), axis=1) / L)
            # --- RANKING DIAGNOSTICS -------------------------------------------------------
            # The headroom above is an ORACLE ceiling (best-of-5 if you always chose right). It says
            # nothing about whether a realisable selector can FIND that best candidate. Earlier in
            # this investigation a larger pose-init candidate pool raised the ceiling while LOWERING
            # selector accuracy (75%->58% going n=3 -> n=5), so this must be checked, not assumed.
            # Cycle-consistency has won three contests as a BINARY filter / triage signal, never yet
            # as a fine-grained 1-of-5 ranker -- the exact role in which cosine similarity turned
            # out to be entangled and flat.
            cand_err = np.linalg.norm(v_template[topi[m]] - v_template[tv[m]][:, None, :], axis=2) \
                       / L[:, None]                                   # (M,k) true error per candidate
            best_idx = cand_err.argmin(1)                             # which candidate is truly best
            sel_idx = cyc_d[m].argmin(1)                              # which one cycle picks
            rank_acc = float((sel_idx == best_idx).mean())            # chance = 1/k
            # is the score even informative, or flat like cosine? spread of the cycle score across
            # the 5 candidates, normalised by its own scale
            cd = cyc_d[m]
            flatness = float(np.mean((cd.max(1) - cd.min(1)) / (cd.mean(1) + 1e-12)))
            # does the TRUE best candidate score better than the others on cycle distance?
            true_best_score = cd[np.arange(len(cd)), best_idx]
            others_mean = (cd.sum(1) - true_best_score) / max(cd.shape[1] - 1, 1)
            sep = float(np.mean(others_mean - true_best_score))       # >0 means correct one scores better
            rows.append(dict(spec=si, seg=segname, n=int(m.sum()), top1=float(err(top1)),
                             oracle=float(oracle), coherence=float(err(coh)), cycle=float(err(cyc)),
                             rank_acc=rank_acc, chance=1.0 / args.topk,
                             score_flatness=flatness, score_separation=sep,
                             aspect=float(np.linalg.norm(P.max(0) - P.min(0)))))
        print(f"[{si+1}/{len(val)}] done", flush=True)

    print(f"\n=== multi-hypothesis feasibility, top-{args.topk} (error as fraction of segment length) ===")
    print(f"{'seg':>4} {'n':>4} | {'top1':>8} {'oracle@k':>9} {'headroom':>9} | "
          f"{'coherence':>10} {'cycle':>8} | {'best captures':>13}")
    res = {}
    for segname in ("tr", "fe"):
        rs = [r for r in rows if r["seg"] == segname]
        if not rs:
            continue
        t1 = np.array([r["top1"] for r in rs]); orc = np.array([r["oracle"] for r in rs])
        coh = np.array([r["coherence"] for r in rs]); cyc = np.array([r["cycle"] for r in rs])
        headroom = (t1.mean() - orc.mean()) / t1.mean()
        gap = t1.mean() - orc.mean()
        cap_coh = (t1.mean() - coh.mean()) / gap if gap > 1e-12 else float("nan")
        cap_cyc = (t1.mean() - cyc.mean()) / gap if gap > 1e-12 else float("nan")
        best, capt = ("coherence", cap_coh) if cap_coh >= cap_cyc else ("cycle", cap_cyc)
        best_arr = coh if best == "coherence" else cyc
        d = best_arr - t1
        nb_ = int((d < 0).sum()); ne_ = int((d != 0).sum())
        sg = stats.binomtest(nb_, ne_, 0.5).pvalue if ne_ else float("nan")
        res[segname] = dict(n=len(rs), top1=float(t1.mean()), oracle=float(orc.mean()),
                            headroom=float(headroom), coherence=float(coh.mean()),
                            cycle=float(cyc.mean()), capture_coherence=float(cap_coh),
                            capture_cycle=float(cap_cyc), best=best, sign_p=float(sg),
                            H1=bool(headroom >= args.headroom_bar),
                            H2=bool(capt >= args.capture_bar and sg < 0.05 and d.mean() < 0))
        print(f"{segname:>4} {len(rs):>4} | {t1.mean():>8.4f} {orc.mean():>9.4f} {headroom:>8.1%} | "
              f"{coh.mean():>10.4f} {cyc.mean():>8.4f} | {best} {capt:>6.1%} p={sg:.3g}")

    print("\n=== RANKING CHECK: can cycle-consistency actually pick the best of 5? ===")
    print(f"{'seg':>4} | {'rank_acc':>9} {'chance':>7} {'sign p':>9} | {'flatness':>9} {'separation':>11} | verdict")
    rank = {}
    for segname in ("tr", "fe"):
        rs = [r for r in rows if r["seg"] == segname]
        if not rs:
            continue
        ra = np.array([r["rank_acc"] for r in rs]); ch = rs[0]["chance"]
        fl = np.array([r["score_flatness"] for r in rs])
        se = np.array([r["score_separation"] for r in rs])
        d = ra - ch
        nb_ = int((d > 0).sum()); ne_ = int((d != 0).sum())
        p = stats.binomtest(nb_, ne_, 0.5).pvalue if ne_ else float("nan")
        ok = ra.mean() > ch and p < 0.05 and se.mean() > 0
        rank[segname] = dict(rank_acc=float(ra.mean()), chance=float(ch), sign_p=float(p),
                             flatness=float(fl.mean()), separation=float(se.mean()), informative=bool(ok))
        print(f"{segname:>4} | {ra.mean():>9.3f} {ch:>7.3f} {p:>9.3g} | {fl.mean():>9.3f} "
              f"{se.mean():>+11.5f} | {'INFORMATIVE' if ok else 'NOT BETTER THAN CHANCE'}")
    print("  rank_acc = how often the selector picks the TRULY best of the k candidates (chance=1/k).")
    print("  separation > 0 means the correct candidate genuinely scores better; ~0 means the score")
    print("  is flat across candidates -- the same entanglement pathology that sank cosine similarity.")

    print(f"\n=== PRE-REGISTERED VERDICT (H1 headroom >= {args.headroom_bar:.0%}, "
          f"H2 selector captures >= {args.capture_bar:.0%} with sign p<0.05) ===")
    for segname, r in res.items():
        if not r["H1"]:
            v = "DEAD ON ARRIVAL -- the top-k set does not contain better answers"
        elif r["H2"]:
            v = "PURSUE -- headroom exists AND a GT-free selector captures it"
        else:
            v = "HEADROOM EXISTS but no selector captures it yet (sound idea, selector is the work)"
        print(f"  {segname}: headroom {r['headroom']:.1%}, best selector {r['best']} "
              f"captures {max(r['capture_coherence'], r['capture_cycle']):.1%} -> {v}")

    # Output path is STAMPED with the checkpoint and specimen count. Before this, every run wrote
    # to a fixed `multihypothesis.json`, so running the script on a different checkpoint silently
    # overwrote the committed C3/n=10 result -- which happened on 2026-08-28 and had to be restored
    # from git. A committed result must not be destroyable by a routine re-run with other options.
    stem = os.path.splitext(os.path.basename(os.path.dirname(args.c3_ckpt) or args.c3_ckpt))[0]
    fname = f"multihypothesis_{stem}_n{args.n_specimens}.json"
    canonical = os.path.join(out_dir, "multihypothesis.json")
    is_canonical = (stem == "out_C3_cse_head_20260826" and args.n_specimens == 10)
    path = canonical if is_canonical else os.path.join(out_dir, fname)
    if os.path.exists(path) and not args.force:
        with open(path) as fh:
            prev = json.load(fh).get("config", {})
        if prev != vars(args):
            raise SystemExit(
                f"{path} exists and was written with a DIFFERENT config. Refusing to overwrite a "
                f"recorded result -- pass --force, or --out_dir to write elsewhere."
            )
    with open(path, "w") as fh:
        json.dump({"config": vars(args), "summary": res, "ranking": rank, "rows": rows}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
