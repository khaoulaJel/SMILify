"""Which confidence signal actually tracks correspondence correctness? (retrain-free)

WHY
---
The pre-registered cosine-similarity confidence check FAILED on 2026-08-27, in the WRONG direction:
it retained `ta` at 95.7% and `ti` at 94.5% (the two segments measured at/below chance) while
suppressing `tr` to 55.6% and `co` to 18.2% (the reliable ones). Without a working confidence
signal, correspondences can only be used where segments are whitelisted from held-out ground truth
in advance -- which is impossible on bench50 real scans. This is the hard blocker.

THE LITERATURE EXPLAINS THE FAILURE, AND IT IS A NAMED TRAP
-----------------------------------------------------------
LightGlue (Lindenberger et al., ICCV 2023, arXiv:2306.13643) states the problem directly:
SuperGlue's dustbin "entangles the similarity score of all points", and LightGlue's contribution is
to DISENTANGLE two distinct quantities -- whether two points are SIMILAR, and whether a point is
MATCHABLE at all. A point in a repetitive or near-symmetric region can score high similarity to the
WRONG match precisely because it is ambiguous. That is exactly our circumferential failure mode, so
no threshold on a similarity score can fix it: the signal itself is the wrong quantity.
R2D2 (arXiv:1906.06195) makes the same split, learning repeatability and reliability as separate
trained quantities rather than thresholding description confidence.

The structurally different signals are consistency-based, not score-based:
* Mutual nearest neighbour (MNN) -- a match counts only if the two points are each other's nearest
  neighbour. Neighbourhood Consensus Networks (arXiv:1810.10510) uses a soft MNN filter as a gating
  mechanism that downweights non-mutual matches.
* Cycle consistency -- match forward then backward and measure how far you land from where you
  started; standard practice filters on forward-backward error rather than on match score.

WHAT THIS SCRIPT DOES
---------------------
Compares FOUR candidate confidence signals on the trained C3 head, with NO retraining, and scores
each one against held-out ground truth:

  1. `cosine`     -- the known-failing control, kept so the comparison is honest.
  2. `cycle`      -- forward-backward: point i -> vertex v -> whichever point v retrieves best;
                     confidence = -(distance back to point i).
  3. `mnn`        -- binary mutual-nearest-neighbour agreement (the strict version of `cycle`).
  4. `stability`  -- agreement across INDEPENDENT resamplings of the same mesh: how tightly the
                     points that retrieve a given vertex cluster in space. This is the cheap
                     stand-in for R2D2's "repeatability", and needs no second network.

Ground truth is used ONLY to VALIDATE the signals here. A signal that works can then be applied at
inference with no ground truth at all -- that is the entire point.

PRE-REGISTERED CRITERIA (fixed before running)
----------------------------------------------
A signal is USABLE iff both:
  S1. Spearman correlation with per-point retrieval error is significantly positive (better
      confidence -> lower error), rho > 0.2, p < 0.01.
  S2. Keeping its top 50% of points LOWERS mean retrieval error versus keeping all points, and
      lowers it by more than the cosine control does.
Additionally reported (not gating): per-segment survival, which must suppress `ta`/`ti` more than
`tr`/`fe` for the signal to solve the deployment blocker.
Signals failing these are reported as failures with the same prominence as any that pass.
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
    ap.add_argument("--n_specimens", type=int, default=12)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--n_repeats", type=int, default=4, help="independent resamples, for `stability`")
    ap.add_argument("--keep_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_confidence_probe_20260827")
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
    print(f"[setup] C3 epoch={ck['epoch']} device={device}", flush=True)

    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].long()
    gt = np.load(os.path.join(REPO, args.corpus), allow_pickle=True)["verts"]
    n_val = max(1, int(len(gt) * args.val_frac))
    val = gt[-n_val:][:args.n_specimens]

    rows = []
    for si, vv in enumerate(val):
        v = torch.as_tensor(vv, dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))

        # ---- repeated independent samples: used for `stability`, first one for everything else ----
        samples = []
        for _ in range(args.n_repeats):
            pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]
            tv = knn_points(pts.unsqueeze(0), v.unsqueeze(0), K=1).idx[0, :, 0].numpy()
            with torch.no_grad():
                q = model(pts.unsqueeze(0).to(device))[0]          # (N,D)
                sim = q @ keys.t()                                  # (N,V)
                conf, ret = sim.max(1)
            samples.append(dict(pts=pts.numpy(), tv=tv, q=q, sim=sim,
                                conf=conf.cpu().numpy(), ret=ret.cpu().numpy()))

        s0 = samples[0]
        N = len(s0["tv"])

        # ---- cycle consistency: vertex -> best point, then distance back to the origin point ----
        # sim is (N,V); the best POINT for vertex v is argmax over N of sim[:, v].
        with torch.no_grad():
            best_pt_for_vertex = s0["sim"].argmax(0).cpu().numpy()   # (V,)
        back_idx = best_pt_for_vertex[s0["ret"]]                     # (N,) point each match returns to
        cycle_d = np.linalg.norm(s0["pts"] - s0["pts"][back_idx], axis=1)
        mnn = (back_idx == np.arange(N)).astype(float)

        # ---- stability: spatial spread of the points that retrieve the same vertex, pooled over
        #      independent resamples. A vertex whose retrievers scatter is unreliable.
        acc = {}
        for s in samples:
            for i, rv in enumerate(s["ret"]):
                acc.setdefault(rv, []).append(s["pts"][i])
        spread = {rv: float(np.linalg.norm(np.std(np.array(p), axis=0))) for rv, p in acc.items()}
        stab = np.array([-spread.get(rv, np.inf) for rv in s0["ret"]])   # higher = tighter = better

        # ---- ground-truth correctness of each retrieval, normalised by segment length ----
        for i in range(N):
            cid = int(seg_class_id[s0["tv"][i]])
            if cid == 0 or cid not in seg_len:
                continue
            err = np.linalg.norm(v_template[s0["ret"][i]] - v_template[s0["tv"][i]]) / seg_len[cid]
            rows.append(dict(spec=si, seg=seg_short[cid], err=float(err),
                             cosine=float(s0["conf"][i]), cycle=float(-cycle_d[i]),
                             mnn=float(mnn[i]), stability=float(stab[i])))
        print(f"[{si+1}/{len(val)}] done ({len(rows)} rows)", flush=True)

    err = np.array([r["err"] for r in rows])
    segs = np.array([r["seg"] for r in rows])
    print(f"\nn={len(rows)} retrieved points | mean err (all points) = {err.mean():.4f} "
          f"of segment length")

    print("\n=== S1: does the signal track correctness? (Spearman vs retrieval error) ===")
    print(f"{'signal':>10} | {'rho':>7} {'p':>10} | {'S1 pass':>7}")
    res = {}
    for sig in ("cosine", "cycle", "mnn", "stability"):
        x = np.array([r[sig] for r in rows])
        ok = np.isfinite(x)
        rho, p = stats.spearmanr(x[ok], err[ok])
        # higher confidence should mean LOWER error -> expect negative rho
        s1 = bool((-rho) > 0.2 and p < 0.01)
        res[sig] = dict(rho=float(rho), p=float(p), S1=s1)
        print(f"{sig:>10} | {rho:>7.3f} {p:>10.2e} | {str(s1):>7}")

    print(f"\n=== S2: does keeping the top {100*args.keep_frac:.0f}% by this signal reduce error? ===")
    base = err.mean()
    print(f"{'signal':>10} | {'kept err':>9} {'vs all':>8} {'improve':>8} | {'S2 pass':>7}")
    cos_improve = None
    for sig in ("cosine", "cycle", "mnn", "stability"):
        x = np.array([r[sig] for r in rows])
        x = np.where(np.isfinite(x), x, -np.inf)
        k = max(1, int(len(x) * args.keep_frac))
        keep = np.argsort(-x)[:k]
        e = err[keep].mean()
        imp = base - e
        if sig == "cosine":
            cos_improve = imp
        s2 = bool(e < base and imp > cos_improve)
        res[sig].update(kept_err=float(e), improve=float(imp), S2=s2,
                        USABLE=bool(res[sig]["S1"] and s2))
        print(f"{sig:>10} | {e:>9.4f} {base:>8.4f} {imp:>+8.4f} | {str(s2):>7}")

    print(f"\n=== per-segment survival at top {100*args.keep_frac:.0f}% "
          f"(want ta/ti LOW, tr/fe HIGH) ===")
    order = ["tr", "fe", "co", "ti", "ta"]
    print(f"{'signal':>10} | " + " ".join(f"{s:>6}" for s in order))
    for sig in ("cosine", "cycle", "mnn", "stability"):
        x = np.array([r[sig] for r in rows])
        x = np.where(np.isfinite(x), x, -np.inf)
        k = max(1, int(len(x) * args.keep_frac))
        keep = np.zeros(len(x), bool)
        keep[np.argsort(-x)[:k]] = True
        surv = {}
        for s in order:
            m = segs == s
            surv[s] = float(keep[m].mean()) if m.sum() else float("nan")
        res[sig]["survival"] = surv
        print(f"{sig:>10} | " + " ".join(f"{100*surv[s]:>5.1f}%" for s in order))

    # ---- segment-conditional quality of what SURVIVES the filter -------------------------------
    # Uniform survival across segments does NOT imply uniform quality of the survivors. Broadening
    # from tr/fe to all segments buys COVERAGE; whether it also buys usable precision on the hard
    # segments (ti/ta) is a separate question, and it is the input a class-balanced retrain needs:
    # if ti/ta survivors remain materially worse than tr/fe survivors, per-segment quotas must be
    # weighted by survivor QUALITY, not merely by vertex count.
    print(f"\n=== segment-conditional error of SURVIVORS at top {100*args.keep_frac:.0f}% "
          f"(cycle filter) ===")
    x = np.array([r["cycle"] for r in rows])
    x = np.where(np.isfinite(x), x, -np.inf)
    k = max(1, int(len(x) * args.keep_frac))
    keep = np.zeros(len(x), bool)
    keep[np.argsort(-x)[:k]] = True
    print(f"{'seg':>4} {'n_all':>6} {'n_kept':>7} {'surv':>6} | {'err_all':>8} {'err_kept':>8} "
          f"{'gain':>7} | {'vs tr/fe kept':>13}")
    ref = err[keep & np.isin(segs, ["tr", "fe"])].mean()
    seg_cond = {}
    for s in order:
        m = segs == s
        if m.sum() == 0:
            continue
        ek_arr = err[m & keep]
        ea, ek = err[m].mean(), (ek_arr.mean() if len(ek_arr) else float("nan"))
        seg_cond[s] = dict(n_all=int(m.sum()), n_kept=int((m & keep).sum()),
                           surv=float(keep[m].mean()), err_all=float(ea), err_kept=float(ek),
                           gain=float(ea - ek), ratio_to_trfe=float(ek / ref) if ref else float("nan"))
        print(f"{s:>4} {m.sum():>6} {int((m&keep).sum()):>7} {100*keep[m].mean():>5.1f}% | "
              f"{ea:>8.3f} {ek:>8.3f} {ea-ek:>+7.3f} | {ek/ref:>12.2f}x")
    print(f"  (tr/fe survivors mean err = {ref:.3f}; 'vs tr/fe kept' >1 means this segment's")
    print("   survivors are still worse than tr/fe's, i.e. breadth gained without matching precision)")

    print("\nVERDICT")
    for sig in ("cosine", "cycle", "mnn", "stability"):
        r = res[sig]
        print(f"  {sig:>10}: {'USABLE' if r['USABLE'] else 'FAILS'} "
              f"(S1={r['S1']}, S2={r['S2']}, rho={r['rho']:+.3f})")

    with open(os.path.join(out_dir, "confidence_signals.json"), "w") as fh:
        json.dump({"config": vars(args), "n_rows": len(rows), "base_err": float(base),
                   "results": res, "segment_conditional": seg_cond}, fh, indent=2)
    print(f"\nwrote {out_dir}/confidence_signals.json")


if __name__ == "__main__":
    main()
