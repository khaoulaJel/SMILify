"""F2 Tier 1 -- a TRAINED 1-of-5 selector. Bar in PREREGISTRATION_F2_trained_selector_20260828.md.

Every prior ranking attempt used ONE existing heuristic directly as the ranker. This trains a
classifier over those signals COMBINED, supervised by which candidate is actually nearest the true
vertex. Candidate generation is copied from multihypothesis_feasibility_20260827.py so Tier 1 is
directly comparable to the cycle/coherence baselines measured there.

Split is at the SPECIMEN level (points from one specimen are not independent), over the corpus tail
the network itself never trained on. Scored PER SEGMENT -- a pooled average is not a pass.
"""
import argparse, json, os, sys

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "fitter_3d"), os.path.join(REPO, "fitter_3d", "pointcloud2smil"),
          os.path.join(REPO, "diagnostics", "correspondence_accuracy"), REPO):
    sys.path.insert(0, p)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
import labels as lb  # noqa: E402
from smil_cse_net import SMILCSENet  # noqa: E402

SEGS = ("tr", "fe")
FEATS = ["sim", "margin_top", "margin_next", "cyc_d", "cyc_rank", "coh_d", "coh_rank",
         "nb_votes", "cand_rank", "sim_spread", "cyc_spread"]


def seg_len_per_vertex(v_template, vseg, vleg):
    """Per-template-vertex segment length, so retrieval error is expressed as a FRACTION OF SEGMENT
    LENGTH -- the exact normalisation multihypothesis_feasibility_20260827.py used. Without this,
    Tier 1's 'gap closed' is not comparable to the 27.0%/15.7% cycle baselines it is barred against.
    """
    L = np.full(len(v_template), np.nan)
    for leg in sorted({x for x in vleg if x is not None}):
        for seg in SEGS:
            m = np.array([(vleg[i] == leg and vseg[i] == seg) for i in range(len(v_template))])
            if m.sum() < 3:
                continue
            X = v_template[m] - v_template[m].mean(0)
            axis = np.linalg.svd(X, full_matrices=False)[2][0]
            t = X @ axis
            ln = float(t.max() - t.min())
            if ln > 1e-9:
                L[m] = ln
    return L


def load_model(ckpt, n_v, dev):
    ck = torch.load(ckpt, map_location=dev)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    m = SMILCSENet(n_vertices=int(ck.get("n_vertices", n_v)), embed_dim=int(ck.get("embed_dim", 16))).to(dev)
    m.load_state_dict(sd, strict=True)          # standing rule: loading must be verified
    k0 = "sa1.conv_blocks.0.0.weight"
    assert torch.allclose(dict(m.named_parameters())[k0].detach().cpu(), sd[k0].cpu()), "load failed"
    print(f"[F2] loaded {ckpt} STRICTLY ({len(sd)} tensors) epoch={ck.get('epoch')}")
    return m.eval()


def build(model, keys, v_template, verts_list, faces, seg_of_v, seg_L, k, n_points, knn, dev):
    """-> per-candidate feature rows + labels + bookkeeping, for the given specimens."""
    X, y, seg, spec, cerr = [], [], [], [], []
    for si, vv in enumerate(verts_list):
        v = torch.as_tensor(vv, dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=n_points)[0]
        tv = knn_points(pts.unsqueeze(0), v.unsqueeze(0), K=1).idx[0, :, 0].numpy()
        P = pts.numpy().astype(np.float64)
        with torch.no_grad():
            sim = model(pts.unsqueeze(0).to(dev))[0] @ keys.t()
            topv, topi = sim.topk(k, dim=1)
            back = sim.argmax(0)
        topi, topv, backn = topi.cpu().numpy(), topv.cpu().numpy(), back.cpu().numpy()
        nb = knn_points(pts.unsqueeze(0), pts.unsqueeze(0), K=knn + 1).idx[0, :, 1:].numpy()

        top1 = topi[:, 0]
        nb_mean = v_template[top1[nb]].mean(1)
        cand_xyz = v_template[topi]                                     # (N,k,3)
        coh_d = np.linalg.norm(cand_xyz - nb_mean[:, None, :], axis=2)  # (N,k)
        cyc_d = np.linalg.norm(P[backn[topi]] - P[:, None, :], axis=2)  # (N,k)
        # local consistency as a COUNT, distinct from coh_d's centroid distance: how many
        # neighbours' own top-1 picks coincide with this candidate.
        nb_votes = (topi[:, None, :] == top1[nb][:, :, None]).sum(1).astype(np.float64)

        F = np.stack([
            topv,
            topv - topv[:, :1],                                         # margin vs the top pick
            topv - np.sort(topv, 1)[:, -2:-1],                          # margin vs runner-up
            cyc_d, cyc_d.argsort(1).argsort(1).astype(float),
            coh_d, coh_d.argsort(1).argsort(1).astype(float),
            nb_votes,
            np.tile(np.arange(k, dtype=float), (len(topi), 1)),
            np.tile((topv.max(1) - topv.min(1))[:, None], (1, k)),
            np.tile((cyc_d.max(1) - cyc_d.min(1))[:, None], (1, k)),
        ], axis=2)                                                       # (N,k,F)

        e = np.linalg.norm(cand_xyz - v_template[tv][:, None, :], axis=2) / seg_L[tv][:, None]
        best = e.argmin(1)
        sv = seg_of_v[tv]
        m = np.isin(sv, SEGS) & np.isfinite(seg_L[tv])
        X.append(F[m].reshape(-1, F.shape[2]))
        y.append((np.arange(k)[None, :] == best[m][:, None]).ravel().astype(int))
        seg.append(np.repeat(sv[m], k)); spec.append(np.full(m.sum() * k, si)); cerr.append(e[m].ravel())
    return (np.concatenate(X), np.concatenate(y), np.concatenate(seg),
            np.concatenate(spec), np.concatenate(cerr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="diagnostics/anatomical_pose_init/out_C11_hardneg_20260826/best_model.pt")
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--out", default="diagnostics/anatomical_pose_init/out_F2_selector_20260828")
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--knn", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dd = config.dd
    faces = torch.tensor(np.asarray(dd["f"], dtype=np.int64))
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    vl = lb.vertex_labels([str(x) for x in dd["J_names"]], np.asarray(dd["weights"]).argmax(axis=1))
    seg_of_v = np.array([s if s is not None else "" for s in vl["leg_seg"]], dtype=object)
    seg_L = seg_len_per_vertex(v_template, vl["leg_seg"], vl["leg_id"])

    model = load_model(args.ckpt, len(v_template), dev)
    with torch.no_grad():
        keys = model.vertex_embeddings()

    gt = np.load(args.corpus, allow_pickle=True)["verts"]
    tail = gt[-max(1, int(len(gt) * 0.05)):]     # never seen by the network
    assert args.n_train + args.n_test <= len(tail), f"tail has only {len(tail)} specimens"
    # SPECIMEN-level split: disjoint specimens, not disjoint points.
    tr_v, te_v = tail[:args.n_train], tail[args.n_train:args.n_train + args.n_test]
    print(f"[F2] specimen-level split: {len(tr_v)} train / {len(te_v)} test (disjoint specimens)")

    Xtr, ytr, str_, _, _ = build(model, keys, v_template, tr_v, faces, seg_of_v, seg_L,
                                 args.topk, args.n_points, args.knn, dev)
    Xte, yte, ste, spte, cete = build(model, keys, v_template, te_v, faces, seg_of_v, seg_L,
                                      args.topk, args.n_points, args.knn, dev)
    print(f"[F2] rows: train {len(Xtr)}  test {len(Xte)}")

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
    clf.fit(Xtr, ytr)
    score = clf.predict_proba(Xte)[:, 1]

    k = args.topk
    res = {}
    print(f"\n{'seg':>4} {'top1':>8} {'oracle':>8} {'cycle':>8} {'LEARNED':>9} | "
          f"{'cycle gap':>10} {'LEARNED gap':>12} {'rank acc':>9}")
    for s in SEGS:
        m = ste == s
        E = cete[m].reshape(-1, k); S = score[m].reshape(-1, k); F = Xte[m].reshape(-1, k, Xte.shape[1])
        cyc_rank = F[:, :, FEATS.index("cyc_rank")]
        top1 = E[:, 0].mean(); orc = E.min(1).mean()
        cyc = E[np.arange(len(E)), cyc_rank.argmin(1)].mean()
        lrn = E[np.arange(len(E)), S.argmax(1)].mean()
        gc = lambda x: (top1 - x) / (top1 - orc) * 100
        ra = float((S.argmax(1) == E.argmin(1)).mean())
        res[s] = dict(top1=float(top1), oracle=float(orc), cycle=float(cyc), learned=float(lrn),
                      cycle_gap_pct=float(gc(cyc)), learned_gap_pct=float(gc(lrn)),
                      rank_acc=ra, chance=1.0 / k, n_points=int(len(E)))
        print(f"{s:>4} {top1:8.4f} {orc:8.4f} {cyc:8.4f} {lrn:9.4f} | "
              f"{gc(cyc):9.1f}% {gc(lrn):11.1f}% {ra:9.3f}")
    res["_bar"] = "Tier 1 PASS requires learned_gap_pct >= 40 on BOTH tr and fe individually"
    for s in SEGS:
        print(f"  {s}: Tier1 {'PASS' if res[s]['learned_gap_pct'] >= 40 else 'FAIL'} "
              f"({res[s]['learned_gap_pct']:.1f}% vs bar 40%, cycle {res[s]['cycle_gap_pct']:.1f}%)")
    json.dump(res, open(os.path.join(args.out, "f2_tier1.json"), "w"), indent=2)
    print(f"\nwrote {args.out}/f2_tier1.json")


if __name__ == "__main__":
    main()
