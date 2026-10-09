"""F4 -- Sinkhorn / optimal-transport assignment instead of per-point argmax.

Bar in REGISTER_F_series_20260828.md, fixed before this ran.

F2 established that per-point INDEPENDENT selection is a dead end: a trained ranker beat chance on
rank accuracy and bought ~3% of the oracle gap. OT attacks the structure that exposed -- points
compete for template vertices, so the assignment is globally consistent rather than N independent
argmaxes. Inference-only: operates on the frozen C11 head's own similarity matrix.
"""
import argparse, json, os, sys

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy import stats

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


def sinkhorn(logits, n_iter=50, eps=1.0, tau=1.0):
    """UNBALANCED entropic OT on (N,V) scores, log-domain.

    CORRECTION (2026-08-28), recorded rather than silently fixed: the first version used BALANCED
    OT with a uniform column marginal over all V template vertices. That forces N=2048 sampled
    points to spread across 10,235 vertices, which is not the problem being modelled, and its
    low-eps limit is the constrained OT solution -- with no reason to resemble argmax. The
    registered void control fired at 0.096 agreement and the arm was recorded VOID: it never
    validly tested the hypothesis, so its numbers were not read as a result.

    Unbalanced OT is both the correct formulation and makes the void control meaningful: the column
    marginal is a KL PENALTY of strength `tau` rather than a hard constraint, so tau -> 0 removes
    the coupling between points and provably recovers per-row argmax, while larger tau makes points
    compete for template vertices -- which is the structure F2 showed was missing.
    """
    N, V = logits.shape
    C = logits / eps
    g = torch.zeros(V, device=logits.device)
    log_b = torch.full((V,), -np.log(V), device=logits.device)
    lam = tau / (tau + 1.0)              # tau=0 -> lam=0 -> no column update -> argmax
    for _ in range(n_iter):
        f = -torch.logsumexp(C + g[None, :], dim=1)
        g = lam * (log_b - torch.logsumexp(C + f[:, None], dim=0))
    return C + f[:, None] + g[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="diagnostics/anatomical_pose_init/out_C11_hardneg_20260826/best_model.pt")
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--out", default="diagnostics/anatomical_pose_init/out_F4_sinkhorn_20260828")
    ap.add_argument("--n_specimens", type=int, default=40)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=1.0, help="column-marginal KL strength; 0 == argmax")
    ap.add_argument("--n_iter", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dd = config.dd
    faces = torch.tensor(np.asarray(dd["f"], dtype=np.int64))
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    vl = lb.vertex_labels([str(x) for x in dd["J_names"]], np.asarray(dd["weights"]).argmax(axis=1))
    seg_of_v = np.array([s if s is not None else "" for s in vl["leg_seg"]], dtype=object)
    leg_of_v = np.array([s if s is not None else "" for s in vl["leg_id"]], dtype=object)

    # per-(leg,segment) length, so error is a fraction of segment length -- same normalisation as
    # multihypothesis_feasibility and F2, so numbers are comparable across arms.
    L = np.full(len(v_template), np.nan)
    for lg in sorted({x for x in leg_of_v if x}):
        for sg in SEGS:
            m = (leg_of_v == lg) & (seg_of_v == sg)
            if m.sum() < 3:
                continue
            X = v_template[m] - v_template[m].mean(0)
            t = X @ np.linalg.svd(X, full_matrices=False)[2][0]
            L[m] = float(t.max() - t.min())

    ck = torch.load(args.ckpt, map_location=dev)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model = SMILCSENet(n_vertices=int(ck.get("n_vertices", len(v_template))),
                       embed_dim=int(ck.get("embed_dim", 16))).to(dev)
    model.load_state_dict(sd, strict=True)     # standing rule: verify the load
    k0 = "sa1.conv_blocks.0.0.weight"
    assert torch.allclose(dict(model.named_parameters())[k0].detach().cpu(), sd[k0].cpu())
    print(f"[F4] loaded {args.ckpt} STRICTLY ({len(sd)} tensors) epoch={ck.get('epoch')}")
    model.eval()
    with torch.no_grad():
        keys = model.vertex_embeddings()

    gt = np.load(args.corpus, allow_pickle=True)["verts"]
    tail = gt[-max(1, int(len(gt) * 0.05)):]
    idxs = np.linspace(0, len(tail) - 1, args.n_specimens).astype(int)
    print(f"[F4] {len(idxs)} held-out specimens, eps={args.eps}, {args.n_iter} iters")

    rec = {s: {"argmax": [], "ot": []} for s in SEGS}
    ent, cov = [], []
    for i in idxs:
        v = torch.as_tensor(tail[i], dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=args.n_points)
        tv = knn_points(pts, v.unsqueeze(0), K=1).idx[0, :, 0].cpu().numpy()
        with torch.no_grad():
            sim = (model(pts.to(dev))[0] @ keys.t())
            am = sim.argmax(1).cpu().numpy()
            P = sinkhorn(sim, args.n_iter, args.eps, args.tau)
            ot = P.argmax(1).cpu().numpy()
            p = P.exp(); p = p / p.sum(1, keepdim=True).clamp_min(1e-30)
            ent.append(float((-(p * p.clamp_min(1e-30).log()).sum(1)).mean()))
        cov.append(len(np.unique(ot)) / len(np.unique(am)))
        for s in SEGS:
            m = (seg_of_v[tv] == s) & np.isfinite(L[tv])
            if m.sum() < 8:
                continue
            e = lambda pr: np.linalg.norm(v_template[pr[m]] - v_template[tv[m]], axis=1) / L[tv[m]]
            rec[s]["argmax"].append(float(np.median(e(am))))
            rec[s]["ot"].append(float(np.median(e(ot))))

    # void control: eps -> 0 must approach argmax
    with torch.no_grad():
        agree = float((sinkhorn(sim, args.n_iter, args.eps, 0.0).argmax(1).cpu().numpy() == am).mean())

    res = {"eps": args.eps, "tau": args.tau, "n_specimens": len(idxs),
           "void_control_lowreg_agreement_with_argmax": agree,
           "mean_assignment_entropy": float(np.mean(ent)),
           "vertex_coverage_ratio_ot_over_argmax": float(np.mean(cov))}
    print(f"\n[void control] tau=0 Sinkhorn agrees with argmax on {agree:.3f} of points "
          f"-> {'ok' if agree > 0.9 else 'ARM VOID: implementation wrong'}")
    print(f"[mechanism] mean assignment entropy {np.mean(ent):.3f} nats; "
          f"OT uses {np.mean(cov):.3f}x the template vertices argmax does "
          f"({'ok' if np.mean(cov) > 0.9 else 'COLLAPSE -- degenerate solution'})")
    print(f"\n{'seg':>4} {'argmax':>9} {'OT':>9} {'rel':>8} {'better':>8} {'sign p':>9}")
    for s in SEGS:
        a, b = np.array(rec[s]["argmax"]), np.array(rec[s]["ot"])
        d = b - a
        sg = stats.binomtest(int((d < 0).sum()), len(d[d != 0]), 0.5).pvalue if (d != 0).any() else 1.0
        rel = (a.mean() - b.mean()) / a.mean() * 100
        ok = rel >= 5 and sg < 0.05
        res[s] = dict(argmax=float(a.mean()), ot=float(b.mean()), rel_pct=float(rel),
                      better=int((d < 0).sum()), n=len(d), sign_p=float(sg), passes=bool(ok))
        print(f"{s:>4} {a.mean():9.4f} {b.mean():9.4f} {rel:7.1f}% {int((d<0).sum()):4d}/{len(d):<3d} {sg:9.4f}"
              f"  -> {'PASS' if ok else 'fail'}")
    res["verdict"] = "PASS" if all(res[s]["passes"] for s in SEGS) else "FAIL"
    print(f"\n>>> F4 VERDICT: {res['verdict']}  (bar: >=5% relative on BOTH segments, sign p<0.05)")
    json.dump(res, open(os.path.join(args.out, "f4_sinkhorn.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
