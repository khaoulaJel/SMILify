"""F5 -- split-conformal calibration of a ground-truth-free fit-quality signal.

Bar in REGISTER_F_series_20260828.md, fixed before this ran.

WHY: nothing in SMILify outputs uncertainty, and on a real scan there is no way to know a fit is
wrong. The catastrophic specimens in this investigation (leg_acc 0.366, 0.560) were found only
because synthetic data had ground truth. Two ingredients already exist -- a GT-free triage signal
that works (cycle-consistency degrades per-specimen, LAB_RECORD section 8) and synthetic specimens
with ground truth to calibrate against.

Split conformal gives distribution-free coverage under exchangeability: calibrate the signal
against measured error on one split, emit intervals on a disjoint split, and the coverage
guarantee holds without assuming the error distribution.
"""
import argparse, json, os, sys

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "fitter_3d"), os.path.join(REPO, "fitter_3d", "pointcloud2smil"),
          os.path.join(REPO, "diagnostics", "correspondence_accuracy"), REPO):
    sys.path.insert(0, p)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
import labels as lb  # noqa: E402
from smil_cse_net import SMILCSENet  # noqa: E402

SEGS = ("tr", "fe", "co", "ti", "ta", "pt")


def collect(model, keys, v_template, verts_list, faces, seg_of_v, L, n_points, dev):
    """Per (specimen, segment): the GT-FREE signals, and the true error they must predict."""
    rows = []
    for i, vv in enumerate(verts_list):
        v = torch.as_tensor(vv, dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=n_points)
        tv = knn_points(pts, v.unsqueeze(0), K=1).idx[0, :, 0].cpu().numpy()
        P = pts[0].cpu().numpy().astype(np.float64)
        with torch.no_grad():
            sim = model(pts.to(dev))[0] @ keys.t()
            top2 = sim.topk(2, dim=1)
            pred = top2.indices[:, 0].cpu().numpy()
            margin = (top2.values[:, 0] - top2.values[:, 1]).cpu().numpy()
            back = sim.argmax(0).cpu().numpy()          # best point for each vertex
        cyc = np.linalg.norm(P[back[pred]] - P, axis=1)  # cycle-consistency distance, GT-free
        for s in SEGS:
            m = (seg_of_v[tv] == s) & np.isfinite(L[tv])
            if m.sum() < 20:
                continue
            err = np.median(np.linalg.norm(v_template[pred[m]] - v_template[tv[m]], axis=1) / L[tv[m]])
            rows.append(dict(spec=i, seg=s,
                             cycle=float(np.median(cyc[m])), margin=float(np.median(margin[m])),
                             err=float(err)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="diagnostics/anatomical_pose_init/out_C11_hardneg_20260826/best_model.pt")
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--out", default="diagnostics/anatomical_pose_init/out_F5_conformal_20260828")
    ap.add_argument("--n_cal", type=int, default=70)
    ap.add_argument("--n_test", type=int, default=70)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--alpha", type=float, default=0.10, help="target miscoverage; 0.10 => 90%")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dd = config.dd
    faces = torch.tensor(np.asarray(dd["f"], dtype=np.int64))
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    vl = lb.vertex_labels([str(x) for x in dd["J_names"]], np.asarray(dd["weights"]).argmax(axis=1))
    seg_of_v = np.array([s if s is not None else "" for s in vl["leg_seg"]], dtype=object)
    leg_of_v = np.array([s if s is not None else "" for s in vl["leg_id"]], dtype=object)
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
    model.load_state_dict(sd, strict=True)
    k0 = "sa1.conv_blocks.0.0.weight"
    assert torch.allclose(dict(model.named_parameters())[k0].detach().cpu(), sd[k0].cpu())
    print(f"[F5] loaded {args.ckpt} STRICTLY ({len(sd)} tensors)")
    model.eval()
    with torch.no_grad():
        keys = model.vertex_embeddings()

    gt = np.load(args.corpus, allow_pickle=True)["verts"]
    tail = gt[-max(1, int(len(gt) * 0.05)):]
    assert args.n_cal + args.n_test <= len(tail)
    # SPECIMEN-level split: calibration and test share no specimen.
    cal_v, te_v = tail[:args.n_cal], tail[args.n_cal:args.n_cal + args.n_test]
    print(f"[F5] specimen-level split: {len(cal_v)} calibration / {len(te_v)} test")

    cal = collect(model, keys, v_template, cal_v, faces, seg_of_v, L, args.n_points, dev)
    te = collect(model, keys, v_template, te_v, faces, seg_of_v, L, args.n_points, dev)

    # Predictor: ridge on the two GT-free signals, fit on calibration only.
    def X(rows): return np.stack([[r["cycle"], r["margin"], 1.0] for r in rows])
    Xc, yc = X(cal), np.array([r["err"] for r in cal])
    W = np.linalg.solve(Xc.T @ Xc + 1e-6 * np.eye(3), Xc.T @ yc)
    resid = np.abs(yc - Xc @ W)
    n = len(resid)
    q = float(np.quantile(resid, min(1.0, np.ceil((n + 1) * (1 - args.alpha)) / n)))  # conformal quantile

    Xt, yt = X(te), np.array([r["err"] for r in te])
    pred = Xt @ W
    lo, hi = pred - q, pred + q
    covered = (yt >= lo) & (yt <= hi)
    width = float(2 * q)

    # Signal-free baseline: constant predictor, same conformal construction.
    resid0 = np.abs(yc - yc.mean())
    q0 = float(np.quantile(resid0, min(1.0, np.ceil((n + 1) * (1 - args.alpha)) / n)))
    cov0 = float(((yt >= yc.mean() - q0) & (yt <= yc.mean() + q0)).mean())
    width_ratio = width / (2 * q0)

    print(f"\n[endpoint] 90% interval: coverage {covered.mean():.3f} (bar 0.85-0.95), "
          f"width {width:.4f}")
    print(f"[endpoint] signal-free constant baseline: coverage {cov0:.3f}, width {2*q0:.4f}  "
          f"-> width ratio {width_ratio:.3f} (bar < 0.80)")
    c_cov = 0.85 <= covered.mean() <= 0.95
    c_w = width_ratio < 0.80
    print(f"\n[mechanism] coverage per segment (marginal coverage can hide segment failure):")
    seg_cov = {}
    for s in SEGS:
        m = np.array([r["seg"] == s for r in te])
        if m.sum() == 0:
            continue
        seg_cov[s] = float(covered[m].mean())
        print(f"    {s}: {covered[m].mean():.3f}  (n={int(m.sum())})")
    worst = np.argsort(-yt)[:max(1, len(yt) // 10)]
    wc = float(covered[worst].mean())
    print(f"[mechanism] worst-decile coverage: {wc:.3f}  "
          f"({'ok' if wc >= 0.70 else 'FAILS WHERE IT MATTERS -- the specimens a user needs flagged'})")
    verdict = "PASS" if (c_cov and c_w) else "FAIL"
    print(f"\n>>> F5 VERDICT: {verdict}  (coverage {'ok' if c_cov else 'out of band'}, "
          f"width {'ok' if c_w else 'not informative'})")
    json.dump(dict(alpha=args.alpha, coverage=float(covered.mean()), width=width,
                   baseline_coverage=cov0, baseline_width=float(2 * q0),
                   width_ratio=width_ratio, per_segment_coverage=seg_cov,
                   worst_decile_coverage=wc, n_cal=len(cal), n_test=len(te), verdict=verdict),
              open(os.path.join(args.out, "f5_conformal.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
