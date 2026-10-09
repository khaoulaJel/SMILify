"""F1 -- is circumferential information present in the frozen C11 backbone's features?

Bar fixed in advance in PREREGISTRATION_F1_circumferential_probe_20260828.md.

Linear (ridge) probes at four depths of the frozen backbone, predicting a CANONICAL circumferential
angle defined once on the rest template (so there is no per-specimen frame ambiguity), with an
AXIAL positive control and a shuffled chance control. Held-out specimens only.
"""
import argparse, json, os, sys

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "correspondence_accuracy"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
import labels as lb  # noqa: E402
from smil_cse_net import SMILCSENet  # noqa: E402

SEGMENTS = ("tr", "fe")  # fixed in advance: the only segments with measured circumferential signal


def segment_frame(V):
    """Principal axis + length, copied verbatim from cse_feasibility_retrieval_20260826.py."""
    X = V - V.mean(0)
    if len(X) < 3:
        return None, None
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    axis = vt[0]
    t = X @ axis
    L = float(t.max() - t.min())
    return (axis, L) if L > 1e-9 else (None, None)


def template_coords(v_template, vseg, vleg):
    """Canonical per-template-vertex (theta, axial) for every (leg, segment) group.

    theta is the angle in the plane orthogonal to the group's PCA axis, measured against an
    in-plane reference built deterministically from that axis -- so it is a fixed property of the
    template, identical for every specimen. axial is position along the axis, normalised to [0,1].
    """
    n = len(vseg)
    theta = np.full(n, np.nan)
    axial = np.full(n, np.nan)
    group = np.full(n, -1, dtype=int)
    g = 0
    for leg in sorted({l for l in vleg if l is not None}):
        for seg in SEGMENTS:
            g += 1
            m = np.array([(vleg[i] == leg and vseg[i] == seg) for i in range(n)])
            if m.sum() < 8:
                continue
            V = v_template[m]
            axis, L = segment_frame(V)
            if axis is None:
                continue
            c = V.mean(0)
            # deterministic in-plane reference: the world axis least aligned with `axis`,
            # orthogonalised. Depends only on `axis`, so it is stable across runs.
            ref = np.eye(3)[np.argmin(np.abs(axis))]
            e1 = ref - (ref @ axis) * axis
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(axis, e1)
            d = V - c
            theta[m] = np.arctan2(d @ e2, d @ e1)
            t = d @ axis
            axial[m] = (t - t.min()) / max(t.max() - t.min(), 1e-9)
            group[m] = g
    return theta, axial, group


@torch.no_grad()
def features_at_depths(model, pts):
    """pts (B,N,3) -> dict depth -> (coords (B,M,3), feats (B,M,C))."""
    xyz = pts.transpose(2, 1)
    l1_xyz, l1_points = model.sa1(xyz, None)
    l2_xyz, l2_points = model.sa2(l1_xyz, l1_points)
    l3_xyz, l3_points = model.sa3(l2_xyz, l2_points)
    d2 = model.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
    d1 = model.fp2(l1_xyz, l2_xyz, l1_points, d2)
    fp1 = model.fp1(xyz, l1_xyz, None, d1)
    return {
        "xyz": (pts, pts),
        "sa1": (l1_xyz.transpose(2, 1), l1_points.transpose(2, 1)),
        "sa2": (l2_xyz.transpose(2, 1), l2_points.transpose(2, 1)),
        "fp1": (pts, fp1.transpose(2, 1)),
    }


def ridge_fit(X, Y, lam=1.0):
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def ridge_pred(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def ang_err(pred_cs, true_theta):
    p = np.arctan2(pred_cs[:, 1], pred_cs[:, 0])
    d = np.abs(np.arctan2(np.sin(p - true_theta), np.cos(p - true_theta)))
    return float(np.degrees(d).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="diagnostics/anatomical_pose_init/out_C11_hardneg_20260826/best_model.pt")
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--out", default="diagnostics/anatomical_pose_init/out_F1_circum_probe_20260828")
    ap.add_argument("--n_train", type=int, default=60)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--n_points", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dd = config.dd
    faces = torch.tensor(np.asarray(dd["f"], dtype=np.int64), device=dev)
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    jnames = [str(x) for x in dd["J_names"]]
    vl = lb.vertex_labels(jnames, np.asarray(dd["weights"]).argmax(axis=1))
    theta_t, axial_t, group_id = template_coords(v_template, vl["leg_seg"], vl["leg_id"])
    valid_v = ~np.isnan(theta_t)
    print(f"[F1] template vertices with a circumferential coord on {SEGMENTS}: {valid_v.sum()}")

    gt = np.load(args.corpus, allow_pickle=True)
    verts = gt["verts"]
    n_val = max(1, int(len(verts) * 0.05))
    train_pool, test_pool = verts[:-n_val], verts[-n_val:]   # SAME split as C3/C11
    tr_idx = np.linspace(0, len(train_pool) - 1, args.n_train).astype(int)
    te_idx = np.linspace(0, len(test_pool) - 1, args.n_test).astype(int)

    ck = torch.load(args.ckpt, map_location=dev)
    # The checkpoint stores the weights under 'model_state_dict'. An earlier version of this script
    # looked for 'model', silently fell through to the whole checkpoint dict, and load_state_dict
    # (strict=False) then matched NOTHING -- probing a randomly initialised network. A non-zero
    # weight check does not catch that, because random weights are non-zero. So: load STRICTLY, and
    # additionally verify against a re-read of the raw tensor.
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model = SMILCSENet(n_vertices=int(ck.get("n_vertices", len(v_template))),
                       embed_dim=int(ck.get("embed_dim", 16))).to(dev)
    model.load_state_dict(sd, strict=True)          # raises if anything is missing
    k0 = "sa1.conv_blocks.0.0.weight"
    assert torch.allclose(dict(model.named_parameters())[k0].detach().cpu(), sd[k0].cpu()), \
        f"{k0} did not survive load_state_dict"
    print(f"[F1] loaded {args.ckpt} STRICTLY ({len(sd)} tensors), epoch={ck.get('epoch')}, val={ck.get('val')}")
    model.eval()

    def collect(pool, idxs):
        acc = {}
        for i in idxs:
            v = torch.as_tensor(pool[i], dtype=torch.float32, device=dev)
            mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))
            pts = sample_points_from_meshes(mesh, num_samples=args.n_points)
            tv = knn_points(pts, v.unsqueeze(0), K=1).idx[0, :, 0].cpu().numpy()
            feats = features_at_depths(model, pts)
            for depth, (coords, F_) in feats.items():
                c = coords[0].cpu().numpy()
                f = F_[0].cpu().numpy()
                if depth in ("xyz", "fp1"):
                    vidx = tv
                else:
                    # subsampled centroids: inherit the coord of their nearest TRUE vertex
                    dist = ((c[:, None, :] - pool[i][None, :, :]) ** 2).sum(-1)
                    vidx = dist.argmin(1)
                keep = valid_v[vidx]
                if keep.sum() == 0:
                    continue
                a = acc.setdefault(depth, ([], [], [], []))
                a[0].append(f[keep]); a[1].append(theta_t[vidx[keep]]); a[2].append(axial_t[vidx[keep]])
                a[3].append(group_id[vidx[keep]])
        return {k: tuple(np.concatenate(x) for x in v) for k, v in acc.items()}

    print("[F1] collecting train ..."); TR = collect(train_pool, tr_idx)
    print("[F1] collecting test  ..."); TE = collect(test_pool, te_idx)

    res = {}
    rng = np.random.default_rng(0)
    for depth in ["xyz", "sa1", "sa2", "fp1"]:
        Xtr, th_tr, ax_tr, g_tr = TR[depth]; Xte, th_te, ax_te, g_te = TE[depth]
        # ONE probe per (leg, segment) group. Pooling groups makes the map nonlinear -- theta is
        # defined in each group's OWN frame -- which a linear probe cannot represent even from raw
        # xyz. An earlier version pooled, and its raw-xyz control sat at chance for that reason.
        cs, r2s, shs = [], [], []
        for gid in np.unique(g_tr):
            mtr, mte = g_tr == gid, g_te == gid
            if mtr.sum() < 50 or mte.sum() < 20:
                continue
            A, B = Xtr[mtr], Xte[mte]
            mu, sd_ = A.mean(0), A.std(0) + 1e-8
            A, B = (A - mu) / sd_, (B - mu) / sd_
            t1, t2 = th_tr[mtr], th_te[mte]
            cs.append(ang_err(ridge_pred(ridge_fit(A, np.stack([np.cos(t1), np.sin(t1)], 1)), B), t2))
            pa = ridge_pred(ridge_fit(A, ax_tr[mtr][:, None]), B)[:, 0]
            ae = ax_te[mte]
            r2s.append(1 - ((pa - ae) ** 2).sum() / max(((ae - ae.mean()) ** 2).sum(), 1e-12))
            sh = t1[rng.permutation(len(t1))]
            shs.append(ang_err(ridge_pred(ridge_fit(A, np.stack([np.cos(sh), np.sin(sh)], 1)), B), t2))
        circ, r2 = float(np.mean(cs)), float(np.mean(r2s))
        res[depth] = dict(circ_deg=circ, axial_r2=r2, shuffled_circ_deg=float(np.mean(shs)),
                          n_groups=len(cs), dim=int(Xtr.shape[1]),
                          n_train=int(len(Xtr)), n_test=int(len(Xte)))
        print(f"  {depth:<5} dim={Xtr.shape[1]:<4} circ={circ:6.2f}deg  axial R2={r2:+.3f}  "
              f"shuffled={res[depth]['shuffled_circ_deg']:6.2f}deg")

    json.dump(res, open(os.path.join(args.out, "f1_probe.json"), "w"), indent=2)
    print(f"\nwrote {args.out}/f1_probe.json   (chance = 90 deg)")


if __name__ == "__main__":
    main()
