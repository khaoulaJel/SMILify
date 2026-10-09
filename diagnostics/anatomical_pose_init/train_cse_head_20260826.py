"""C3 -- train the CSE-style embedding head (`SMILCSENet`) with a SurfEmb-style contrastive loss.

Replaces ONLY the decode head + loss of the B2 setup. Data pipeline, sampler, corpus and the
held-out split are byte-identical to `train_correspondence_net_B2_20260825.py`, so a C3-vs-B2
comparison is not confounded by data changes:
  * same corpus `synth_b2_train_corrb06.npz`, same `val_frac=0.05` tail as validation,
  * same on-the-fly labelling rule (sample the posed surface, nearest TRUE vertex via knn),
  * same n_points=2048.

The ONE deliberate difference in supervision: B2 regressed (segment class, chain_pos). C3 is
supervised by the TRUE VERTEX INDEX itself -- which is what `Dense_GT_oracle` fed the fitter when
it produced the only validated correspondence win in this investigation (seg_acc 0.832->0.874,
12/12 specimens). C3 therefore targets the quantity already proven to matter, instead of a
hand-designed coordinate proven insufficient.

Warm-starts the backbone from B2's trained checkpoint (asserted non-zero, so a bad path fails loudly
rather than silently training from scratch).

Validation reports HELD-OUT full-vocabulary retrieval accuracy -- argmax over ALL V template
vertices, not over the sampled negatives the training loss uses -- plus the median 3D error of the
retrieved vertex in canonical rest space. The in-loss accuracy is an optimistic proxy and is logged
only to watch convergence.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from smil_correspondence_net import load_model_dict  # noqa: E402
from smil_cse_net import (  # noqa: E402
    SMILCSENet,
    build_circumferential_hard_pool,
    build_geodesic_soft_targets,
    info_nce_loss,
    info_nce_loss_hard,
    info_nce_soft_loss,
    load_backbone_from_correspondence_ckpt,
)


class VertexCorrespondenceDataset(Dataset):
    """Identical sampling/labelling to B2's CorrespondenceDataset, but returns the TRUE VERTEX
    INDEX per point instead of (segment class, chain_pos)."""

    def __init__(self, npz_path, faces, n_points=2048, split="train", val_frac=0.05):
        gt = np.load(npz_path, allow_pickle=True)
        verts = gt["verts"]
        n_val = max(1, int(len(verts) * val_frac))
        self.verts = verts[:-n_val] if split == "train" else verts[-n_val:]
        self.faces = faces
        self.n_points = n_points

    def __len__(self):
        return len(self.verts)

    def __getitem__(self, idx):
        v = torch.as_tensor(self.verts[idx], dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=self.faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=self.n_points)[0]
        true_vert = knn_points(pts.unsqueeze(0), v.unsqueeze(0), K=1).idx[0, :, 0]
        return pts, true_vert


@torch.no_grad()
def validate(model, loader, device, v_template_t, chunk=4096):
    """Held-out FULL-vocabulary retrieval: argmax over all V vertices.

    Reports top-1 / top-5 vertex accuracy and the median canonical-space 3D error of the retrieved
    vertex. Exact-vertex accuracy is a harsh metric on a dense mesh (neighbouring vertices are
    nearly indistinguishable), so the 3D error is the one to weigh -- it says how far off the
    retrieved correspondence lands, which is what the fitter actually consumes.
    """
    model.eval()
    keys = model.vertex_embeddings()                       # (V,D) unit-norm
    tot = top1 = top5 = 0
    err = []
    for pts, true_vert in loader:
        pts, true_vert = pts.to(device), true_vert.to(device)
        q = model(pts).reshape(-1, model.embed_dim)        # (B*N,D)
        tv = true_vert.reshape(-1)
        for s in range(0, q.shape[0], chunk):
            sim = q[s:s + chunk] @ keys.t()                # (c,V)
            t = tv[s:s + chunk]
            top = sim.topk(5, dim=1).indices
            top1 += (top[:, 0] == t).sum().item()
            top5 += (top == t.unsqueeze(1)).any(1).sum().item()
            err.append(torch.norm(v_template_t[top[:, 0]] - v_template_t[t], dim=-1).cpu())
            tot += t.numel()
    err = torch.cat(err)
    return {"top1": top1 / tot, "top5": top5 / tot,
            "median_3d_err": float(err.median()), "mean_3d_err": float(err.mean()), "n": int(tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826")
    ap.add_argument("--warm_start", default="diagnostics/anatomical_pose_init/out_B2_correspondence_net_20260826/best_model.pt")
    ap.add_argument("--no_warm_start", action="store_true", help="ablation: train the backbone from scratch")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--embed_dim", type=int, default=16)
    ap.add_argument("--n_negatives", type=int, default=4096)
    ap.add_argument("--soft_targets", action="store_true",
                    help="F6: geodesic-softened targets instead of one-hot vertex identity")
    ap.add_argument("--soft_neighbors", type=int, default=24)
    ap.add_argument("--soft_sigma", type=float, default=0.0, help="0 = read from the mesh")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--hard_neg", action="store_true",
                    help="mine CIRCUMFERENTIAL hard negatives: same segment, similar axial position. "
                         "Justified because the LRF check showed the circumferential signal IS "
                         "present in raw geometry (circum_gap 0.38-0.44 vs a cylinder's 0.031 floor) "
                         "while retrieval sits near chance -- i.e. present but unlearned, a training "
                         "problem rather than an architecture one. Uniform negatives are nearly all "
                         "on other segments and are solved long before circumferential detail matters.")
    ap.add_argument("--hard_frac", type=float, default=0.5)
    ap.add_argument("--axial_band", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dd = load_model_dict(config.SMAL_FILE)
    v_template = np.asarray(dd["v_template"], dtype=np.float32)
    n_vertices = v_template.shape[0]
    v_template_t = torch.as_tensor(v_template, device=device)

    # F6: geodesic-softened targets. Default off, so every prior arm's behaviour is unchanged.
    soft_idx = soft_w = None
    if args.soft_targets:
        import sys as _sys
        _sys.path.insert(0, os.path.join(REPO, "diagnostics", "correspondence_accuracy"))
        import labels as _lb
        _vl = _lb.vertex_labels([str(x) for x in dd["J_names"]],
                                np.asarray(dd["weights"]).argmax(axis=1))
        _grp = np.array([x if x is not None else "" for x in _vl["leg_id"]], dtype=object)
        soft_idx, soft_w = build_geodesic_soft_targets(
            np.asarray(dd["v_template"]), np.asarray(dd["f"]),
            n_neighbors=args.soft_neighbors, sigma=(args.soft_sigma or None), group=_grp)

    hard_pool = None
    if args.hard_neg:
        sys.path.insert(0, HERE)
        from smil_correspondence_net import build_segment_taxonomy, build_vertex_labels
        from cse_feasibility_retrieval_20260826 import segment_frame
        cn, n2i = build_segment_taxonomy(list(dd["J_names"]))
        sci, _ = build_vertex_labels(dd, cn, n2i)
        seg_axis = {}
        for cid in range(1, len(cn)):
            ax, L = segment_frame(np.asarray(dd["v_template"], dtype=np.float64)[sci == cid])
            if ax is not None:
                seg_axis[cid] = (ax, L)
        hard_pool = build_circumferential_hard_pool(
            np.asarray(dd["v_template"], dtype=np.float64), sci, seg_axis,
            axial_band=args.axial_band).to(device)
        n_with = int((hard_pool >= 0).any(1).sum())
        print(f"[hard-neg] circumferential pools for {n_with}/{n_vertices} vertices "
              f"(mean pool size {float((hard_pool >= 0).sum(1).float().mean()):.1f}), "
              f"hard_frac={args.hard_frac}, axial_band={args.axial_band}", flush=True)

    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].long()

    corpus = os.path.join(REPO, args.train_corpus)
    train_ds = VertexCorrespondenceDataset(corpus, faces, args.n_points, "train", args.val_frac)
    val_ds = VertexCorrespondenceDataset(corpus, faces, args.n_points, "val", args.val_frac)
    persist = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, persistent_workers=persist)

    model = SMILCSENet(n_vertices=n_vertices, embed_dim=args.embed_dim,
                       temperature=args.temperature).to(device)
    if not args.no_warm_start:
        copied, skipped = load_backbone_from_correspondence_ckpt(
            model, os.path.join(REPO, args.warm_start), map_location=device)
        if copied == 0:
            raise SystemExit(f"warm start copied 0 tensors from {args.warm_start} -- refusing to "
                             f"train from scratch silently. Pass --no_warm_start if intended.")
        print(f"[warm-start] copied {copied} backbone tensors from B2 ({skipped} skipped)", flush=True)
    else:
        print("[warm-start] DISABLED (--no_warm_start): backbone trains from scratch", flush=True)

    print(f"[setup] V={n_vertices} embed_dim={args.embed_dim} negatives={args.n_negatives} "
          f"train={len(train_ds)} val={len(val_ds)} device={device}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    history, best = [], -1.0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl = ta = nb = 0
        for pts, true_vert in train_loader:
            pts, true_vert = pts.to(device), true_vert.to(device)
            if soft_idx is not None:
                loss, acc = info_nce_soft_loss(model(pts), true_vert, model, soft_idx, soft_w,
                                               args.n_negatives)
            elif hard_pool is not None:
                loss, acc = info_nce_loss_hard(model(pts), true_vert, model, hard_pool,
                                               args.hard_frac, args.n_negatives)
            else:
                loss, acc = info_nce_loss(model(pts), true_vert, model, args.n_negatives)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item(); ta += acc.item(); nb += 1
        rec = {"epoch": ep, "train_loss": tl / nb, "train_sampled_acc": ta / nb,
               "logit_scale": float(model.logit_scale()), "elapsed": time.time() - t0}

        if ep % args.val_every == 0 or ep == args.epochs - 1:
            v = validate(model, val_loader, device, v_template_t)
            rec.update({f"val_{k}": val for k, val in v.items()})
            sched.step(-v["top1"])
            if v["top1"] > best:
                best = v["top1"]
                torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                            "val": v, "args": vars(args), "n_vertices": n_vertices,
                            "embed_dim": args.embed_dim},
                           os.path.join(out_dir, "best_model.pt"))
            print(f"[{ep:3d}] loss={rec['train_loss']:.4f} sampled_acc={rec['train_sampled_acc']:.3f} "
                  f"| HELD-OUT top1={v['top1']:.4f} top5={v['top5']:.4f} "
                  f"med3d={v['median_3d_err']:.5f} | {rec['elapsed']:.0f}s", flush=True)
        else:
            print(f"[{ep:3d}] loss={rec['train_loss']:.4f} sampled_acc={rec['train_sampled_acc']:.3f}",
                  flush=True)

        history.append(rec)
        with open(os.path.join(out_dir, "training_history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

    print(f"done. best held-out top1={best:.4f}. wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
