"""B2 -- train Variant 1 (SMILCorrespondenceNet, B1) on A1's corrected-sampler corpus, supervised
by A2's validated dense-correspondence label rule (nearest TRUE vertex in 3D space).

Per-stage loss-component logging (cls/pos, train/val) mirrors the existing `chamfer/edge/lap/
sym/limit/mid/dense_gt`-style logging convention already used elsewhere in this repo
(`trainer_hierarchical.py`), so loss composition here is auditable the same way every other run
in this investigation has been -- not a bespoke, unaudited training loop.

Labels are generated ON THE FLY per batch (not precomputed once), using the SAME KNN-nearest-
vertex rule A2 validated (`validate_dense_label_pipeline_A2_20260825.py`): sample points from a
specimen's own posed mesh surface, find each point's nearest TRUE vertex on that SAME mesh, read
off that vertex's (pose-invariant) segment-class/chain-position labels. A2 already quantified this
rule's floor (~0.8% wrong-segment rate) -- this training run inherits that floor as label noise,
not a new unvalidated assumption.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
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
from smil_correspondence_net import (  # noqa: E402
    SMILCorrespondenceNet,
    build_segment_taxonomy,
    build_vertex_labels,
    correspondence_loss,
    load_model_dict,
)


class CorrespondenceDataset(Dataset):
    def __init__(self, npz_path, faces, seg_class_id, chain_pos, n_points=2048, split="train", val_frac=0.05):
        gt = np.load(npz_path, allow_pickle=True)
        verts = gt["verts"]  # (N,V,3)
        n = len(verts)
        n_val = max(1, int(n * val_frac))
        self.verts = verts[:-n_val] if split == "train" else verts[-n_val:]
        self.faces = faces  # (F,3) torch long, fixed topology
        self.seg_class_id = torch.as_tensor(seg_class_id, dtype=torch.long)  # (V,)
        self.chain_pos = torch.as_tensor(chain_pos, dtype=torch.float32)  # (V,)
        self.n_points = n_points

    def __len__(self):
        return len(self.verts)

    def __getitem__(self, idx):
        v = torch.as_tensor(self.verts[idx], dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=self.faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=self.n_points)[0]  # (n_points,3)
        nn_idx = knn_points(pts.unsqueeze(0), v.unsqueeze(0), K=1).idx[0, :, 0]  # (n_points,)
        seg_true = self.seg_class_id[nn_idx]
        chain_pos_true = self.chain_pos[nn_idx]
        return pts, seg_true, chain_pos_true


def compute_class_weights(seg_class_id, n_classes):
    counts = np.bincount(seg_class_id, minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (n_classes * counts)
    return torch.as_tensor(w, dtype=torch.float32)


def run_epoch(model, loader, optimizer, device, class_weight, train=True):
    model.train(train)
    tot_cls, tot_pos, tot_n = 0.0, 0.0, 0
    seg_correct, seg_total = 0, 0
    with torch.set_grad_enabled(train):
        for pts, seg_true, chain_pos_true in loader:
            pts, seg_true, chain_pos_true = pts.to(device), seg_true.to(device), chain_pos_true.to(device)
            seg_logits, chain_pos_pred = model(pts)
            cls_loss, pos_loss = correspondence_loss(seg_logits, chain_pos_pred, seg_true, chain_pos_true, class_weight)
            loss = cls_loss + pos_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = pts.shape[0]
            tot_cls += cls_loss.item() * b
            tot_pos += pos_loss.item() * b
            tot_n += b
            seg_correct += (seg_logits.argmax(-1) == seg_true).sum().item()
            seg_total += seg_true.numel()
    return dict(cls=tot_cls / tot_n, pos=tot_pos / tot_n, seg_acc=seg_correct / seg_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_B2_correspondence_net_20260825")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(os.path.join(REPO, args.out_dir), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dd = load_model_dict(config.SMAL_FILE)
    jnames = list(dd["J_names"])
    class_names, name_to_id = build_segment_taxonomy(jnames)
    seg_class_id, chain_pos = build_vertex_labels(dd, class_names, name_to_id)
    class_weight = compute_class_weights(seg_class_id, len(class_names)).to(device)

    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].long()  # (F,3), fixed topology, CPU (dataset workers are CPU)

    corpus_path = os.path.join(REPO, args.train_corpus)
    train_ds = CorrespondenceDataset(corpus_path, faces, seg_class_id, chain_pos, args.n_points, "train", args.val_frac)
    val_ds = CorrespondenceDataset(corpus_path, faces, seg_class_id, chain_pos, args.n_points, "val", args.val_frac)
    persist = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               drop_last=True, persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             persistent_workers=persist)

    model = SMILCorrespondenceNet(n_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = []
    best_val = float("inf")
    out_dir = os.path.join(REPO, args.out_dir)
    t0 = time.time()
    for epoch in range(args.epochs):
        tr = run_epoch(model, train_loader, optimizer, device, class_weight, train=True)
        va = run_epoch(model, val_loader, optimizer, device, class_weight, train=False)
        scheduler.step(va["cls"] + va["pos"])
        elapsed = time.time() - t0
        row = dict(epoch=epoch, elapsed_sec=elapsed,
                   train_cls=tr["cls"], train_pos=tr["pos"], train_seg_acc=tr["seg_acc"],
                   val_cls=va["cls"], val_pos=va["pos"], val_seg_acc=va["seg_acc"])
        history.append(row)
        print(f"[epoch {epoch:3d}] {elapsed:7.0f}s  train cls={tr['cls']:.4f} pos={tr['pos']:.4f} seg_acc={tr['seg_acc']:.4f}  "
              f"| val cls={va['cls']:.4f} pos={va['pos']:.4f} seg_acc={va['seg_acc']:.4f}")

        val_total = va["cls"] + va["pos"]
        if val_total < best_val:
            best_val = val_total
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "n_points": args.n_points,
                "epoch": epoch,
                "val_total": val_total,
            }, os.path.join(out_dir, "best_model.pt"))

        with open(os.path.join(out_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"[B2] done. best_val={best_val:.4f}, checkpoint -> {out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
