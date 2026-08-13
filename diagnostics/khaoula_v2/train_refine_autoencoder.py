"""Train the Uzolas-style geometry-refined DINO descriptor.

Pre-registered gate and named risk are in diagnostics/V2_CHANGELOG.md and
diagnostics/khaoula_v2/REPORT_V2.md section 2.3, written BEFORE this script was run --
read those before reading these results, not after.

WHAT THIS DOES
  1. Geodesic distance on the template's REST-POSE mesh (pose-invariant under the
     isometric-bending assumption, same assumption HKS elsewhere in this investigation
     already relies on), via Dijkstra shortest-path on the mesh edge graph
     (scipy.sparse.csgraph.dijkstra, edge weight = rest-pose Euclidean edge length --
     a standard discrete geodesic approximation). From a sampled anchor set, not full
     V x V. Cached to disk.
  2. DINO features for the template + TRAINING specimens (synth_004..synth_011 --
     disjoint from synth_000..synth_003, which T1.1 already scored raw DINO/HKS on and
     which stays held out for eval), reusing t10_diff3f_sanity.py's render/extract/
     backproject pipeline. Cached to disk (T1.0/T1.1 did NOT cache this -- confirmed by
     direct check before writing this script -- so every prior run re-paid the
     extraction cost; this one doesn't have to be re-paid again after this).
  3. Train RefineNet (refine_net.py): reconstruction MSE + a margin-based contrastive
     term on geodesic distance -- near pairs (normalised geodesic dist < NEAR_THRESH)
     pulled together, far pairs (> FAR_THRESH) pushed apart up to a margin. Pairs are
     (anchor vertex, any other vertex) using the precomputed anchor-to-all distance
     rows, restricted to vertices actually SEEN (not occluded) in the specimen the
     features came from.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, "..", "moonshot"))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("HF_HOME", "/p/scratch/cias-7/jellal1/hf_cache")

from refine_net import RefineNet  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = os.path.join(HERE, "out", "dino_cache")
os.makedirs(CACHE, exist_ok=True)


def load_template():
    import config

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    v = np.asarray(dd["v_template"], dtype=np.float64)
    f = np.asarray(dd["f"]).astype(np.int64)
    return v, f


def geodesic_anchors(v_rest, faces, n_anchors=500, seed=0, cache_path=None):
    """Dijkstra shortest-path from n_anchors sampled vertices to ALL vertices, on the
    mesh edge graph weighted by rest-pose Euclidean edge length. Returns
    (anchor_idx (n_anchors,), dist (n_anchors, V)).
    """
    if cache_path and os.path.isfile(cache_path):
        d = np.load(cache_path)
        print(f"  geodesic cache hit: {cache_path}")
        return d["anchor_idx"], d["dist"]

    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    V = len(v_rest)
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    w = np.linalg.norm(v_rest[e0] - v_rest[e1], axis=1)
    graph = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([e0, e1]), np.concatenate([e1, e0]))),
                           shape=(V, V)).tocsr()

    rng = np.random.default_rng(seed)
    anchor_idx = rng.choice(V, size=min(n_anchors, V), replace=False)
    print(f"  running Dijkstra from {len(anchor_idx)} anchors on {V} vertices ...", flush=True)
    dist = dijkstra(graph, directed=False, indices=anchor_idx)  # (n_anchors, V)

    if cache_path:
        np.savez(cache_path, anchor_idx=anchor_idx, dist=dist)
        print(f"  wrote {cache_path}")
    return anchor_idx, dist


def get_dino_cached(name, v_np, f_np):
    """DINO descriptor for a named mesh (template or a synth specimen), cached to disk."""
    cache_path = os.path.join(CACHE, f"{name}.npz")
    if os.path.isfile(cache_path):
        d = np.load(cache_path)
        print(f"  DINO cache hit: {name}")
        return d["feat"], d["seen"]

    from t10_diff3f_sanity import render_views, dino_features, backproject

    v = torch.tensor(v_np, dtype=torch.float32, device=DEV)
    c = v.mean(0)
    v = (v - c) / (v - c).abs().max()
    f = torch.tensor(f_np, dtype=torch.int64, device=DEV)
    imgs, rasterizers = render_views(v, f, n_views=8, image_size=512)
    dmaps = dino_features(imgs)
    feat, seen = backproject(v, f, rasterizers, dmaps)
    np.savez(cache_path, feat=feat, seen=seen)
    print(f"  DINO extracted + cached: {name}")
    return feat, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_specimens", nargs="+",
                     default=[f"synth_{i:03d}" for i in range(4, 12)])
    ap.add_argument("--n_anchors", type=int, default=500)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--pairs_per_epoch", type=int, default=200000)
    ap.add_argument("--near_thresh", type=float, default=0.05)
    ap.add_argument("--far_thresh", type=float, default=0.20)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("[refine] loading template + geodesic anchors ...", flush=True)
    v_tpl, faces = load_template()
    anchor_idx, geo = geodesic_anchors(
        v_tpl, faces, n_anchors=args.n_anchors, seed=args.seed,
        cache_path=os.path.join(HERE, "out", "template_geodesic.npz"))
    diam = float(geo.max())
    geo_norm = geo / diam
    print(f"  mesh geodesic diameter (rest pose): {diam:.4f} model units; "
          f"anchor distance matrix {geo.shape}")

    print("[refine] extracting/caching DINO features: template ...", flush=True)
    feat_tpl, seen_tpl = get_dino_cached("template", v_tpl, faces)

    gt = np.load(os.path.join(MOON, "synth_clean", "ground_truth.npz"))
    gtv = gt["verts"]
    names = [str(x) for x in gt["names"]]
    name2idx = {n: i for i, n in enumerate(names)}

    train_feats, train_seens = [], []
    for nm in args.train_specimens:
        print(f"[refine] extracting/caching DINO features: {nm} ...", flush=True)
        tgt = gtv[name2idx[nm]].astype(np.float64)
        feat, seen = get_dino_cached(nm, tgt, faces)
        train_feats.append(feat)
        train_seens.append(seen)

    # anchors restricted to those SEEN in the template (only these have a valid query feature)
    anchor_seen_mask = seen_tpl[anchor_idx]
    valid_anchor_pos = np.nonzero(anchor_seen_mask)[0]  # positions into anchor_idx/geo rows
    print(f"[refine] {len(valid_anchor_pos)}/{len(anchor_idx)} anchors seen in template")

    model = RefineNet(in_dim=768, embed_dim=args.embed_dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[refine] training {args.epochs} epochs, {args.pairs_per_epoch} pairs/epoch, "
          f"{len(train_feats)} specimens ...", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_recon, total_contrast, n_batches = 0.0, 0.0, 0
        n_pairs_done = 0
        while n_pairs_done < args.pairs_per_epoch:
            bs = min(args.batch_size, args.pairs_per_epoch - n_pairs_done)
            spec_i = rng.integers(0, len(train_feats))
            feat_s, seen_s = train_feats[spec_i], train_seens[spec_i]

            a_pos = rng.choice(valid_anchor_pos, size=bs, replace=True)
            a_vidx = anchor_idx[a_pos]  # actual vertex indices

            # "other" vertex: any vertex seen in BOTH template and this training specimen
            joint_seen = np.nonzero(seen_tpl & seen_s)[0]
            b_vidx = rng.choice(joint_seen, size=bs, replace=True)

            d = geo_norm[a_pos, b_vidx]  # normalised geodesic distance, anchor -> other

            xa = torch.tensor(feat_tpl[a_vidx], dtype=torch.float32, device=DEV)
            xb = torch.tensor(feat_s[b_vidx], dtype=torch.float32, device=DEV)
            dgeo = torch.tensor(d, dtype=torch.float32, device=DEV)

            za, ra = model(xa)
            zb, rb = model(xb)

            recon = F.mse_loss(ra, xa) + F.mse_loss(rb, xb)

            ed = (za - zb).pow(2).sum(-1).clamp_min(1e-9).sqrt()
            near = dgeo < args.near_thresh
            far = dgeo > args.far_thresh
            contrast = torch.zeros((), device=DEV)
            if near.any():
                contrast = contrast + ed[near].pow(2).mean()
            if far.any():
                contrast = contrast + F.relu(args.margin - ed[far]).pow(2).mean()

            loss = recon + contrast
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_recon += float(recon.item())
            total_contrast += float(contrast.item())
            n_batches += 1
            n_pairs_done += bs

        print(f"  epoch {epoch+1:3d}/{args.epochs}  recon={total_recon/n_batches:.4f}  "
              f"contrast={total_contrast/n_batches:.4f}", flush=True)

    out_path = os.path.join(HERE, "out", "refine_net.pt")
    torch.save(dict(state_dict=model.state_dict(), embed_dim=args.embed_dim,
                     train_specimens=args.train_specimens, args=vars(args)), out_path)
    print(f"\n[refine] wrote {out_path}")


if __name__ == "__main__":
    main()
