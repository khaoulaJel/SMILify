"""W3 -- a descriptor trained FOR within-part cross-specimen correspondence.

Bar fixed in PREREGISTRATION_W3_contextual_embedding.md BEFORE this ran.

WHY THIS IS NOT W2 AGAIN
------------------------
C11 is already a contextual learned embedding (PointNet++ MSG, InfoNCE, exact vertex supervision).
What it is NOT is trained for the task W2 measured. C11's objective is
`query -> argmax over a fixed per-template-vertex key table`, so its negatives spread over all
10,235 vertices and are overwhelmingly OTHER PARTS (femur or gaster?). The task that matters is
`query on A -> match among candidates on B, restricted to the same part`, whose negatives are the
~100-300 vertices of that same femur. C11 was optimised on easy negatives and judged on hard ones.

W3 changes that one thing: TRAIN WITH THE NEGATIVES THE EVALUATION USES.

TWO DESIGN DECISIONS, EACH FORCED BY A MEASURED RESULT
------------------------------------------------------
1. PART-CONDITIONED HEAD. The protocol grants the oracle part label on both sides, so a descriptor
   that must rediscover "this is a femur" from geometry is solving a problem it was handed the
   answer to. The backbone stays xyz-only (preserving an exact C11 warm start, since sa1 has
   in_channel=0), and a learned part embedding is concatenated to the per-point backbone feature
   BEFORE the projection head. The network therefore spends its capacity on *which point within
   this femur*, which is the only unsolved half.
2. NO CANONICALISATION. Input is raw posed vertex coordinates; part length and radius are NOT
   normalised away. W1 measured that removing them is actively harmful -- inter-specimen
   differences in segment proportion are correspondence-bearing signal, not nuisance.

THE DENSITY TRAP (W2's registered control caught this; not repeated here)
------------------------------------------------------------------------
Feeding all 10,235 vertices to this backbone is ~5x its training density and drives it to
at-or-below chance -- out-of-distribution, not a descriptor result. W3 fixes ONE density
everywhere: `n_points` vertices sampled BY INDEX, identically for both specimens of a pair, at
train and at eval. The per-part candidate set is then smaller than W1's, so the task is easier and
W1's baselines are NOT the fair comparison. Every arm is recomputed on the same subsets and the bar
is set against the subset-matched XYZ_RIGID.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    build_segment_taxonomy, build_vertex_labels, load_model_dict,
)
from smil_cse_net import SMILCSENet  # noqa: E402

from w1_within_part_retrieval import (  # noqa: E402
    LEG_SEGMENTS, MIN_VERTS, BAR_SEGMENTS,
    part_frame, canonical_coords, rigid_into, three_tests, nn_retrieve,
)

TRAIN_LO, TRAIN_HI = 0, 3700        # inside the registered 0-3799 training range
VAL_LO, VAL_HI = 3700, 3800         # model selection only; never the reported eval
EVAL_LO, EVAL_HI = 3988, 4000       # the 12 specimens W1 and W2 used
BACKBONE_PROBE = "sa1.conv_blocks.0.0.weight"


# =============================================================================================
# Model
# =============================================================================================
class W3Net(nn.Module):
    """C11's PointNet++ backbone (xyz-only, warm-startable) + a PART-CONDITIONED projection head.

    The part embedding is concatenated to the per-point backbone feature rather than to the input,
    which is what lets sa1 keep `in_channel=0` and accept C11's weights verbatim. Output is
    L2-normalised, so cosine NN == Euclidean NN and the retrieval performed at inference is exactly
    the quantity the InfoNCE loss optimises.
    """

    def __init__(self, n_parts, embed_dim=16, part_dim=16, temperature=0.07):
        super().__init__()
        self.embed_dim = embed_dim
        base = SMILCSENet(n_vertices=1, embed_dim=embed_dim)
        # reuse the exact backbone modules (identical shapes to C11's)
        self.sa1, self.sa2, self.sa3 = base.sa1, base.sa2, base.sa3
        self.fp3, self.fp2, self.fp1 = base.fp3, base.fp2, base.fp1
        self.part_embed = nn.Embedding(n_parts, part_dim)
        nn.init.normal_(self.part_embed.weight, std=0.02)
        self.head = nn.Sequential(
            nn.Conv1d(128 + part_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, embed_dim, 1),
        )
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def logit_scale(self):
        return self.log_temp.clamp(max=torch.log(torch.tensor(100.0,
                                                              device=self.log_temp.device))).exp()

    def backbone_features(self, x):
        """(B,N,3) -> (B,128,N). Mirrors SMILCSENet.backbone exactly."""
        xyz = x.transpose(2, 1)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        return self.fp1(xyz, l1_xyz, None, l1_points)

    def forward(self, x, part_ids):
        """x: (B,N,3); part_ids: (N,) long -> (B,N,D) unit-norm."""
        f = self.backbone_features(x)                                  # (B,128,N)
        p = self.part_embed(part_ids).t().unsqueeze(0)                 # (1,P,N)
        p = p.expand(f.shape[0], -1, -1)
        e = self.head(torch.cat([f, p], dim=1))                        # (B,D,N)
        return F.normalize(e.transpose(2, 1), dim=-1)


def warm_start_backbone(model, ckpt_path):
    """Copy sa*/fp* from C11. Returns (n_copied, changed) so the caller can assert it happened."""
    raw = torch.load(ckpt_path, map_location="cpu")
    src = raw["model_state_dict"]
    own = model.state_dict()
    before = own[BACKBONE_PROBE].clone()
    copied = 0
    for k, v in src.items():
        if (k.startswith("sa") or k.startswith("fp")) and k in own and own[k].shape == v.shape:
            own[k] = v.clone()
            copied += 1
    model.load_state_dict(own)
    changed = not torch.allclose(before, model.state_dict()[BACKBONE_PROBE])
    return copied, changed


def load_w3_strict(path, n_parts, device):
    """MECHANISM CHECK 2: strict reload + named-tensor byte check against the raw file."""
    raw = torch.load(path, map_location="cpu")
    sd = raw["model_state_dict"]
    m = W3Net(n_parts, int(raw["embed_dim"]), int(raw["part_dim"]))
    inc = m.load_state_dict(sd, strict=True)
    if list(getattr(inc, "missing_keys", [])) or list(getattr(inc, "unexpected_keys", [])):
        raise RuntimeError("VOID: non-empty missing/unexpected on strict load")
    probe = "head.0.weight"
    if not torch.allclose(dict(m.named_parameters())[probe].detach(), sd[probe], atol=0, rtol=0):
        raise RuntimeError(f"VOID: {probe} does not match the raw file after load")
    m.to(device).eval()
    return m, {"probe": probe, "probe_abs_sum": float(sd[probe].abs().sum()),
               "epoch": int(raw.get("epoch", -1)), "val": raw.get("val")}


# =============================================================================================
# Loss
# =============================================================================================
def within_part_infonce(qa, qb, part_rows, logit_scale):
    """InfoNCE whose candidate set for each query is its OWN PART on the other specimen.

    Positive for row i is row i of qb; negatives are the other rows of the SAME part. That is the
    evaluation task used directly as the loss -- the single change W3 makes against C11.
    """
    losses, accs = [], []
    for rows in part_rows:
        if len(rows) < 4:
            continue
        logits = (qa[rows] @ qb[rows].t()) * logit_scale
        tgt = torch.arange(len(rows), device=qa.device)
        losses.append(F.cross_entropy(logits, tgt))
        accs.append((logits.argmax(1) == tgt).float().mean())
    if not losses:
        return None, None
    return torch.stack(losses).mean(), torch.stack(accs).mean()


# =============================================================================================
# Sampling + evaluation geometry
# =============================================================================================
def sample_subset(rng, part_index_lists, n_points, n_verts):
    """Sample vertex indices; report which sampled rows belong to which part, and a per-row part id.

    Returns (sel, rows_per_part, row_part_id). `row_part_id` is what conditions the head; rows that
    fall on no tracked part get id 0 (a dedicated 'other' class), so every sampled point has a
    valid, honest label rather than a silently wrong one.
    """
    sel = np.sort(rng.choice(n_verts, size=n_points, replace=False))
    pos = {v: i for i, v in enumerate(sel)}
    rows_per_part, row_part_id = [], np.zeros(n_points, dtype=np.int64)
    for pi, idx in enumerate(part_index_lists):
        rows = np.array([pos[v] for v in idx if v in pos], dtype=np.int64)
        rows_per_part.append(rows)
        row_part_id[rows] = pi + 1          # 0 reserved for 'other'
    return sel, rows_per_part, row_part_id


def subset_frame(P, proximal_ref, body_centroid):
    """W1's anatomy-oriented frame, built from the SAMPLED points of one part.

    Uses W1's construction verbatim (axis sign from the parent segment, theta from the body
    centroid) so XYZ_RIGID and PART_FRAME mean here exactly what they meant in W1 -- only the point
    density differs.
    """
    return part_frame(P, proximal_ref, body_centroid)


@torch.no_grad()
def evaluate(model, V, parts, part_names, parent_of, body_idx, device, rng,
             n_points, n_subsets, n_pairs, c11_model=None):
    """W1's protocol, every arm recomputed on the SAME density-matched subsets."""
    model.eval()
    n_verts = V.shape[1]
    part_idx_lists = [parts[c] for c in part_names]
    body_centroids = V[:, body_idx, :].mean(axis=1)
    rows_out = []

    for sub in range(n_subsets):
        sel, rows_per_part, row_pid = sample_subset(rng, part_idx_lists, n_points, n_verts)
        pid_t = torch.as_tensor(row_pid, device=device)
        emb, emb_c11 = {}, {}
        for s in range(len(V)):
            x = torch.as_tensor(V[s][sel], dtype=torch.float32, device=device).unsqueeze(0)
            emb[s] = model(x, pid_t)[0].cpu().numpy()
            if c11_model is not None:
                emb_c11[s] = c11_model(x)[0].cpu().numpy()
        # independent per-specimen permutation -> a genuine chance control
        shuf = {s: emb[s][rng.permutation(n_points)] for s in range(len(V))}

        pairs = []
        while len(pairs) < n_pairs:
            a, b = rng.integers(0, len(V), 2)
            if a != b:
                pairs.append((int(a), int(b)))

        for (a, b) in pairs:
            for pname, rws in zip(part_names, rows_per_part):
                if len(rws) < 8:
                    continue
                vsel = sel[rws]
                VA, VB = V[a][vsel], V[b][vsel]
                par = parent_of[pname]
                fa = subset_frame(VA, V[a][par].mean(0), body_centroids[a])
                fb = subset_frame(VB, V[b][par].mean(0), body_centroids[b])
                if fa is None or fb is None:
                    continue
                seglen, n = fb["L"], len(rws)
                ret = {
                    "RANDOM": rng.integers(0, n, n),
                    "XYZ_RIGID": nn_retrieve(rigid_into(VA, fa, fb), VB),
                    "PART_FRAME": nn_retrieve(canonical_coords(VA, fa), canonical_coords(VB, fb)),
                    "W3": nn_retrieve(emb[a][rws], emb[b][rws]),
                    "W3_SHUFFLED": nn_retrieve(shuf[a][rws], shuf[b][rws]),
                }
                if c11_model is not None:
                    ret["CSE_C11"] = nn_retrieve(emb_c11[a][rws], emb_c11[b][rws])
                for arm, j in ret.items():
                    e = np.linalg.norm(VB[j] - VB[np.arange(n)], axis=1) / seglen
                    rows_out.append({"subset": sub, "pair": f"{a}->{b}", "part": pname,
                                     "seg": pname.split("_")[2], "arm": arm,
                                     "median_err": float(np.median(e)), "n": int(n)})
    return rows_out


def bootstrap_ci(x, w, n_boot=2000, seed=0):
    """Bootstrap CI on the relative reduction, resampling part-pairs. Guards against a verdict
    that rests on a handful of parts."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rels = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        mx, mw = np.median(x[k]), np.median(w[k])
        if mx > 0:
            rels.append((mx - mw) / mx)
    return (float(np.percentile(rels, 2.5)), float(np.percentile(rels, 97.5))) if rels else (np.nan,) * 2


# =============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--c11", default="diagnostics/anatomical_pose_init/out_C11_hardneg_20260826/best_model.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps_per_epoch", type=int, default=250)
    ap.add_argument("--batch_pairs", type=int, default=4)
    ap.add_argument("--n_points", type=int, default=4096)
    ap.add_argument("--embed_dim", type=int, default=16)
    ap.add_argument("--part_dim", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_subsets", type=int, default=8)
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="diagnostics/within_part_correspondence/out_W3")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"device: {device}", flush=True)

    # ---------------------------------------------------------------- taxonomy
    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    labels, _ = build_vertex_labels(dd, class_names, name_to_id)
    body_idx = np.where(labels == name_to_id["body"])[0]
    parts = {}
    for cls, cid in name_to_id.items():
        if cls == "body":
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) >= MIN_VERTS:
            parts[cls] = idx
    part_names = sorted(parts)
    part_idx_lists = [parts[c] for c in part_names]
    parent_of = {}
    for cls in part_names:
        leg, side, seg = cls.split("_")
        pi = LEG_SEGMENTS.index(seg)
        parent_of[cls] = (body_idx if pi == 0
                          else np.where(labels == name_to_id[f"{leg}_{side}_{LEG_SEGMENTS[pi-1]}"])[0])
    print(f"parts used: {len(parts)} (+1 'other' class for unsampled regions)")

    data = np.load(os.path.join(REPO, args.corpus))
    verts_all = data["verts"]
    n_verts = verts_all.shape[1]

    # ---------------------------------------------------------------- check 1: splits
    train_ids, val_ids = set(range(TRAIN_LO, TRAIN_HI)), set(range(VAL_LO, VAL_HI))
    eval_ids = set(range(EVAL_LO, EVAL_HI))
    assert not (train_ids & eval_ids) and not (val_ids & eval_ids) and not (train_ids & val_ids)
    print(f"[CHECK 1] splits disjoint: train {TRAIN_LO}-{TRAIN_HI-1}, val {VAL_LO}-{VAL_HI-1}, "
          f"eval {EVAL_LO}-{EVAL_HI-1}   PASS")
    V_train = verts_all[TRAIN_LO:TRAIN_HI].astype(np.float32)
    V_eval = verts_all[EVAL_LO:EVAL_HI].astype(np.float64)

    # ---------------------------------------------------------------- check 3: warm start
    model = W3Net(len(parts) + 1, args.embed_dim, args.part_dim).to(device)
    n_copied, changed = warm_start_backbone(model, os.path.join(REPO, args.c11))
    ok3 = n_copied > 0 and changed
    print(f"[CHECK 3] warm start from C11: {n_copied} backbone tensors copied, "
          f"'{BACKBONE_PROBE}' changed={changed}   {'PASS' if ok3 else 'VOID'}")
    if not ok3:
        raise RuntimeError("VOID: warm start did not transfer")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---------------------------------------------------------------- train
    history, best, best_path = [], -1.0, os.path.join(out_dir, "w3_best.pt")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl, ta, nb = 0.0, 0.0, 0
        for _ in range(args.steps_per_epoch):
            sel, rows_per_part, row_pid = sample_subset(rng, part_idx_lists, args.n_points, n_verts)
            pid_t = torch.as_tensor(row_pid, device=device)
            rows_t = [torch.as_tensor(r, device=device) for r in rows_per_part if len(r) >= 4]
            xa, xb = [], []
            for _ in range(args.batch_pairs):
                a, b = rng.integers(0, len(V_train), 2)
                while a == b:
                    b = rng.integers(0, len(V_train))
                xa.append(V_train[a][sel])
                xb.append(V_train[b][sel])
            QA = model(torch.as_tensor(np.stack(xa), dtype=torch.float32, device=device), pid_t)
            QB = model(torch.as_tensor(np.stack(xb), dtype=torch.float32, device=device), pid_t)
            ls = model.logit_scale()
            ls_, as_ = [], []
            for i in range(args.batch_pairs):
                l, a_ = within_part_infonce(QA[i], QB[i], rows_t, ls)
                if l is not None:
                    ls_.append(l)
                    as_.append(a_)
            if not ls_:
                continue
            loss = torch.stack(ls_).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss)
            ta += float(torch.stack(as_).mean())
            nb += 1
        sched.step()
        el, ea = tl / max(nb, 1), ta / max(nb, 1)
        history.append({"epoch": ep, "loss": el, "inpart_acc": ea})
        print(f"  epoch {ep:3d}  loss {el:.4f}  in-part top1 {ea:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)
        if ea > best:
            best = ea
            torch.save({"model_state_dict": model.state_dict(), "embed_dim": args.embed_dim,
                        "part_dim": args.part_dim, "epoch": ep, "val": ea}, best_path)

    conv = {"loss_first": history[0]["loss"], "loss_last": history[-1]["loss"],
            "acc_first": history[0]["inpart_acc"], "acc_last": history[-1]["inpart_acc"]}
    print(f"\n[CHECK 6] convergence: loss {conv['loss_first']:.4f} -> {conv['loss_last']:.4f}; "
          f"in-part top1 {conv['acc_first']:.4f} -> {conv['acc_last']:.4f}")

    # ---------------------------------------------------------------- check 2 + eval
    print("\n[CHECK 2] strict reload of the trained checkpoint ...")
    w3, rep = load_w3_strict(best_path, len(parts) + 1, device)
    print(f"  0 missing / 0 unexpected; '{rep['probe']}' byte-matches raw file "
          f"(abs sum {rep['probe_abs_sum']:.6f}); best epoch {rep['epoch']}   PASS")

    c11 = SMILCSENet(n_vertices=int(torch.load(os.path.join(REPO, args.c11),
                                               map_location="cpu")["n_vertices"]),
                     embed_dim=args.embed_dim)
    craw = torch.load(os.path.join(REPO, args.c11), map_location="cpu")["model_state_dict"]
    c11.load_state_dict(craw, strict=True)
    c11.to(device).eval()

    print("\nevaluating on HELD-OUT specimens ...", flush=True)
    rows = evaluate(w3, V_eval, parts, part_names, parent_of, body_idx, device,
                    np.random.default_rng(args.seed + 1), args.n_points, args.n_subsets,
                    args.n_pairs, c11_model=c11)
    print("evaluating on TRAINING specimens (check 5: overfitting) ...", flush=True)
    V_tr = verts_all[TRAIN_LO:TRAIN_LO + len(V_eval)].astype(np.float64)
    rows_tr = evaluate(w3, V_tr, parts, part_names, parent_of, body_idx, device,
                       np.random.default_rng(args.seed + 2), args.n_points,
                       max(2, args.n_subsets // 4), args.n_pairs)

    ARMS = ["RANDOM", "XYZ_RIGID", "PART_FRAME", "CSE_C11", "W3", "W3_SHUFFLED"]

    def med(rws, arm, seg):
        return np.array([r["median_err"] for r in rws if r["arm"] == arm and r["seg"] == seg])

    segs = sorted({r["seg"] for r in rows}, key=LEG_SEGMENTS.index)
    print("\n" + "=" * 104)
    print(f"median normalised 3D within-part retrieval error -- density-matched subsets "
          f"({args.n_points} verts x {args.n_subsets} subsets); ceiling = 0")
    print("=" * 104)
    print(f"{'seg':<5}" + "".join(f"{a:>16}" for a in ARMS))
    summary = {}
    for seg in segs:
        summary[seg] = {a: float(np.median(med(rows, a, seg))) for a in ARMS}
        print(f"{seg:<5}" + "".join(f"{summary[seg][a]:>16.4f}" for a in ARMS))

    print("\n" + "=" * 104)
    print("PRE-REGISTERED BAR: W3 vs SUBSET-MATCHED XYZ_RIGID, >=15% relative + sign p<0.05")
    print("=" * 104)
    print(f"{'seg':<5}{'XYZ_RIGID':>12}{'W3':>11}{'rel.red.':>11}{'95% CI':>20}"
          f"{'better':>12}{'sign p':>12}{'v':>7}")
    bar, bar_pass = {}, {}
    for seg in segs:
        x, w = med(rows, "XYZ_RIGID", seg), med(rows, "W3", seg)
        t = three_tests(w, x)
        mx, mw = float(np.median(x)), float(np.median(w))
        rel = (mx - mw) / mx if mx > 0 else float("nan")
        lo, hi = bootstrap_ci(x, w, seed=args.seed)
        ok = (rel >= 0.15) and (t["sign_p"] < 0.05) and (mw < mx)
        bar[seg] = {"xyz": mx, "w3": mw, "rel": rel, "ci": [lo, hi], **t, "passes": bool(ok)}
        if seg in BAR_SEGMENTS:
            bar_pass[seg] = bool(ok)
        print(f"{seg:<5}{mx:>12.4f}{mw:>11.4f}{rel:>10.1%}"
              f"{f'[{lo:+.1%}, {hi:+.1%}]':>20}"
              f"{str(t.get('n_better',0))+'/'+str(t['n']):>12}{t['sign_p']:>12.2e}"
              f"{('PASS' if ok else 'no'):>7}")

    verdict = ("PASS" if all(bar_pass.get(s, False) for s in BAR_SEGMENTS)
               else "FAIL" if not any(bar_pass.values()) else "PARTIAL")

    print("\n" + "=" * 104)
    print("SECONDARY (gates nothing): did the objective change help vs C11, on the same subsets?")
    print("=" * 104)
    print(f"{'seg':<5}{'CSE_C11':>12}{'W3':>11}{'rel.red.':>11}{'sign p':>12}")
    sec = {}
    for seg in segs:
        c, w = med(rows, "CSE_C11", seg), med(rows, "W3", seg)
        t = three_tests(w, c)
        mc, mw = float(np.median(c)), float(np.median(w))
        sec[seg] = {"c11": mc, "w3": mw, "rel": (mc - mw) / mc if mc > 0 else float("nan"),
                    "sign_p": t["sign_p"]}
        print(f"{seg:<5}{mc:>12.4f}{mw:>11.4f}{sec[seg]['rel']:>10.1%}{t['sign_p']:>12.2e}")

    print("\n" + "=" * 104)
    print("[CHECK 4] chance control -- independently shuffled W3 must land at RANDOM")
    print("=" * 104)
    chance_ok = True
    for seg in segs:
        sh, rd = summary[seg]["W3_SHUFFLED"], summary[seg]["RANDOM"]
        r = sh / rd
        chance_ok &= 0.85 <= r <= 1.15
        print(f"  {seg:<4} shuffled {sh:.4f}   random {rd:.4f}   ratio {r:.3f}")
    print(f"  -> {'PASS' if chance_ok else 'FAIL'}")

    print("\n" + "=" * 104)
    print("[CHECK 5] overfitting -- W3 on TRAINING specimens vs held-out")
    print("=" * 104)
    over = {}
    for seg in segs:
        tr = float(np.median(med(rows_tr, "W3", seg)))
        ho = summary[seg]["W3"]
        over[seg] = {"train": tr, "heldout": ho, "gap": ho - tr}
        print(f"  {seg:<4} train {tr:.4f}   held-out {ho:.4f}   gap {ho-tr:+.4f}")

    print("\n" + "=" * 104)
    print(f"ENDPOINT VERDICT (pre-registered): {verdict}")
    print("=" * 104)
    print("  bar segments: " + ", ".join(f"{s}={bar_pass.get(s)}" for s in BAR_SEGMENTS))

    payload = {"config": vars(args), "verdict": verdict, "summary": summary, "bar": bar,
               "secondary_vs_c11": sec, "history": history, "convergence": conv,
               "checkpoint": rep, "warm_start_tensors": int(n_copied),
               "splits": {"train": [TRAIN_LO, TRAIN_HI], "val": [VAL_LO, VAL_HI],
                          "eval": [EVAL_LO, EVAL_HI]},
               "chance_control_ok": bool(chance_ok), "overfit": over}
    with open(os.path.join(out_dir, "w3_results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(os.path.join(out_dir, "w3_rows.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {out_dir}/w3_results.json and w3_rows.csv")


if __name__ == "__main__":
    main()
