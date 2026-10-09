"""W2 -- does the LEARNED descriptor (C11 CSE head) beat plain proximity inside a correct part?

Bars fixed in PREREGISTRATION_W2_learned_descriptor.md BEFORE this ran.

W1 closed the handcrafted half: local-PCA eigen-features sit near chance, and anatomy-conditioned
normalised part coordinates are WORSE than part-local rigid proximity on every segment. It did not
close the learned half. A learned dense descriptor trained on exactly this corpus already exists
(C11), so this is inference only -- measure what is on disk before spending GPU on a new one.

Reuses W1's protocol wholesale (same specimens, pairs, parts, metric) by importing it, so the new
arm is directly comparable and nothing is silently re-implemented.

THE SILENT FAILURE THIS GUARDS AGAINST
--------------------------------------
The C11 file stores weights under `model_state_dict`. F1 looked that up as `model`, matched NOTHING
under strict=False (178 missing keys), and a randomly-initialised network produced a plausible
result. So the load here is strict=True with 0 missing / 0 unexpected asserted, PLUS a named tensor
compared element-wise against the value read straight from the raw file. A non-zero-weight check
does not substitute: random weights are also non-zero.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    build_segment_taxonomy,
    build_vertex_labels,
    load_model_dict,
)
from smil_cse_net import SMILCSENet  # noqa: E402

from w1_within_part_retrieval import (  # noqa: E402
    LEG_SEGMENTS, MIN_VERTS, BAR_SEGMENTS,
    part_frame, canonical_coords, rigid_into, local_geo_features, zscore,
    three_tests, nn_retrieve,
)

# W1's established baseline, per segment. Hard-coded here as the bar, per the pre-registration.
W1_XYZ_RIGID = {"co": 0.2813, "tr": 0.0790, "fe": 0.0993, "ti": 0.0910, "ta": 0.1706}
C11_PUBLISHED_MEDIAN_3D = 0.02687


# ---------------------------------------------------------------------------------------------
def load_c11_strict(ckpt_path, device):
    """Load C11 with every guard the record says this failure class needs.

    Returns (model, report). Raises on anything that would leave a partly/never-loaded network.
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in raw:
        raise RuntimeError(f"no 'model_state_dict' in {ckpt_path}; keys={list(raw)[:10]}")
    sd = raw["model_state_dict"]
    n_vertices, embed_dim = int(raw["n_vertices"]), int(raw["embed_dim"])

    model = SMILCSENet(n_vertices=n_vertices, embed_dim=embed_dim)
    incompatible = model.load_state_dict(sd, strict=True)   # raises if keys mismatch
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"VOID: missing={len(missing)} unexpected={len(unexpected)}")

    # --- named-tensor byte check against the RAW file, not against the object we just built ---
    probe_name = "embed_head.0.weight"
    if probe_name not in sd:
        probe_name = sorted(sd)[0]
    got = dict(model.named_parameters()).get(probe_name)
    if got is None:
        got = dict(model.named_buffers())[probe_name]
    ref = sd[probe_name]
    if not torch.allclose(got.detach().cpu(), ref.cpu(), atol=0, rtol=0):
        raise RuntimeError(f"VOID: {probe_name} does not match the raw file after load")

    model.to(device).eval()
    return model, {"ckpt": ckpt_path, "epoch": int(raw.get("epoch", -1)),
                   "n_vertices": n_vertices, "embed_dim": embed_dim,
                   "n_missing": 0, "n_unexpected": 0,
                   "probe_tensor": probe_name,
                   "probe_abs_sum": float(ref.abs().sum()),
                   "val_at_save": raw.get("val", None)}


@torch.no_grad()
def embed_points(model, pts_np, device, chunk_pts=20000):
    """(N,3) -> (N,D) unit-norm query embeddings."""
    x = torch.as_tensor(pts_np, dtype=torch.float32, device=device).unsqueeze(0)
    out = model(x)                       # (1,N,D), already L2-normalised
    return out[0].cpu().numpy()


@torch.no_grad()
def reproduce_c11_retrieval(model, verts, faces_t, v_template_t, device, n_points=2048, seed=0):
    """MECHANISM CHECK 2: reproduce C11's published held-out median rest-space 3D error.

    Uses C11's OWN training convention -- 2048 points sampled from the posed surface, labelled by
    nearest true vertex, then full-vocabulary query->key argmax. If this does not reproduce, the
    input convention is wrong and no W2 number may be read.
    """
    torch.manual_seed(seed)
    keys = model.vertex_embeddings()                        # (V,D)
    errs, top1, tot = [], 0, 0
    for vs in verts:
        vt = torch.as_tensor(vs, dtype=torch.float32, device=device).unsqueeze(0)
        mesh = Meshes(verts=vt, faces=faces_t.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, n_points)      # (1,N,3)
        true_vert = knn_points(pts, vt, K=1).idx[0, :, 0]    # (N,)
        q = model(pts)[0]                                    # (N,D)
        sim = q @ keys.t()
        pred = sim.argmax(dim=1)
        top1 += (pred == true_vert).sum().item()
        tot += true_vert.numel()
        errs.append(torch.norm(v_template_t[pred] - v_template_t[true_vert], dim=-1).cpu())
    e = torch.cat(errs)
    return {"median_3d_err": float(e.median()), "mean_3d_err": float(e.mean()),
            "top1": top1 / tot, "n": int(tot)}


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--ckpt", default="diagnostics/anatomical_pose_init/out_C11_hardneg_20260826/best_model.pt")
    ap.add_argument("--n_specimens", type=int, default=12)
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density_control_points", type=int, default=2048)
    ap.add_argument("--out_dir", default="diagnostics/within_part_correspondence/out_W2")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"device: {device}")

    # ---------------------------------------------------------------- model + check 1
    print("\n[MECHANISM CHECK 1] strict checkpoint load ...")
    model, ck = load_c11_strict(os.path.join(REPO, args.ckpt), device)
    print(f"  loaded {ck['ckpt']}")
    print(f"  epoch={ck['epoch']}  n_vertices={ck['n_vertices']}  embed_dim={ck['embed_dim']}")
    print(f"  missing=0 unexpected=0 (strict=True)")
    print(f"  named-tensor byte check on '{ck['probe_tensor']}': MATCHES raw file "
          f"(abs sum {ck['probe_abs_sum']:.6f})")
    print(f"  val recorded at save time: {ck['val_at_save']}")

    # ---------------------------------------------------------------- data
    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    labels, _ = build_vertex_labels(dd, class_names, name_to_id)
    faces_t = torch.as_tensor(np.asarray(dd["f"]).astype(np.int64), device=device)
    v_template_t = torch.as_tensor(np.asarray(dd["v_template"]), dtype=torch.float32, device=device)

    data = np.load(os.path.join(REPO, args.corpus))
    verts_all = data["verts"]
    n_total = verts_all.shape[0]
    spec_ids = np.arange(n_total - args.n_specimens, n_total)
    V = verts_all[spec_ids].astype(np.float64)
    print(f"\ncorpus: {n_total} specimens; using held-out tail {spec_ids[0]}..{spec_ids[-1]}")

    # ---------------------------------------------------------------- check 2
    print("\n[MECHANISM CHECK 2] reproduce C11's published retrieval ...")
    rep = reproduce_c11_retrieval(model, V, faces_t, v_template_t, device, seed=args.seed)
    dev = abs(rep["median_3d_err"] - C11_PUBLISHED_MEDIAN_3D) / C11_PUBLISHED_MEDIAN_3D
    print(f"  median rest-space 3D err: {rep['median_3d_err']:.5f}  "
          f"(published {C11_PUBLISHED_MEDIAN_3D}, deviation {dev:.1%})")
    print(f"  top-1: {rep['top1']:.4f}   n={rep['n']}")
    check2_ok = dev <= 0.20
    print(f"  -> {'PASS' if check2_ok else 'FAIL -- RUN IS VOID'} (bar: within 20%)")

    # ---------------------------------------------------------------- parts + frames
    body_id = name_to_id["body"]
    body_idx = np.where(labels == body_id)[0]
    body_centroids = V[:, body_idx, :].mean(axis=1)

    parts, excluded = {}, []
    for cls, cid in name_to_id.items():
        if cls == "body":
            continue
        idx = np.where(labels == cid)[0]
        (parts.__setitem__(cls, idx) if len(idx) >= MIN_VERTS
         else excluded.append((cls, int(len(idx)))))

    frames = {}
    for cls, idx in parts.items():
        leg, side, seg = cls.split("_")
        pi = LEG_SEGMENTS.index(seg)
        parent_idx = (body_idx if pi == 0
                      else np.where(labels == name_to_id[f"{leg}_{side}_{LEG_SEGMENTS[pi-1]}"])[0])
        for s in range(len(V)):
            frames[(cls, s)] = part_frame(V[s][idx], V[s][parent_idx].mean(0), body_centroids[s])

    # ---------------------------------------------------------------- embeddings
    print("\ncomputing embeddings ...")
    E_full = {}          # primary: full 10,235-vertex input
    for s in range(len(V)):
        E_full[s] = embed_points(model, V[s], device)
    print(f"  full-vertex embeddings: {E_full[0].shape}")

    # density control: C11's own 2048-point training density, transferred to vertices by nearest
    E_dens = {}
    with torch.no_grad():
        for s in range(len(V)):
            vt = torch.as_tensor(V[s], dtype=torch.float32, device=device).unsqueeze(0)
            mesh = Meshes(verts=vt, faces=faces_t.unsqueeze(0))
            pts = sample_points_from_meshes(mesh, args.density_control_points)
            q = model(pts)[0].cpu().numpy()
            p = pts[0].cpu().numpy()
            nn = cKDTree(p).query(V[s], k=1)[1]
            E_dens[s] = q[nn]
    print(f"  density-control embeddings ({args.density_control_points} pts): {E_dens[0].shape}")

    # chance control: INDEPENDENT permutation per specimen, on the density-matched (in-distribution)
    # embeddings. A shared permutation would leave vertex perm[i] on A still paired with perm[i] on
    # B and so would not destroy correspondence at all; and running it on the OOD full-density
    # embeddings would test a representation the run already treats as invalid.
    E_shuf = {s: E_dens[s][rng.permutation(V.shape[1])] for s in range(len(V))}

    # ---------------------------------------------------------------- local geo (W1 arm)
    all_part_idx = np.concatenate([parts[c] for c in sorted(parts)])
    lg_lookup = {}
    for s in range(len(V)):
        F = local_geo_features(V[s], all_part_idx)
        lg_lookup[s] = np.zeros((len(labels), 6))
        lg_lookup[s][all_part_idx] = F

    # ---------------------------------------------------------------- pairs + retrieval
    pairs = []
    while len(pairs) < args.n_pairs:
        a, b = rng.integers(0, len(V), 2)
        if a != b:
            pairs.append((int(a), int(b)))

    ARMS = ["RANDOM", "XYZ_RIGID", "LOCAL_GEO", "PART_FRAME",
            "CSE_C11", "CSE_C11_DENSITY", "CSE_DENSITY_SHUFFLED"]
    rows = []
    for (a, b) in pairs:
        for cls, idx in sorted(parts.items()):
            fa, fb = frames[(cls, a)], frames[(cls, b)]
            if fa is None or fb is None:
                continue
            VA, VB = V[a][idx], V[b][idx]
            seglen, n = fb["L"], len(idx)

            ret = {
                "RANDOM": rng.integers(0, n, n),
                "XYZ_RIGID": nn_retrieve(rigid_into(VA, fa, fb), VB),
                "LOCAL_GEO": nn_retrieve(zscore(lg_lookup[a][idx]), zscore(lg_lookup[b][idx])),
                "PART_FRAME": nn_retrieve(canonical_coords(VA, fa), canonical_coords(VB, fb)),
                # unit-norm embeddings -> Euclidean NN == cosine NN, the metric the loss optimises
                "CSE_C11": nn_retrieve(E_full[a][idx], E_full[b][idx]),
                "CSE_C11_DENSITY": nn_retrieve(E_dens[a][idx], E_dens[b][idx]),
                "CSE_DENSITY_SHUFFLED": nn_retrieve(E_shuf[a][idx], E_shuf[b][idx]),
            }
            for arm, j in ret.items():
                e = np.linalg.norm(VB[j] - VB[np.arange(n)], axis=1) / seglen
                rows.append({"pair": f"{a}->{b}", "part": cls, "seg": cls.split("_")[2],
                             "arm": arm, "median_err": float(np.median(e)), "n": n})

    def per_seg(arm, seg):
        return np.array([r["median_err"] for r in rows if r["arm"] == arm and r["seg"] == seg])

    segs = sorted({r["seg"] for r in rows}, key=LEG_SEGMENTS.index)

    print("\n" + "=" * 104)
    print("median normalised 3D within-part retrieval error (lower better); ceiling = 0")
    print("=" * 104)
    print(f"{'seg':<5}" + "".join(f"{a:>17}" for a in ARMS))
    summary = {}
    for seg in segs:
        summary[seg] = {a: float(np.median(per_seg(a, seg))) for a in ARMS}
        print(f"{seg:<5}" + "".join(f"{summary[seg][a]:>17.4f}" for a in ARMS))

    print("\n" + "=" * 104)
    print("PRE-REGISTERED BAR: CSE_C11 vs XYZ_RIGID (W1 baseline), >=15% relative + sign p<0.05")
    print("=" * 104)
    print(f"{'seg':<5}{'W1 bar':>10}{'XYZ_RIGID':>12}{'CSE_C11':>11}{'rel.red.':>11}"
          f"{'better':>12}{'sign p':>12}{'verdict':>9}")
    bar_pass, bar_detail = {}, {}
    for seg in segs:
        x, c = per_seg("XYZ_RIGID", seg), per_seg("CSE_C11", seg)
        t = three_tests(c, x)
        mx, mc = float(np.median(x)), float(np.median(c))
        rel = (mx - mc) / mx if mx > 0 else float("nan")
        ok = (rel >= 0.15) and (t["sign_p"] < 0.05) and (mc < mx)
        if seg in BAR_SEGMENTS:
            bar_pass[seg] = bool(ok)
        bar_detail[seg] = {"xyz": mx, "cse": mc, "rel": rel, **t, "passes": bool(ok)}
        mark = ("PASS" if ok else "no") + ("" if seg in BAR_SEGMENTS else " (n/b)")
        print(f"{seg:<5}{W1_XYZ_RIGID.get(seg, float('nan')):>10.4f}{mx:>12.4f}{mc:>11.4f}"
              f"{rel:>10.1%}{str(t.get('n_better',0))+'/'+str(t['n']):>12}"
              f"{t['sign_p']:>12.2e}{mark:>9}")

    print("\n" + "=" * 104)
    print("SAME BAR APPLIED TO THE DENSITY-MATCHED ARM (the one the pre-registration says counts)")
    print("=" * 104)
    print(f"{'seg':<5}{'XYZ_RIGID':>12}{'CSE_density':>13}{'rel.red.':>11}{'better':>12}{'sign p':>12}")
    dens_detail, dens_pass = {}, {}
    for seg in segs:
        x, c = per_seg("XYZ_RIGID", seg), per_seg("CSE_C11_DENSITY", seg)
        t = three_tests(c, x)
        mx, mc = float(np.median(x)), float(np.median(c))
        rel = (mx - mc) / mx if mx > 0 else float("nan")
        ok = (rel >= 0.15) and (t["sign_p"] < 0.05) and (mc < mx)
        dens_detail[seg] = {"xyz": mx, "cse_density": mc, "rel": rel, **t, "passes": bool(ok)}
        if seg in BAR_SEGMENTS:
            dens_pass[seg] = bool(ok)
        print(f"{seg:<5}{mx:>12.4f}{mc:>13.4f}{rel:>10.1%}"
              f"{str(t.get('n_better',0))+'/'+str(t['n']):>12}{t['sign_p']:>12.2e}")

    verdict = ("PASS" if all(bar_pass.get(s, False) for s in BAR_SEGMENTS)
               else "FAIL" if not any(bar_pass.values()) else "PARTIAL")
    verdict_density = ("PASS" if all(dens_pass.get(s, False) for s in BAR_SEGMENTS)
                       else "FAIL" if not any(dens_pass.values()) else "PARTIAL")
    if not check2_ok:
        verdict = "VOID (C11 retrieval did not reproduce)"

    # ---------------------------------------------------------------- checks 3 + 4
    print("\n" + "=" * 104)
    print("[MECHANISM CHECK 3] density control -- verdict must not flip at C11's training density")
    print("=" * 104)
    print(f"{'seg':<5}{'CSE_C11':>12}{'CSE_density':>14}{'XYZ_RIGID':>12}{'density beats XYZ?':>20}")
    flip = False
    for seg in segs:
        c, d, x = (float(np.median(per_seg(a, seg))) for a in
                   ("CSE_C11", "CSE_C11_DENSITY", "XYZ_RIGID"))
        beats = d < x
        if seg in BAR_SEGMENTS and beats != bar_pass.get(seg, False):
            flip = True
        print(f"{seg:<5}{c:>12.4f}{d:>14.4f}{x:>12.4f}{str(beats):>20}")
    print(f"  verdict flips under density control: {flip}")

    print("\n" + "=" * 104)
    print("[MECHANISM CHECK 4] chance control -- shuffled embeddings must land at RANDOM")
    print("=" * 104)
    print(f"{'seg':<5}{'SHUFFLED':>12}{'RANDOM':>12}{'ratio':>10}")
    for seg in segs:
        sh, rd = summary[seg]["CSE_DENSITY_SHUFFLED"], summary[seg]["RANDOM"]
        print(f"{seg:<5}{sh:>12.4f}{rd:>12.4f}{sh/rd:>10.3f}")

    print("\n" + "=" * 104)
    print(f"ENDPOINT VERDICT (pre-registered, full-vertex arm): {verdict}")
    print(f"ENDPOINT VERDICT (density-matched arm -- THE ONE THAT COUNTS): {verdict_density}")
    print("=" * 104)
    print(f"  bar segments {BAR_SEGMENTS}: " + ", ".join(f"{s}={bar_pass.get(s)}" for s in BAR_SEGMENTS))
    print(f"  [check 1] strict load, 0 missing / 0 unexpected, byte-verified : PASS")
    print(f"  [check 2] C11 retrieval reproduces ({dev:.1%} dev, bar 20%)     : "
          f"{'PASS' if check2_ok else 'FAIL'}")
    print(f"  [check 3] verdict flips under density control                  : {flip}")

    payload = {"config": vars(args), "checkpoint": ck, "c11_reproduction": rep,
               "spec_ids": spec_ids.tolist(), "pairs": pairs, "excluded_parts": excluded,
               "summary": summary, "bar": bar_detail, "verdict": verdict, "verdict_density": verdict_density, "density_bar": dens_detail,
               "density_flip": bool(flip), "check2_ok": bool(check2_ok)}
    with open(os.path.join(out_dir, "w2_results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    import csv
    with open(os.path.join(out_dir, "w2_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_dir}/w2_results.json and w2_rows.csv")


if __name__ == "__main__":
    main()
