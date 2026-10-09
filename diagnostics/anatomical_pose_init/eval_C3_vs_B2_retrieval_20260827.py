"""C3 (CSE head) vs B2 (frozen backbone) -- identical cross-specimen retrieval metric.

Runs the SAME measurement as `cse_feasibility_retrieval_20260826.py` so the two are directly
comparable: same held-out specimens, same seed, same segment frames, same measured RANDOM floor and
achievable ceiling, same gap_closed definition, same sign+Wilcoxon+paired-t reporting.

The only thing swapped is the per-point representation:
  * B2   : frozen backbone `fp1`/`l0_points` 128-d features (what the feasibility probe measured)
  * C3   : the trained CSE head's 16-d L2-normalised embedding

It additionally reports what only C3 can do: DIRECT query -> template-vertex retrieval, i.e. the
actual correspondence a CSE head hands the fitter, scored in canonical rest space against the
point's true vertex and normalised by the segment's own length.

Reference points for that direct number, both MEASURED not assumed (standing rule -- never assume a
ceiling is perfect):
  * RANDOM  : a uniformly drawn template vertex from the same segment (chance).
  * SAMPLING: the true vertex itself, i.e. 0 by construction -- so the meaningful ceiling for the
    direct metric is 0 and the raw normalised error is already interpretable on its own.
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

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    SMILCorrespondenceNet, build_segment_taxonomy, build_vertex_labels, load_model_dict,
)
from smil_cse_net import SMILCSENet  # noqa: E402

# reuse the feasibility probe's own helpers so the metric cannot drift between the two scripts
from cse_feasibility_retrieval_20260826 import (  # noqa: E402
    decompose, gap_closed, per_point_features, segment_frame, three_tests,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2_ckpt", default="diagnostics/anatomical_pose_init/out_B2_correspondence_net_20260826/best_model.pt")
    ap.add_argument("--c3_ckpt", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/best_model.pt")
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--n_specimens", type=int, default=12)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_C3_eval_20260827")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    seg_class_id, chain_pos = build_vertex_labels(dd, class_names, name_to_id)
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    seg_short = {c: (None if n == "body" else n.split("_")[-1]) for c, n in enumerate(class_names)}

    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].long()

    b2 = SMILCorrespondenceNet(n_classes=len(class_names)).to(device)
    ck = torch.load(os.path.join(REPO, args.b2_ckpt), map_location=device)
    b2.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    b2.eval()

    ck3 = torch.load(os.path.join(REPO, args.c3_ckpt), map_location=device)
    c3 = SMILCSENet(n_vertices=ck3["n_vertices"], embed_dim=ck3["embed_dim"]).to(device)
    c3.load_state_dict(ck3["model_state_dict"])
    c3.eval()
    print(f"[setup] C3 ckpt epoch={ck3['epoch']} val={ck3['val']}", flush=True)
    keys = c3.vertex_embeddings()                       # (V,D) unit-norm

    canon = {}
    for cid in range(1, len(class_names)):
        ax, L = segment_frame(v_template[seg_class_id == cid])
        if ax is not None:
            canon[cid] = (ax, L)

    gt = np.load(os.path.join(REPO, args.corpus), allow_pickle=True)["verts"]
    n_val = max(1, int(len(gt) * args.val_frac))
    val = gt[-n_val:][:args.n_specimens]

    store, direct = [], []
    for si, vv in enumerate(val):
        v = torch.as_tensor(vv, dtype=torch.float32)
        mesh = Meshes(verts=v.unsqueeze(0), faces=faces.unsqueeze(0))
        pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]
        nn_idx = knn_points(pts.unsqueeze(0), v.unsqueeze(0), K=1).idx[0, :, 0].numpy()
        with torch.no_grad():
            p = pts.unsqueeze(0).to(device)
            f_b2 = per_point_features(b2, p)[0].T.cpu().numpy()
            e_c3 = c3(p)[0]
            retrieved = (e_c3 @ keys.t()).argmax(1).cpu().numpy()   # direct template-vertex match
            f_c3 = e_c3.cpu().numpy()
        store.append({"b2": f_b2, "c3": f_c3, "tv": nn_idx, "seg": seg_class_id[nn_idx]})

        # ---- direct query -> template-vertex retrieval (only C3 can do this) ----
        for cid in np.unique(seg_class_id[nn_idx]):
            if cid == 0 or cid not in canon:
                continue
            m = np.where(seg_class_id[nn_idx] == cid)[0]
            if len(m) < 8:
                continue
            axis, L = canon[cid]
            tv, rv = nn_idx[m], retrieved[m]
            tot, ax_, ci = decompose(v_template[tv], v_template[rv], axis, L)
            pool = np.where(seg_class_id == cid)[0]
            rnd = rng.choice(pool, size=len(m))
            rtot, _, _ = decompose(v_template[tv], v_template[rnd], axis, L)
            direct.append(dict(specimen=si, seg=seg_short[cid], class_id=int(cid), n=len(m),
                               err=float(tot.mean()), axial=float(ax_.mean()), circ=float(ci.mean()),
                               rand=float(rtot.mean()),
                               same_seg=float((seg_class_id[rv] == cid).mean())))
        print(f"[{si+1}/{len(val)}] done", flush=True)

    # ---------------- cross-specimen, both representations, identical metric ----------------
    recs = []
    for a in range(len(store)):
        b = (a + 1) % len(store)
        if b == a:
            break
        A, B = store[a], store[b]
        for cid in np.unique(A["seg"]):
            if cid == 0 or cid not in canon:
                continue
            ia, ib = np.where(A["seg"] == cid)[0], np.where(B["seg"] == cid)[0]
            if len(ia) < 8 or len(ib) < 8:
                continue
            axis, L = canon[cid]
            va, vb = A["tv"][ia], B["tv"][ib]
            Dt = ((v_template[va][:, None, :] - v_template[vb][None, :, :]) ** 2).sum(-1)
            partners = {"tnn": Dt.argmin(1), "rand": rng.integers(0, len(ib), size=len(ia))}
            for rep in ("b2", "c3"):
                Dr = ((A[rep][ia][:, None, :] - B[rep][ib][None, :, :]) ** 2).sum(-1)
                partners[rep] = Dr.argmin(1)
            for tag, j in partners.items():
                tot, ax_, ci = decompose(v_template[va], v_template[vb][j], axis, L)
                recs.append(dict(pair=f"{a}->{b}", seg=seg_short[cid], class_id=int(cid),
                                 partner=tag, total=float(tot.mean()),
                                 axial=float(ax_.mean()), circ=float(ci.mean())))

    def pick(seg, tag, field="total"):
        rs = sorted([r for r in recs if r["seg"] == seg and r["partner"] == tag],
                    key=lambda r: (r["pair"], r["class_id"]))
        return np.array([r[field] for r in rs])

    print("\n=== CROSS-SPECIMEN retrieval: C3 (CSE head) vs B2 (frozen backbone), identical metric ===")
    print(f"{'seg':>4} {'n':>3} | {'B2 gap':>7} {'C3 gap':>7} | {'B2 err':>7} {'C3 err':>7} "
          f"{'rand':>7} {'ceil':>7} | {'C3 ax':>6} {'C3 ci':>6} | {'sign_p':>8} {'wilcox':>8} {'t_p':>8}")
    summary = {}
    for seg in sorted({r["seg"] for r in recs}):
        rd, tn = pick(seg, "rand"), pick(seg, "tnn")
        fb, fc = pick(seg, "b2"), pick(seg, "c3")
        if not (len(rd) == len(tn) == len(fb) == len(fc)) or len(rd) == 0:
            continue
        gb, gc = gap_closed(fb.mean(), rd.mean(), tn.mean()), gap_closed(fc.mean(), rd.mean(), tn.mean())
        t = three_tests(fc, fb)     # C3 vs B2, paired on the same (pair, class)
        axg = gap_closed(pick(seg, "c3", "axial").mean(), pick(seg, "rand", "axial").mean(), pick(seg, "tnn", "axial").mean())
        cig = gap_closed(pick(seg, "c3", "circ").mean(), pick(seg, "rand", "circ").mean(), pick(seg, "tnn", "circ").mean())
        summary[seg] = dict(n=len(rd), b2_gap=gb, c3_gap=gc, b2_err=float(fb.mean()),
                            c3_err=float(fc.mean()), rand=float(rd.mean()), ceil=float(tn.mean()),
                            c3_axial_gap=axg, c3_circ_gap=cig, c3_vs_b2=t)
        print(f"{seg:>4} {len(rd):>3} | {gb:>7.3f} {gc:>7.3f} | {fb.mean():>7.3f} {fc.mean():>7.3f} "
              f"{rd.mean():>7.3f} {tn.mean():>7.3f} | {axg:>6.3f} {cig:>6.3f} | "
              f"{t['sign_p']:>8.2e} {t['wilcoxon_p']:>8.2e} {t['ttest_p']:>8.2e}")
    print("  (sign/wilcox/t compare C3 vs B2 directly; C3 better == negative mean delta)")

    print("\n=== DIRECT query -> template-vertex retrieval (what C3 hands the fitter) ===")
    print(f"{'seg':>4} {'n':>3} | {'err':>7} {'rand':>7} | {'axial':>7} {'circ':>7} | {'same-seg':>8}")
    dsum = {}
    for seg in sorted({d["seg"] for d in direct}):
        ds = [d for d in direct if d["seg"] == seg]
        e = np.array([d["err"] for d in ds]); r = np.array([d["rand"] for d in ds])
        dsum[seg] = dict(n=len(ds), err=float(e.mean()), rand=float(r.mean()),
                         axial=float(np.mean([d["axial"] for d in ds])),
                         circ=float(np.mean([d["circ"] for d in ds])),
                         same_seg=float(np.mean([d["same_seg"] for d in ds])),
                         vs_random=three_tests(e, r))
        d0 = dsum[seg]
        print(f"{seg:>4} {d0['n']:>3} | {d0['err']:>7.3f} {d0['rand']:>7.3f} | "
              f"{d0['axial']:>7.3f} {d0['circ']:>7.3f} | {d0['same_seg']:>8.3f}")
    print("  err/rand/axial/circ as a FRACTION of the segment's own length; same-seg = fraction of")
    print("  retrieved vertices landing in the correct segment (a coarse correctness check).")

    with open(os.path.join(out_dir, "c3_vs_b2_retrieval.json"), "w") as fh:
        json.dump({"config": vars(args), "cross_summary": summary, "direct_summary": dsum,
                   "cross_records": recs, "direct_records": direct}, fh, indent=2)
    print(f"\nwrote {out_dir}/c3_vs_b2_retrieval.json")


if __name__ == "__main__":
    main()
