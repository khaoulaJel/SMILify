"""Turn C3's (CSE head) predicted correspondences into a per-template-vertex target for the fitter.

WHY THIS SHAPE OF OUTPUT
------------------------
`Dense_GT_oracle` -- the only validated correspondence win in this investigation (seg_acc
0.832->0.874, 12/12 specimens) -- works by assigning each target point its TRUE corresponding
vertex and pulling the FITTED mesh's vertex at that exact index onto it. It never uses the biased
centroid/IK conversion that the D1 arms used, and which was independently shown to discard ~41% of
the achievable gain (GTLabel_IK_oracle 0.8522 vs Dense_GT_oracle 0.8745 on identical specimens).

This script produces the same KIND of signal from C3's PREDICTIONS instead of ground truth:
for every template vertex v, where does the network think v is on this target scan?

  pseudo_verts[n, v] = mean position of the sampled target points that retrieved vertex v
  mask[n, v]         = whether v was retrieved confidently enough to be used at all

The fitter term is then a masked squared error between the fitted mesh's vertex v and
pseudo_verts[v] -- direct vertex identity, no conversion step.

PRE-REGISTERED CONFIDENCE RULE (fixed before looking at any fit result)
----------------------------------------------------------------------
`eval_C3_vs_B2_retrieval_20260827.py` measured, on held-out specimens, that C3's direct
template-vertex retrieval is strong on `tr` (0.146 of segment length) and `fe` (0.258), weak on
`co` (0.324), AT CHANCE on `ti` (0.340 vs 0.339 random), and WORSE THAN CHANCE on `ta` (0.881 vs
0.363). Feeding all of it indiscriminately would inject known-bad correspondences.

Rather than hand-pick segments after the fact (which would be cherry-picking), points are filtered
by a DATA-DRIVEN confidence signal available at inference without any ground truth: the cosine
similarity between the query embedding and its retrieved vertex key. `--keep_frac` keeps the most
confident fraction of points GLOBALLY. The per-segment survival rate is reported so it is visible
whether the confidence signal is in fact suppressing the segments known to be unreliable -- that is
a genuine check of the confidence signal, not an assumption that it works.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.ops import sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402

from smil_correspondence_net import (  # noqa: E402
    build_segment_taxonomy, build_vertex_labels, load_model_dict,
)
from smil_cse_net import SMILCSENet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", default="diagnostics/moonshot/synth_clean")
    ap.add_argument("--c3_ckpt", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/best_model.pt")
    ap.add_argument("--out", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/cse_correspondence.npz")
    ap.add_argument("--n_points", type=int, default=2048, help="match C3 training density")
    ap.add_argument("--keep_frac", type=float, default=1.0,
                    help="keep this most-confident fraction of retrieved points. DEFAULT IS NOW 1.0 "
                         "(no filtering): the pre-registered check of the cosine-similarity "
                         "confidence signal FAILED on 2026-08-27 -- it survived ta at 95.7%% and ti "
                         "at 94.5%% (the two segments measured at/below chance) while suppressing "
                         "tr to 55.6%% and co to 18.2%% (the reliable ones). Confidence is "
                         "ANTI-correlated with correctness here, so filtering by it makes the "
                         "correspondence set worse. Kept as a flag for reproducing that finding.")
    ap.add_argument("--segments", default="tr,fe",
                    help="comma-separated segment short names to emit correspondences for, or "
                         "'all'. Default tr,fe -- the only two whose HELD-OUT direct-vertex "
                         "retrieval beat random by a margin in eval_C3_vs_B2_retrieval_20260827 "
                         "(tr 0.146 vs 0.377; fe 0.258 vs 0.345). co is weak (0.324 vs 0.490), ti "
                         "is at chance (0.340 vs 0.339) and ta is WORSE than chance (0.881 vs "
                         "0.363). This restriction is decided from independent held-out specimens "
                         "BEFORE any fit result is seen -- it is an evidence rule, not post-hoc "
                         "cherry-picking, and the excluded segments are named explicitly.")
    ap.add_argument("--n_repeats", type=int, default=8,
                    help="independent point samples per mesh, aggregated. Raises vertex coverage "
                         "WITHOUT changing the input density the network sees -- deliberately not "
                         "done by raising --n_points, because feeding this network a point "
                         "distribution unlike its training distribution is a recorded failure mode.")
    ap.add_argument("--filter", default="cosine", choices=["cosine", "cycle"],
                    help="which confidence signal --keep_frac ranks by. 'cycle' is the one that "
                         "actually works: forward-backward consistency cut mean retrieval error "
                         "0.726 -> 0.449 at top-50%%, while 'cosine' RAISED it to 1.044 "
                         "(confidence_signal_comparison_20260827.py, 4991 held-out points). "
                         "'cosine' is retained only to reproduce that negative result.")
    ap.add_argument("--normalise", action="store_true",
                    help="apply fitter_3d/utils.py:load_meshes' normalisation (centre by mean, "
                         "divide by max abs coord) before inference. REQUIRED for real scans: "
                         "bench50 meshes are ~1800-3300x the model frame and PointNet++'s "
                         "ball-query radii are ABSOLUTE, so without this sa1 raises "
                         "'index N out of bounds'. Default OFF only to keep synth_clean results "
                         "comparable with C4/C5, where raw .obj already sits within ~3-5%% of the "
                         "training frame (maxabs 0.97 vs 0.94-1.02). That residual few-percent "
                         "offset is a known small inaccuracy to clean up in a follow-up.")
    ap.add_argument("--shuffle_segments", default="",
                    help="comma-separated segments whose retrieved vertex is REPLACED by a random "
                         "vertex from the SAME segment. This is the control for the three-way test "
                         "of what ti/ta actually contribute: adding them HELPED (+0.0203) even "
                         "though their per-point retrieval is worse than random, so either "
                         "(a) correspondence identity still matters a little, (b) only the presence "
                         "of a soft pull toward the right region matters and identity is "
                         "irrelevant, or (c) these segments are not the source of the gain at all. "
                         "Comparing real vs shuffled-within-segment vs NO-constraint (the tr/fe/co "
                         "arm) separates the three: shuffled ~= real means identity does not "
                         "matter; shuffled ~= no-constraint means the soft-prior story is also "
                         "wrong and the gain comes from elsewhere in the optimiser.")
    ap.add_argument("--min_points_per_vertex", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    seg_class_id, _ = build_vertex_labels(dd, class_names, name_to_id)
    seg_short = {c: (None if n == "body" else n.split("_")[-1]) for c, n in enumerate(class_names)}

    ck = torch.load(os.path.join(REPO, args.c3_ckpt), map_location=device)
    model = SMILCSENet(n_vertices=ck["n_vertices"], embed_dim=ck["embed_dim"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    keys = model.vertex_embeddings()
    V = ck["n_vertices"]
    print(f"[setup] C3 epoch={ck['epoch']} V={V} D={ck['embed_dim']} device={device}", flush=True)

    mesh_dir = os.path.join(REPO, args.mesh_dir)
    files = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))
    if not files:
        raise SystemExit(f"no .obj files in {mesh_dir}")

    # evidence-based segment restriction (see --segments help for the held-out justification)
    if args.segments.strip().lower() == "all":
        allowed_verts, allowed_names = None, "all"
    else:
        want = {s.strip() for s in args.segments.split(",") if s.strip()}
        ids = [c for c, s in seg_short.items() if s in want]
        allowed_verts = np.where(np.isin(seg_class_id, ids))[0]
        allowed_names = ",".join(sorted(want))
        print(f"[setup] restricting correspondences to segments [{allowed_names}] "
              f"= {len(allowed_verts)} template vertices", flush=True)

    shuffle_ids = None
    if args.shuffle_segments.strip():
        want_sh = {x.strip() for x in args.shuffle_segments.split(",") if x.strip()}
        shuffle_ids = [c for c, s_ in seg_short.items() if s_ in want_sh]
        print(f"[setup] SHUFFLE CONTROL active for segments {sorted(want_sh)}: retrieved vertices "
              f"replaced by random same-segment vertices ({len(shuffle_ids)} classes)", flush=True)

    names, all_verts, all_mask = [], [], []
    seg_kept, seg_tot = {}, {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        vt, fc, _ = load_obj(f, load_textures=False)
        if args.normalise:
            centre = vt.mean(0)
            vt = vt - centre
            vt = vt / vt.abs().max(0)[0].max()
        mesh = Meshes(verts=[vt], faces=[fc.verts_idx])

        ret_l, pts_l, conf_l = [], [], []
        for _ in range(args.n_repeats):
            pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]
            with torch.no_grad():
                q = model(pts.unsqueeze(0).to(device))[0]      # (N,D) unit-norm
                sim = q @ keys.t()                             # (N,V) cosine
                c, r = sim.max(1)
                if args.filter == "cycle":
                    # CYCLE CONSISTENCY (the signal that actually works -- see
                    # confidence_signal_comparison_20260827.py): match point -> vertex, then ask
                    # which point that vertex retrieves back, and score by how far that lands from
                    # where we started. Measured to cut mean retrieval error 0.726 -> 0.449 at the
                    # top 50%, where COSINE made it WORSE (0.726 -> 1.044). Cosine is entangled with
                    # ambiguity (LightGlue arXiv:2306.13643); cycle consistency is a structural
                    # self-consistency test and is not.
                    back = sim.argmax(0)[r]                    # (N,) point each match returns to
                    c = -torch.norm(pts.to(device) - pts.to(device)[back], dim=1)
            ret_l.append(r.cpu()); pts_l.append(pts); conf_l.append(c.cpu())
        ret = torch.cat(ret_l); pts = torch.cat(pts_l); conf = torch.cat(conf_l)

        k = max(1, int(len(conf) * args.keep_frac))
        thr = torch.topk(conf, k).values.min()
        sel = conf >= thr
        if allowed_verts is not None:
            sel &= torch.as_tensor(np.isin(ret.numpy(), allowed_verts))
        ret_s = ret[sel].numpy()
        pts_s = pts[sel].numpy()

        if shuffle_ids is not None:
            # replace the retrieved vertex with a RANDOM vertex of the same segment, destroying
            # correspondence identity while preserving segment membership and point coverage
            rs = np.random.default_rng(args.seed + len(names))
            for cid in shuffle_ids:
                pool = np.where(seg_class_id == cid)[0]
                m = seg_class_id[ret_s] == cid
                if m.any() and len(pool):
                    ret_s = ret_s.copy()
                    ret_s[m] = rs.choice(pool, size=int(m.sum()))

        # per-segment survival, to check the confidence signal is doing something sensible
        r_all = ret.numpy()
        for sh in set(v for v in seg_short.values() if v):
            ids = np.array([c for c, s in seg_short.items() if s == sh])
            m_all = np.isin(r_all, np.where(np.isin(seg_class_id, ids))[0])
            m_sel = np.isin(ret_s, np.where(np.isin(seg_class_id, ids))[0])
            seg_tot[sh] = seg_tot.get(sh, 0) + int(m_all.sum())
            seg_kept[sh] = seg_kept.get(sh, 0) + int(m_sel.sum())

        # aggregate retrieved points into one target position per template vertex
        acc = np.zeros((V, 3), dtype=np.float64)
        cnt = np.zeros(V, dtype=np.int64)
        np.add.at(acc, ret_s, pts_s)
        np.add.at(cnt, ret_s, 1)
        valid = cnt >= args.min_points_per_vertex
        pseudo = np.zeros((V, 3), dtype=np.float32)
        pseudo[valid] = (acc[valid] / cnt[valid, None]).astype(np.float32)

        names.append(stem)
        all_verts.append(pseudo)
        all_mask.append(valid)
        print(f"  {stem}: {int(valid.sum())}/{V} vertices covered "
              f"({100*valid.mean():.1f}%), kept {int(sel.sum())}/{len(conf)} points", flush=True)

    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, names=np.array(names), verts=np.stack(all_verts),
                        mask=np.stack(all_mask), keep_frac=args.keep_frac,
                        n_points=args.n_points, c3_ckpt=args.c3_ckpt)

    print("\nper-segment survival of the confidence filter (kept / retrieved):")
    for sh in sorted(seg_tot):
        t = seg_tot[sh]
        print(f"  {sh:>3}: {seg_kept[sh]:6d}/{t:6d}  = {100*seg_kept[sh]/max(t,1):5.1f}%")
    print("\n(If the filter is working, the segments measured unreliable -- ta, ti -- should survive")
    print(" at a LOWER rate than tr/fe. If they do not, the confidence signal is not tracking")
    print(" correctness and that must be reported, not worked around.)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
