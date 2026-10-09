"""Does the CSE head produce sane correspondence on REAL bench50 scans, or does it collapse?

WHY THIS RUNS BEFORE ANY MEAN-TEACHER PIPELINE
----------------------------------------------
Mean-teacher self-training assumes the teacher's pseudo-labels are worth learning from. We have
ZERO evidence this correspondence mechanism produces sane output on real, damaged,
differently-distributed bench50 scans -- every result so far is synthetic, single corpus, shared
topology. And this project has ALREADY been bitten once by exactly this class of bug: feeding the
network raw vertices instead of a matched-distribution point sample broke it outright (the C2
point-density collapse). Discovering an equivalent collapse three days into a training run would
waste the scarcest resource here: bench50 has 50 specimens, no ground truth, and never will.

There is no ground truth on bench50, so correctness cannot be measured. What CAN be measured is
whether the output is DEGENERATE -- and every check below is a collapse detector with a
synthetic reference value to compare against, not a quality score.

COLLAPSE SIGNATURES CHECKED (each vs the same statistic on synth_clean)
----------------------------------------------------------------------
  1. vertex_coverage    -- fraction of template vertices retrieved at all. Collapse looks like a
                           tiny number: everything mapping onto a handful of vertices.
  2. top1_share         -- fraction of all points retrieving the single most-popular vertex.
                           Collapse looks like a large number.
  3. retrieval_entropy  -- normalised entropy of the retrieval distribution. Collapse -> ~0.
  4. segment_share      -- fraction of retrievals landing on each anatomical segment. A real scan
                           should look broadly like the synthetic distribution; wild divergence
                           (e.g. everything landing on `body`) is a distribution-shift collapse.
  5. cycle_consistency  -- mean forward-backward distance, the signal C5 actually filters on. If
                           this is far worse on real scans than synthetic, the filter itself is
                           not transferring and mean-teacher would be built on sand.
  6. embedding_norm_spread -- std of pre-normalisation embedding magnitude, a cheap detector of
                           the network being driven outside its trained regime.

PASS = real-scan statistics are within the stated tolerance of the synthetic reference.
FAIL on any of 1-3 or 5 means STOP: fix distribution handling before building a self-training loop.
This script deliberately reports a verdict, not a score.
"""
import argparse
import glob
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
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.ops import sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402

from smil_correspondence_net import (  # noqa: E402
    build_segment_taxonomy, build_vertex_labels, load_model_dict,
)
from smil_cse_net import SMILCSENet  # noqa: E402


def normalise_like_fitter(verts):
    """Reproduce fitter_3d/utils.py:load_meshes' normalisation EXACTLY (lines 340-344):
    centre by the mean vertex, then divide by the max absolute coordinate.

    This is not cosmetic. bench50 scans are ~1800-3300x larger than the model frame
    (diag 4664-8577 vs ~2.5) and off-centre, while PointNet++'s ball-query radii
    (0.1/0.2/0.4) are ABSOLUTE in model units. Without this, ball query finds zero
    neighbours, `group_idx` falls back to N, and sa1 raises
    `IndexError: index 2048 is out of bounds` -- which is exactly how this was found.

    Matching the FITTER's normalisation specifically (rather than any reasonable
    normalisation) matters because the correspondence targets this network produces are
    consumed in the fitter's own target frame; any other convention would introduce a
    systematic offset between predicted vertex positions and `fitted`.
    """
    centre = verts.mean(0)
    v = verts - centre
    return v / v.abs().max(0)[0].max()


def stats_for_mesh(model, keys, pts, seg_class_id, seg_short, device):
    with torch.no_grad():
        q_raw = model.embed_head(model.backbone(pts.unsqueeze(0).to(device))).transpose(2, 1)[0]
        norms = torch.norm(q_raw, dim=-1)
        q = torch.nn.functional.normalize(q_raw, dim=-1)
        sim = q @ keys.t()
        conf, ret = sim.max(1)
        back = sim.argmax(0)[ret]
        cyc = torch.norm(pts.to(device) - pts.to(device)[back], dim=1)

    r = ret.cpu().numpy()
    V = keys.shape[0]
    counts = np.bincount(r, minlength=V).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(V))

    seg_of = seg_class_id[r]
    shares = {}
    for cid, sh in seg_short.items():
        key = sh if sh else "body"
        shares[key] = shares.get(key, 0.0) + float((seg_of == cid).mean())

    # scale-normalise cycle distance by the scan's own extent so meshes of different size compare
    extent = float(np.linalg.norm(pts.numpy().max(0) - pts.numpy().min(0)))
    return dict(vertex_coverage=float((counts > 0).mean()),
                top1_share=float(counts.max() / counts.sum()),
                retrieval_entropy=ent,
                cycle_mean=float(cyc.mean().cpu()) / max(extent, 1e-9),
                embed_norm_std=float(norms.std().cpu()),
                cosine_mean=float(conf.mean().cpu()),
                segment_share=shares)


def summarise(rows):
    out = {}
    for k in ("vertex_coverage", "top1_share", "retrieval_entropy", "cycle_mean",
              "embed_norm_std", "cosine_mean"):
        v = np.array([r[k] for r in rows])
        out[k] = dict(mean=float(v.mean()), std=float(v.std()))
    segs = {}
    for r in rows:
        for s, val in r["segment_share"].items():
            segs.setdefault(s, []).append(val)
    out["segment_share"] = {s: float(np.mean(v)) for s, v in segs.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3_ckpt", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/best_model.pt")
    ap.add_argument("--real_dir", default="diagnostics/moonshot/bench50_clean")
    ap.add_argument("--synth_dir", default="diagnostics/moonshot/synth_clean")
    ap.add_argument("--n_real", type=int, default=8)
    ap.add_argument("--n_synth", type=int, default=8)
    ap.add_argument("--n_points", type=int, default=2048, help="MUST match C3 training density")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_bench50_sanity_20260827")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    seg_class_id, _ = build_vertex_labels(dd, class_names, name_to_id)
    seg_short = {c: (None if n == "body" else n.split("_")[-1]) for c, n in enumerate(class_names)}

    ck = torch.load(os.path.join(REPO, args.c3_ckpt), map_location=device)
    model = SMILCSENet(n_vertices=ck["n_vertices"], embed_dim=ck["embed_dim"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    keys = model.vertex_embeddings()
    print(f"[setup] C3 epoch={ck['epoch']} | n_points={args.n_points} (training density)", flush=True)

    res = {}
    for tag, d, n in (("synth", args.synth_dir, args.n_synth), ("real", args.real_dir, args.n_real)):
        files = sorted(glob.glob(os.path.join(REPO, d, "*.obj")))[:n]
        rows = []
        for f in files:
            vt, fc, _ = load_obj(f, load_textures=False)
            vt = normalise_like_fitter(vt)
            mesh = Meshes(verts=[vt], faces=[fc.verts_idx])
            pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]
            rows.append(stats_for_mesh(model, keys, pts, seg_class_id, seg_short, device))
            print(f"  [{tag}] {os.path.basename(f)[:44]:44s} "
                  f"cov={rows[-1]['vertex_coverage']:.3f} ent={rows[-1]['retrieval_entropy']:.3f} "
                  f"cyc={rows[-1]['cycle_mean']:.4f}", flush=True)
        res[tag] = summarise(rows)

    S, R = res["synth"], res["real"]
    print("\n=== COLLAPSE CHECK: real bench50 vs synthetic reference ===")
    print(f"{'statistic':>20} | {'synth':>9} {'real':>9} | {'ratio':>7} | verdict")
    checks = [
        ("vertex_coverage", "higher_better", 0.5),
        ("top1_share", "lower_better", 3.0),
        ("retrieval_entropy", "higher_better", 0.7),
        ("cycle_mean", "lower_better", 2.0),
    ]
    verdict = {}
    for name, direction, tol in checks:
        s, r = S[name]["mean"], R[name]["mean"]
        ratio = r / s if s else float("inf")
        ok = (ratio >= tol) if direction == "higher_better" else (ratio <= tol)
        verdict[name] = bool(ok)
        print(f"{name:>20} | {s:>9.4f} {r:>9.4f} | {ratio:>7.2f}x | "
              f"{'OK' if ok else 'COLLAPSE'}  (need {'>=' if direction=='higher_better' else '<='}{tol}x)")
    for name in ("embed_norm_std", "cosine_mean"):
        s, r = S[name]["mean"], R[name]["mean"]
        print(f"{name:>20} | {s:>9.4f} {r:>9.4f} | {r/s if s else float('nan'):>7.2f}x | (context only)")

    print("\n=== segment share of retrievals (distribution shift check) ===")
    allsegs = sorted(set(S["segment_share"]) | set(R["segment_share"]))
    print(f"{'seg':>6} | {'synth':>7} {'real':>7} | {'shift':>7}")
    for s_ in allsegs:
        a, b = S["segment_share"].get(s_, 0.0), R["segment_share"].get(s_, 0.0)
        print(f"{s_:>6} | {a:>7.3f} {b:>7.3f} | {b-a:>+7.3f}")

    ok_all = all(verdict.values())
    print(f"\nVERDICT: {'PASS -- no collapse detected; mean-teacher is on solid ground' if ok_all else 'FAIL -- distribution handling must be fixed BEFORE any self-training loop'}")
    print("NOTE: bench50 has no ground truth, so this measures DEGENERACY, not correctness.")
    print("Passing means 'not obviously broken', never 'correspondence is accurate on real scans'.")

    with open(os.path.join(out_dir, "bench50_sanity.json"), "w") as fh:
        json.dump({"config": vars(args), "synth": S, "real": R,
                   "verdict": verdict, "pass": ok_all}, fh, indent=2)
    print(f"\nwrote {out_dir}/bench50_sanity.json")


if __name__ == "__main__":
    main()
