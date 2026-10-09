"""WHY does cycle-consistency degrade on some real bench50 scans and not others?

The bench50 sanity gate found cycle_mean 2.62x worse on real scans than synthetic -- but the
degradation is NOT uniform: 4 of 8 sampled specimens sat at ~1.7x while 3 sat at 2.7-4.8x. Used as
a bare numeric cutoff that is a black-box triage rule nobody can account for later.

This script asks whether the split has a LEGIBLE CAUSE, over all 50 specimens rather than the 8
sampled before. The distinction matters for scoping a mean-teacher pass:
  * if degradation tracks a fixable DATA property (density, mesh damage, scale), the bad specimens
    can be reprocessed and recovered;
  * if it tracks MORPHOLOGY (genus/size/shape far from the training distribution), they are a
    population this method genuinely cannot serve yet, and the inclusion criterion should say so
    honestly rather than hide behind a threshold.

Cheap covariates only -- all computable without ground truth, which is the point, since any rule
derived here has to run on real scans that will never have ground truth:
  n_verts, n_faces        -- mesh resolution
  components              -- connected components; >1 suggests debris/detached parts
  boundary_edge_frac      -- fraction of edges used by exactly one face: holes/incompleteness
  degenerate_face_frac    -- zero-area faces
  aspect_ratio            -- longest/shortest bounding-box side, a crude morphology descriptor
  extent_diag_raw         -- original scale before normalisation
  edge_len_cv             -- coefficient of variation of edge length: meshing irregularity
  nn_dist_cv              -- CV of nearest-neighbour spacing among sampled points AFTER
                             normalisation: the direct "point density uniformity" measure, and the
                             closest cheap proxy for the distribution the network was trained on
  genus                   -- first token of the filename, for the morphology question

Reports Spearman correlation of each covariate against cycle_mean, plus the same split the gate
used (good vs bad specimens), so a cause is either visible or demonstrably absent.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

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
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402
from scipy import stats  # noqa: E402

from smil_correspondence_net import load_model_dict  # noqa: E402
from smil_cse_net import SMILCSENet  # noqa: E402
from bench50_sanity_check_20260827 import normalise_like_fitter  # noqa: E402

COVARIATES = ["n_verts", "n_faces", "components", "boundary_edge_frac", "degenerate_face_frac",
              "aspect_ratio", "extent_diag_raw", "edge_len_cv", "nn_dist_cv"]


def mesh_covariates(verts_raw, faces):
    v = verts_raw.numpy()
    f = faces.numpy()
    ext = v.max(0) - v.min(0)

    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = np.sort(e, 1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    boundary_frac = float((counts == 1).mean())

    el = np.linalg.norm(v[uniq[:, 0]] - v[uniq[:, 1]], axis=1)
    edge_cv = float(el.std() / el.mean()) if el.mean() > 0 else float("nan")

    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    degen = float((area <= 1e-12).mean())

    # connected components over the vertex graph (union-find, cheap enough at this size)
    parent = np.arange(len(v))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in uniq:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    comps = len({find(i) for i in range(len(v))})

    return dict(n_verts=len(v), n_faces=len(f), components=comps,
                boundary_edge_frac=boundary_frac, degenerate_face_frac=degen,
                aspect_ratio=float(ext.max() / max(ext.min(), 1e-9)),
                extent_diag_raw=float(np.linalg.norm(ext)), edge_len_cv=edge_cv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3_ckpt", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/best_model.pt")
    ap.add_argument("--real_dir", default="diagnostics/moonshot/bench50_clean")
    ap.add_argument("--synth_dir", default="diagnostics/moonshot/synth_clean")
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="diagnostics/anatomical_pose_init/out_bench50_sanity_20260827")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dd = load_model_dict(config.SMAL_FILE)
    ck = torch.load(os.path.join(REPO, args.c3_ckpt), map_location=device)
    model = SMILCSENet(n_vertices=ck["n_vertices"], embed_dim=ck["embed_dim"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    keys = model.vertex_embeddings()

    rows = []
    for tag, d in (("synth", args.synth_dir), ("real", args.real_dir)):
        for f in sorted(glob.glob(os.path.join(REPO, d, "*.obj"))):
            name = os.path.basename(f)[:-4]
            vt_raw, fc, _ = load_obj(f, load_textures=False)
            cov = mesh_covariates(vt_raw, fc.verts_idx)
            vt = normalise_like_fitter(vt_raw)
            mesh = Meshes(verts=[vt], faces=[fc.verts_idx])
            pts = sample_points_from_meshes(mesh, num_samples=args.n_points)[0]

            nnd = knn_points(pts.unsqueeze(0), pts.unsqueeze(0), K=2).dists[0, :, 1].sqrt()
            cov["nn_dist_cv"] = float(nnd.std() / nnd.mean())

            with torch.no_grad():
                q = model(pts.unsqueeze(0).to(device))[0]
                sim = q @ keys.t()
                ret = sim.argmax(1)
                back = sim.argmax(0)[ret]
                cyc = torch.norm(pts.to(device) - pts.to(device)[back], dim=1)
            extent = float(np.linalg.norm(pts.numpy().max(0) - pts.numpy().min(0)))
            cov.update(name=name, tag=tag, genus=name.split("_")[0],
                       cycle_mean=float(cyc.mean().cpu()) / max(extent, 1e-9))
            rows.append(cov)
        print(f"[{tag}] {sum(1 for r in rows if r['tag']==tag)} meshes done", flush=True)

    real = [r for r in rows if r["tag"] == "real"]
    syn = [r for r in rows if r["tag"] == "synth"]
    s_ref = float(np.mean([r["cycle_mean"] for r in syn]))
    cyc = np.array([r["cycle_mean"] for r in real])
    ratio = cyc / s_ref
    print(f"\nsynthetic reference cycle_mean = {s_ref:.5f}")
    print(f"real n={len(real)}  cycle ratio: median {np.median(ratio):.2f}x  "
          f"range {ratio.min():.2f}-{ratio.max():.2f}x  |  <=2x: {(ratio<=2).sum()}/{len(ratio)}")

    print("\n=== does any cheap covariate explain the degradation? (Spearman vs cycle_mean) ===")
    print(f"{'covariate':>22} | {'rho':>7} {'p':>10} | note")
    corr = {}
    for c in COVARIATES:
        x = np.array([r[c] for r in real], dtype=float)
        if np.allclose(x, x[0]):
            print(f"{c:>22} | {'--':>7} {'--':>10} | constant across specimens")
            continue
        rho, p = stats.spearmanr(x, cyc)
        corr[c] = dict(rho=float(rho), p=float(p))
        flag = "**SIGNIFICANT**" if p < 0.01 else ("marginal" if p < 0.05 else "")
        print(f"{c:>22} | {rho:>7.3f} {p:>10.2e} | {flag}")

    print("\n=== good (<=2x) vs bad (>2x) specimens, covariate means ===")
    good = [r for r in real if r["cycle_mean"] / s_ref <= 2]
    bad = [r for r in real if r["cycle_mean"] / s_ref > 2]
    print(f"  n_good={len(good)}  n_bad={len(bad)}")
    print(f"{'covariate':>22} | {'good':>12} {'bad':>12} | {'ratio':>7} {'MWU p':>9}")
    split = {}
    for c in COVARIATES:
        g = np.array([r[c] for r in good], dtype=float)
        b = np.array([r[c] for r in bad], dtype=float)
        if len(g) == 0 or len(b) == 0:
            continue
        try:
            p = stats.mannwhitneyu(g, b).pvalue
        except ValueError:
            p = float("nan")
        split[c] = dict(good=float(g.mean()), bad=float(b.mean()), p=float(p))
        print(f"{c:>22} | {g.mean():>12.4f} {b.mean():>12.4f} | "
              f"{b.mean()/g.mean() if g.mean() else float('nan'):>7.2f} {p:>9.3g}")

    print("\n=== morphology: is it a genus/species population effect? ===")
    byg = defaultdict(list)
    for r in real:
        byg[r["genus"]].append(r["cycle_mean"] / s_ref)
    multi = {g: v for g, v in byg.items() if len(v) >= 2}
    print(f"  {len(byg)} genera, {len(multi)} with >=2 specimens")
    for g, v in sorted(byg.items(), key=lambda kv: -np.mean(kv[1]))[:10]:
        print(f"    {g:>22}: n={len(v)} mean_ratio={np.mean(v):.2f}x")
    if multi:
        within = np.mean([np.std(v) for v in multi.values()])
        between = np.std([np.mean(v) for v in multi.values()])
        print(f"  within-genus spread {within:.3f} vs between-genus spread {between:.3f} -> "
              f"{'BETWEEN dominates: morphology/population effect' if between > within else 'WITHIN dominates: per-scan quality effect, not species'}")

    with open(os.path.join(out_dir, "bench50_degradation_causes.json"), "w") as fh:
        json.dump({"config": vars(args), "synth_ref": s_ref, "spearman": corr,
                   "good_bad_split": split, "rows": rows}, fh, indent=2)
    print(f"\nwrote {out_dir}/bench50_degradation_causes.json")


if __name__ == "__main__":
    main()
