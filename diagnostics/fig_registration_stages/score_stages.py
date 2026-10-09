"""Score every registration stage of a D1_PROD run on the Atta scans.

For all 20 specimens and stages init, H0_body, H1_legs, H2_joint, Stage_2, Stage_3:
  - gate 1: verts rebuilt from each npz's parameters must reproduce the saved verts (< 1e-4);
            init = SMAL3DFitter default state, which must carry zero deform;
  - chamfer_l2 and F@0.01 from diagnostics/moonshot/metrics.surface_metrics (5 sampling seeds);
  - secondary F@0.01 from fitter_3d/eval_metrics.f_score (tau = 1% of bbox diagonal);
  - penetration_loss_batched with trainer.py's eval arguments (hard / soft / gaster-legs / unique
    vertices) and GWN gaster-legs num_inside as a cross-check;
  - gate 2: focus Stage_3 vs D1_PROD's own metrics.csv;
  - gate 3: independent numpy + scipy cKDTree recompute for the focus specimen.

Writes stages_all20.csv, stages.csv, probes/gates.json, probes/focus_meshes.npz.

    python diagnostics/fig_registration_stages/score_stages.py \
        --hier_dir <results>/D1_s0_hier --moon_dir <results>/D1_s0 --mesh_dir <atta_scans>
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "moonshot"))

import config  # noqa: E402
import metrics as M  # noqa: E402  (diagnostics/moonshot/metrics.py, as optimise_moonshot --eval)
from pytorch3d.structures import Meshes  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from fitter_3d.eval_metrics import f_score  # noqa: E402
from fitter_3d.gwn_penetration_loss import gwn_penetration_loss_batched, precompute_capped_topology  # noqa: E402
from fitter_3d.part_groups import PART_GROUPS_COARSE, get_non_adjacent_pairs, get_part_vertex_indices  # noqa: E402
from fitter_3d.penetration_loss import _build_part_faces, penetration_loss_batched  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

PARAMS = ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales", "betas_trans", "deform_verts"]
REPRO_TOL = 1e-4
SURF_SEEDS = [0, 1, 2, 3, 4]
FIG_STAGES = ["init", "H2_joint", "Stage_2_deform_coarse", "Stage_3_deform_fine"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def stage_verts(args, names):
    """{stage: (B, V, 3) tensor} in specimen order `names`, after the reproduction guard."""
    out, guard = {}, {}
    smal = SMAL3DFitter(batch_size=len(names), device=DEV, shape_family=-1)
    with torch.no_grad():
        v0 = smal()
    assert float(smal.deform_verts.abs().max()) == 0.0, "init state carries non-zero deform"
    vt = torch.as_tensor(np.asarray(config.dd["v_template"], np.float32), device=DEV)
    guard["init_vs_v_template_maxabs"] = float((v0[0] - vt).abs().max())
    out["init"] = v0
    files = [
        (os.path.join(args.hier_dir, f"{s}.npz"), s) for s in ("H0_body", "H1_legs", "H2_joint")
    ] + [(os.path.join(args.moon_dir, f"{s}.npz"), s) for s in ("Stage_2_deform_coarse", "Stage_3_deform_fine")]
    for path, st in files:
        z = np.load(path, allow_pickle=True)
        labels = [os.path.splitext(str(x))[0] for x in z["labels"]]
        assert labels == names, f"{st}: label order {labels} != mesh order {names}"
        with torch.no_grad():
            for k in PARAMS:
                src = torch.as_tensor(np.asarray(z[k], np.float32), device=DEV)
                dst = getattr(smal, k)
                assert tuple(dst.shape) == tuple(src.shape), (st, k, dst.shape, src.shape)
                dst.copy_(src)
            rebuilt = smal()
        saved = torch.as_tensor(np.asarray(z["verts"], np.float32), device=DEV)
        err = float((rebuilt - saved).abs().max())
        guard[f"{st}_repro_maxabs"] = err
        guard[f"{st}_deform_absmax"] = float(np.abs(z["deform_verts"]).max())
        if err >= REPRO_TOL:
            raise SystemExit(f"GATE 1 FAIL: {st} rebuilt verts differ from saved by {err:.2e}")
        out[st] = saved
    faces = smal.faces[0].detach()
    return out, faces, guard


def surface(pred_v, faces, tgt):
    rows = []
    for s in SURF_SEEDS:
        torch.manual_seed(s)
        r = M.surface_metrics(Meshes(verts=[pred_v], faces=[faces]), tgt)
        rows.append((r["chamfer_l2"], r["fscore@0.01"]))
    a = np.array(rows)
    return a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std()


def numpy_surface(pred_v, faces, tv, tf, n=30000, seed=0, tau=0.01):
    """Gate 3: area-weighted sampling + cKDTree, no pytorch3d."""
    rng = np.random.default_rng(seed)

    def sample(v, f):
        tri = v[f]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        idx = rng.choice(len(f), n, p=area / area.sum())
        u, w = rng.random(n), rng.random(n)
        flip = u + w > 1
        u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
        t = tri[idx]
        return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + w[:, None] * (t[:, 2] - t[:, 0])

    p, q = sample(pred_v, faces), sample(tv, tf)
    d_pq, _ = cKDTree(q).query(p)
    d_qp, _ = cKDTree(p).query(q)
    prec, rec = (d_pq < tau).mean(), (d_qp < tau).mean()
    return float((d_pq**2).mean() + (d_qp**2).mean()), float(2 * prec * rec / max(prec + rec, 1e-12))


def pair_sum(per_pair, a, b, key):
    tot = 0.0
    for k in (f"{a}__{b}", f"{b}__{a}"):
        if k in per_pair:
            tot = tot + per_pair[k]["A_into_B"][key] + per_pair[k]["B_into_A"][key]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hier_dir", required=True)
    ap.add_argument("--moon_dir", required=True)
    ap.add_argument("--mesh_dir", required=True)
    ap.add_argument("--focus", default="13")
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "probes"), exist_ok=True)
    assert os.path.basename(config.SMAL_FILE) == "OmniAnt_25PCs_joint_limited.pkl", (
        f"config resolved {config.SMAL_FILE}; export SMILIFY_SMAL_FILE as in submit.sbatch")

    files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))  # same order as both fitters
    names, targets = load_meshes(mesh_files=files, device=DEV)
    fi = names.index(args.focus)

    verts, faces, guard = stage_verts(args, names)
    print("[gate 1] " + json.dumps(guard, indent=1), flush=True)

    pvi = get_part_vertex_indices(PART_GROUPS_COARSE)
    pairs = get_non_adjacent_pairs(PART_GROUPS_COARSE)
    pfaces = _build_part_faces(faces.cpu().numpy(), pvi)
    capped = precompute_capped_topology(np.asarray(config.dd["v_template"], np.float32), pfaces)
    gl_pair = [p for p in pairs if set(p) == {"gaster", "legs"}]
    assert len(gl_pair) == 1, f"gaster-legs not a non-adjacent pair: {pairs}"

    rows = []
    for st, V in verts.items():
        with torch.no_grad():
            _, pd = penetration_loss_batched(
                verts_padded=V, part_vertex_indices=pvi, part_faces=pfaces, non_adjacent_pairs=pairs,
                proximity_tau_fraction=0.03, max_depth_fraction=0.08, iteration=0, n_ramp_iters=0,
                return_diagnostics=True, return_per_pair=True,
            )
            _, gd = gwn_penetration_loss_batched(
                verts_padded=V, part_vertex_indices=pvi, capped_topology=capped, pairs=gl_pair,
                iteration=0, n_ramp_iters=0, return_diagnostics=True,
            )
            fb = f_score(Meshes(verts=list(V), faces=[faces] * len(names)), targets, tau_fractions=[0.01],
                         n_samples=10_000)
        for i, name in enumerate(names):
            ch, ch_sd, f1, f1_sd = surface(V[i], faces, targets[i])
            rows.append({
                "specimen": name, "stage": st,
                "chamfer_l2": ch, "chamfer_l2_sd_sampling": ch_sd,
                "fscore@0.01": f1, "fscore@0.01_sd_sampling": f1_sd,
                "fscore@0.01bboxdiag": float(fb["f_score"][i, 0]),
                "pen_hard_triples": float(pd["num_penetrating"][i]),
                "pen_soft": float(pd["soft_num_penetrating"][i]),
                "pen_gaster_legs_hard": float(pair_sum(pd["per_pair"], "gaster", "legs", "num_penetrating")[i]),
                "pen_unique_verts": int(pd["penetrating_vertex_mask"][i].sum()),
                "gwn_gaster_legs_inside": float(gd["num_inside"][i]),
            })
        print(f"[score] {st} done", flush=True)

    keys = list(rows[0])
    with open(os.path.join(args.out_dir, "stages_all20.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["rank_chamfer_of20", "rank_fscore_of20"])
        w.writeheader()
        for st in verts:
            sr = [r for r in rows if r["stage"] == st]
            ch_rank = {r["specimen"]: k + 1 for k, r in enumerate(sorted(sr, key=lambda r: r["chamfer_l2"]))}
            f_rank = {r["specimen"]: k + 1 for k, r in enumerate(sorted(sr, key=lambda r: -r["fscore@0.01"]))}
            for r in sr:
                w.writerow({**r, "rank_chamfer_of20": ch_rank[r["specimen"]], "rank_fscore_of20": f_rank[r["specimen"]]})
    with open(os.path.join(args.out_dir, "stages.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["panel"] + keys + ["median20_chamfer_l2", "median20_fscore@0.01",
                                                              "median20_pen_hard_triples"])
        w.writeheader()
        for st in verts:
            sr = [r for r in rows if r["stage"] == st]
            r = next(r for r in sr if r["specimen"] == args.focus)
            w.writerow({"panel": st in FIG_STAGES, **r,
                        "median20_chamfer_l2": float(np.median([x["chamfer_l2"] for x in sr])),
                        "median20_fscore@0.01": float(np.median([x["fscore@0.01"] for x in sr])),
                        "median20_pen_hard_triples": float(np.median([x["pen_hard_triples"] for x in sr]))})

    # gate 2: our Stage_3 numbers vs D1_PROD's own metrics.csv (one 30k sample there)
    gates = {"gate1": guard}
    mine = next(r for r in rows if r["specimen"] == args.focus and r["stage"] == "Stage_3_deform_fine")
    with open(os.path.join(args.moon_dir, "metrics.csv")) as f:
        theirs = next(r for r in csv.DictReader(f) if os.path.splitext(r["mesh"])[0] == args.focus)
    g2 = {}
    for k, sdk in (("chamfer_l2", "chamfer_l2_sd_sampling"), ("fscore@0.01", "fscore@0.01_sd_sampling")):
        d = abs(mine[k] - float(theirs[k]))
        tol = max(4 * mine[sdk], 0.02 * abs(mine[k]), 1e-6)
        g2[k] = {"ours_mean5": mine[k], "d1prod_metrics_csv": float(theirs[k]), "absdiff": d, "tol": tol,
                 "pass": bool(d <= tol)}
    gates["gate2"] = g2

    # gate 3: independent recompute for the focus specimen, every figure stage
    tv = targets[fi].verts_packed().cpu().numpy()
    tf = targets[fi].faces_packed().cpu().numpy()
    g3 = {}
    for st in FIG_STAGES:
        ch, f1 = numpy_surface(verts[st][fi].cpu().numpy(), faces.cpu().numpy(), tv, tf)
        r = next(r for r in rows if r["specimen"] == args.focus and r["stage"] == st)
        g3[st] = {"numpy_chamfer_l2": ch, "p3d_chamfer_l2": r["chamfer_l2"], "numpy_fscore@0.01": f1,
                  "p3d_fscore@0.01": r["fscore@0.01"],
                  "pass": bool(abs(ch - r["chamfer_l2"]) <= max(0.05 * r["chamfer_l2"], 1e-6)
                               and abs(f1 - r["fscore@0.01"]) <= 0.01)}
    gates["gate3"] = g3
    with open(os.path.join(args.out_dir, "probes", "gates.json"), "w") as f:
        json.dump(gates, f, indent=1)
    print("[gates 2/3] " + json.dumps({"gate2": g2, "gate3": g3}, indent=1), flush=True)

    with torch.no_grad():
        _, pd = penetration_loss_batched(
            verts_padded=verts["Stage_3_deform_fine"], part_vertex_indices=pvi, part_faces=pfaces,
            non_adjacent_pairs=pairs, iteration=0, n_ramp_iters=0, return_diagnostics=True,
        )
    part_of_vertex = np.full(int(faces.max()) + 1, "", dtype=object)
    for p, ix in pvi.items():
        part_of_vertex[np.asarray(ix, dtype=np.int64)] = p
    np.savez(
        os.path.join(args.out_dir, "probes", "focus_meshes.npz"),
        faces=faces.cpu().numpy(), target_verts=tv, target_faces=tf,
        pen_mask_stage3=pd["penetrating_vertex_mask"][fi].cpu().numpy(),
        part_of_vertex=part_of_vertex,
        **{st: verts[st][fi].cpu().numpy() for st in verts},
    )
    print("[score] done", flush=True)


if __name__ == "__main__":
    main()
