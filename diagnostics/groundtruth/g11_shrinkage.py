"""G11 -- is it shrinkage, and does relaxing the shape prior fix it?

G10 left one thing unexplained: with EXACT joint targets the fit reaches only R = 0.507 while the
targets are 1.000 by construction. The fit does not fully honour the skeleton it is given.

HYPOTHESIS: shrinkage toward the mean shape. `betas` are clamped to |z| <= 1 and the surface term
pulls every specimen toward configurations the shape space prefers. That REDUCES absolute joint
position error -- exactly what G9 measured and called denoising -- while COMPRESSING between-specimen
variation, which is exactly what R measures. A shrinkage estimator is simultaneously more accurate
per specimen and worse at ranking specimens, which is the only way G9 and G10 can both be true.

This does two things, because a diagnosis that does not point at a fix is not worth a run:

  1. DIAGNOSE. Compare the spread (sd of log-shape-ratios) of ground truth, the noisy targets, the
     production fit, and the joint-constrained fit. Compression relative to GT is the signature.
  2. TREAT. Sweep the clamp |z| <= {1, 2, 3, inf} and ask whether relaxing it lifts R toward the
     targets-direct baseline -- i.e. whether the prior is doing work the joint term now does better.

CONTROL, and it decides how the result is attributed: every clamp level is run BOTH with and
WITHOUT the joint term. If relaxing the prior improves R even with no joint term, the finding is
"the shape prior is too tight", which is a different and more general claim than "the prior fights
the joint term".

DEGENERACY WATCH: G6 showed unconstrained betas reach |z| ~ 26 and destroy the mesh when nothing
else constrains them. Here the full surface objective is active, which should hold it together --
but |z| and chamfer are reported at every level so a degenerate solution is visible rather than
inferred.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import pickle  # noqa: E402

import yaml  # noqa: E402

import measure as ms  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
N_IT = int(os.environ.get("G11_ITERS", "700"))
SIGMA = 15.0
LAM = 0.1
CLAMPS = [1.0, 2.0, 3.0, float("inf")]
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]


def sim_fit(src, tgt):
    cs, ct = src.mean(0), tgt.mean(0)
    s0, t0 = src - cs, tgt - ct
    U, S, Vt = np.linalg.svd(s0.T @ t0)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    sc = (S * np.array([1, 1, d])).sum() / (s0 ** 2).sum()
    return sc * ((src - cs) @ R) + ct


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    sd_t = torch.as_tensor(np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64))),
                           device=dev, dtype=torch.float32)
    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float64)
    M = ms.load_model(); bones = ms.bone_table(M); bnames = [b["name"] for b in bones]

    prod, pj = {}, {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        V = np.asarray(d["verts"], dtype=np.float64)
        Jf = (np.einsum("jv,nvc->njc", Jr, V) if Jr.shape[1] == V.shape[1]
              else np.einsum("vj,nvc->njc", Jr, V))
        for i, lab in enumerate(d["labels"]):
            sid = str(lab).replace("_processed.obj", "").replace(".obj", "")
            prod[sid] = {k: np.asarray(d[k], dtype=np.float64)[i] for k in
                         ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                          "betas_trans", "deform_verts"]}
            pj[sid] = Jf[i]

    ann = []
    for f in sorted(glob.glob(os.path.join(REPO, "annotation/gt_batch1/*_joints.json"))):
        gt = json.load(open(f)); sid = gt["specimen_id"]
        obj = os.path.join(MESHDIR, f"{sid}_processed.obj")
        if sid not in prod or not os.path.exists(obj):
            continue
        P = {j["joint_name"]: np.array(j["position"], dtype=np.float64)
             for j in gt["joints"] if j.get("position") is not None}
        use = [n for n in names if n in P and n not in WING and n != "b_h"]
        idx = [names.index(n) for n in use]
        g = torch.as_tensor(sim_fit(np.array([P[n] for n in use]), pj[sid][idx]),
                            device=dev, dtype=torch.float32)
        full = torch.full((len(names), 3), float("nan"), device=dev); full[idx] = g
        ann.append({"sid": sid, "obj": obj, "idx": idx, "gt": g, "gt_full": full,
                    "bl": float((g[use.index("b_a_3")] - g[use.index("b_t")]).norm())})
    NS = len(ann)
    _, targets = load_meshes(mesh_files=[a["obj"] for a in ann], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in ann], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/g11")
    P0 = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in ann]), device=dev,
                             dtype=torch.float32) for k in prod[ann[0]["sid"]]}

    def setp(d):
        with torch.no_grad():
            for k, v in d.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def joints_of(v):
        return (torch.einsum("jv,nvc->njc", smal.smal_model.J_regressor, v)
                if smal.smal_model.J_regressor.shape[1] == v.shape[1]
                else torch.einsum("vj,nvc->njc", smal.smal_model.J_regressor, v))

    def blen(Jfull):
        return np.stack([np.linalg.norm(Jfull[:, b["child"]] - Jfull[:, b["parent"]], axis=-1)
                         for b in bones], axis=1)

    GT = np.stack([a["gt_full"].cpu().numpy() for a in ann])
    Lgt = blen(GT)
    okref = np.isfinite(Lgt).all(0) & (Lgt > 0).all(0)

    def zscores(L):
        ok = okref & np.isfinite(L).all(0) & (L > 0).all(0)
        z = np.log(L[:, ok]) - np.log(L[:, ok]).mean(1, keepdims=True)
        return z, [bnames[i] for i in np.where(ok)[0]], ok

    def stats_vs_gt(L):
        """median R across bones, and the SPREAD ratio (fit sd / GT sd) -- the shrinkage signature."""
        z, cols, ok = zscores(L)
        gz = np.log(Lgt[:, ok]) - np.log(Lgt[:, ok]).mean(1, keepdims=True)
        rs, spread = [], []
        for k in range(z.shape[1]):
            g, f = gz[:, k], z[:, k]
            if g.std() < 1e-12 or f.std() < 1e-12:
                continue
            rs.append(float(np.corrcoef(f, g)[0, 1]))
            spread.append(f.std() / g.std())
        return float(np.median(rs)), float(np.median(spread))

    # noisy targets, fixed across all arms so the comparison is like-for-like
    # NOTE: a single noise realisation. G10 used a different seed and the J-vs-T comparison
    # flipped sign between them (see RESULTS_G11), so that comparison needs G12's multi-seed run.
    rng = np.random.default_rng(int(os.environ.get("G11_SEED", "97")))
    tgt, tfull = [], np.full((NS, len(names), 3), np.nan)
    for i, a in enumerate(ann):
        e = torch.as_tensor(rng.normal(0.0, SIGMA / 100.0 * a["bl"], size=(a["gt"].shape[0], 3)),
                            device=dev, dtype=torch.float32)
        t = a["gt"] + e
        tgt.append(t); tfull[i, a["idx"]] = t.cpu().numpy()

    setp(P0)
    Rp, Sp = stats_vs_gt(blen(joints_of(smal()).detach().cpu().numpy()))
    Rt, St = stats_vs_gt(blen(tfull))
    print(f"{NS} specimens.  sigma={SIGMA}%  lambda={LAM}\n")
    print("=" * 100)
    print("G11 -- shrinkage: does the fit compress between-specimen variation, and does relaxing")
    print("       the shape prior fix it?   spread = sd(fit) / sd(GT); 1.0 = no compression")
    print("=" * 100)
    print(f"{'arm':<34}{'median R':>10}{'spread':>9}{'chamfer':>11}{'betas |z|':>11}"
          f"{'joint err':>11}")
    print(f"{'GT (definition)':<34}{1.000:>10.3f}{1.000:>9.3f}{'--':>11}{'--':>11}{'0.00%':>11}")
    print(f"{'T targets direct':<34}{Rt:>+10.3f}{St:>9.3f}{'--':>11}{'--':>11}"
          f"{SIGMA:>10.1f}%")
    print(f"{'P production':<34}{Rp:>+10.3f}{Sp:>9.3f}"
          f"{'':>11}{float((P0['betas'].abs()/sd_t).mean()):>11.2f}{'':>11}")

    rows = [{"arm": "T targets", "R": Rt, "spread": St},
            {"arm": "P production", "R": Rp, "spread": Sp}]

    for use_joint in [False, True]:
        for cl in CLAMPS:
            setp(P0)
            for nm in FREE:
                getattr(smal, nm).requires_grad_(True)
            opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
            for it in range(N_IT):
                opt.zero_grad()
                v = smal()
                src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
                tot, _ = stage.forward(src, it=0)
                loss = tot
                if use_joint:
                    J = joints_of(v)
                    loss = loss + LAM * torch.stack(
                        [((J[i, ann[i]["idx"]] - tgt[i]) ** 2).sum(-1).mean()
                         for i in range(NS)]).mean()
                loss.backward(); opt.step()
                if np.isfinite(cl):
                    with torch.no_grad():
                        smal.betas.clamp_(min=-cl * sd_t, max=cl * sd_t)
            with torch.no_grad():
                v = smal()
                ch = float(stage.forward(stage.src_mesh.offset_verts(
                    (v - stage.src_verts).view(-1, 3)), it=0)[1].get("chamfer", np.nan))
                Jf = joints_of(v)
                je = float(np.median([float(torch.median((Jf[i, a["idx"]] - a["gt"]).norm(dim=-1)))
                                      / a["bl"] * 100 for i, a in enumerate(ann)]))
                z = float((smal.betas.detach().abs() / sd_t).mean())
            R, S = stats_vs_gt(blen(Jf.detach().cpu().numpy()))
            lbl = ("J joint term" if use_joint else "N no joint term") + \
                  (f", |z|<={cl:.0f}" if np.isfinite(cl) else ", |z| free")
            rows.append({"arm": lbl, "R": R, "spread": S, "chamfer": ch, "z": z, "joint_err": je})
            print(f"{lbl:<34}{R:>+10.3f}{S:>9.3f}{ch:>11.5f}{z:>11.2f}{je:>10.1f}%", flush=True)
        print()

    print("=" * 100)
    Jr_ = [r for r in rows if r["arm"].startswith("J")]
    Nr_ = [r for r in rows if r["arm"].startswith("N")]
    base = [r for r in Jr_ if "|z|<=1" in r["arm"]][0]
    bestJ = max(Jr_, key=lambda r: r["R"])
    bestN = max(Nr_, key=lambda r: r["R"])
    print(f"[DIAGNOSIS] spread of the joint-constrained fit at |z|<=1: {base['spread']:.3f} "
          f"(GT = 1.000)")
    print(f"  -> {'CONFIRMED: the fit compresses between-specimen variation' if base['spread'] < 0.9 else 'NOT CONFIRMED: spread is not compressed; shrinkage is not the mechanism'}")
    print(f"\n[CONTROL] best WITHOUT the joint term: {bestN['arm']} R {bestN['R']:+.3f} "
          f"(production {Rp:+.3f})")
    print(f"  -> {'relaxing the prior helps even with no joint term: the prior is simply too tight' if bestN['R'] > Rp + 0.1 else 'relaxing alone does little: the prior specifically fights the joint term'}")
    print(f"\n[TREATMENT] best WITH the joint term: {bestJ['arm']} R {bestJ['R']:+.3f}   "
          f"targets-direct {Rt:+.3f}")
    if bestJ["R"] > Rt:
        print("  => the relaxed prior lifts the fit ABOVE reading the skeleton directly.")
        print("     G10's PARTIAL becomes a PASS, and the recipe change is specific: relax the")
        print("     shape prior while the joint term is active.")
    else:
        print("  => still below targets-direct. Relaxing the prior is not sufficient; the")
        print("     ~20% crossover rule from G10 stands.")
    print("=" * 100)
    json.dump(rows, open(os.path.join(HERE, "g11_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
