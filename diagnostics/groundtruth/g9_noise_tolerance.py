"""G9 -- does the fit BEAT its own joint targets, or merely reproduce them?

WHY THE OBVIOUS VERSION OF THIS EXPERIMENT IS NOT WORTH RUNNING
"Does the benefit degrade as the targets get noisier" has an arithmetic answer: a squared-distance
penalty pulls the solution toward its targets, so it degrades roughly with target error. Measuring
that confirms the definition of the term.

THE QUESTION THAT IS ACTUALLY DECISIVE
The surface objective is acting at the same time. So when the targets are wrong by eps, where does
the fit land?

  fit residual <  eps   the surface term DENOISES the targets. A mediocre predictor still yields a
                        good fit; SMILify contributes something beyond what it is handed. The
                        deployment path from G7 is open even with a weak predictor.
  fit residual ~= eps   PASS-THROUGH. The joint term makes the fit reproduce whatever skeleton it
                        is given. SMILify is then an interpolator on this axis and one may as well
                        use the predictor's joints directly -- which would close the deployment
                        path before anyone builds the predictor.
  fit residual >  eps   amplification; worse than using the targets raw.

So the comparator is not "production" but **the targets themselves**, and the endpoint is the ratio
fit/target. That is the number this exists to produce.

DESIGN
Full coverage (G8 showed partial coverage does not propagate, so a noisy-target study on a subset
would confound two effects). Isotropic Gaussian perturbation of every target, sigma expressed in %
of each specimen's mesosoma length so it is comparable across specimens and directly readable
against G1's error scale. Realised target error is MEASURED rather than assumed, because the
median of a perturbed set is not the perturbation sigma.

Two controls:
  * sigma = 0 must reproduce G7's numbers, or the harness is not the same experiment.
  * "targets raw" is reported at every sigma -- the trivial baseline of simply believing the
    predictor. The fit has to beat that to be worth running at all.
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
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import pickle  # noqa: E402

import yaml  # noqa: E402

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
N_IT = int(os.environ.get("G9_ITERS", "700"))
SIGMAS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]     # % of mesosoma length
LAM = float(os.environ.get("G9_LAM", "0.1"))
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
        bl = float((g[use.index("b_a_3")] - g[use.index("b_t")]).norm())
        ann.append({"sid": sid, "obj": obj, "idx": idx, "gt": g, "bl": bl})
    NS = len(ann)
    _, targets = load_meshes(mesh_files=[a["obj"] for a in ann], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in ann], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/g9")
    P0 = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in ann]), device=dev,
                             dtype=torch.float32) for k in prod[ann[0]["sid"]]}

    def setp(d):
        with torch.no_grad():
            for k, v in d.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def joints(v):
        return (torch.einsum("jv,nvc->njc", smal.smal_model.J_regressor, v)
                if smal.smal_model.J_regressor.shape[1] == v.shape[1]
                else torch.einsum("vj,nvc->njc", smal.smal_model.J_regressor, v))

    def resid_vs_truth():
        with torch.no_grad():
            J = joints(smal())
        return np.median([float(torch.median((J[i, a["idx"]] - a["gt"]).norm(dim=-1))) / a["bl"]
                          * 100 for i, a in enumerate(ann)])

    setp(P0)
    prod_resid = resid_vs_truth()
    print(f"{NS} specimens.  production fit vs truth: {prod_resid:.2f}%   lambda={LAM}\n")
    print("=" * 104)
    print("G9 -- does the fit beat its own targets?   (full coverage; sigma in % of mesosoma length)")
    print("=" * 104)
    print(f"{'sigma':>7}{'targets raw':>14}{'FIT vs truth':>15}{'fit/targets':>13}"
          f"{'vs production':>15}{'chamfer':>11}")
    rows = []
    for sig in SIGMAS:
        rng = np.random.default_rng(int(sig * 7) + 11)
        tgt, terr = [], []
        for a in ann:
            n = a["gt"].shape[0]
            e = torch.as_tensor(rng.normal(0.0, sig / 100.0 * a["bl"], size=(n, 3)),
                                device=dev, dtype=torch.float32)
            t = a["gt"] + e
            tgt.append(t)
            terr.append(float(torch.median(e.norm(dim=-1))) / a["bl"] * 100)
        t_raw = float(np.median(terr))          # MEASURED target error, not assumed

        setp(P0)
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            J = joints(v)
            jt = torch.stack([((J[i, ann[i]["idx"]] - tgt[i]) ** 2).sum(-1).mean()
                              for i in range(NS)]).mean()
            (tot + LAM * jt).backward()
            opt.step()
            with torch.no_grad():
                smal.betas.clamp_(min=-sd_t, max=sd_t)
        fit = resid_vs_truth()
        with torch.no_grad():
            ch = float(stage.forward(stage.src_mesh.offset_verts(
                (smal() - stage.src_verts).view(-1, 3)), it=0)[1].get("chamfer", np.nan))
        ratio = fit / t_raw if t_raw > 1e-9 else float("nan")
        rows.append({"sigma": sig, "target_err": t_raw, "fit": fit, "ratio": ratio,
                     "chamfer": ch})
        print(f"{sig:>6.0f}%{t_raw:>13.2f}%{fit:>14.2f}%{ratio:>13.2f}"
              f"{fit / prod_resid:>14.2f}x{ch:>11.5f}", flush=True)

    print("\n" + "=" * 104)
    z = rows[0]
    print(f"[CONTROL] sigma=0 reproduces G7 (expected ~9% at lambda 0.1): {z['fit']:.2f}%")
    nz = [r for r in rows if r["sigma"] > 0]
    med = float(np.median([r["ratio"] for r in nz]))
    print(f"median fit/target ratio over the noisy levels: {med:.2f}")
    beats = [r for r in nz if r["ratio"] < 0.9]
    useful = [r for r in nz if r["fit"] < prod_resid]
    print(f"levels where the fit BEATS its targets (ratio<0.9): {len(beats)}/{len(nz)}")
    print(f"levels where the fit still beats the production fit: {len(useful)}/{len(nz)}"
          f"  (worst usable target error: "
          f"{max([r['target_err'] for r in useful], default=float('nan')):.1f}%)")
    print()
    if med < 0.9:
        print("=> DENOISING. The fit lands closer to truth than the skeleton it was given, so a")
        print("   mediocre predictor still yields a good fit and SMILify adds value on this axis.")
    elif med < 1.1:
        print("=> PASS-THROUGH. The fit reproduces whatever skeleton it is handed. On this axis")
        print("   SMILify is an interpolator: the accuracy comes entirely from the predictor, and")
        print("   the deployment path needs a predictor already as good as the target accuracy.")
    else:
        print("=> AMPLIFICATION. The fit is further from truth than its own targets.")
    print("=" * 104)
    json.dump({"production": prod_resid, "lambda": LAM, "rows": rows},
              open(os.path.join(HERE, "g9_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
