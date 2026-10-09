"""G7 -- is the surface objective COMPATIBLE with anatomical accuracy, or in conflict with it?

G6 measured the two corners: the production objective alone gives 38.9% joint residual at chamfer
0.00094; joints alone (inside the prior) give 11.0% at chamfer 0.032. It did not show whether the
middle is populated.

That distinction decides how deep the problem is:

  COMPATIBLE  a knee exists where chamfer stays near 0.00094 AND the joint residual approaches
              11%. Then the surface data does not actually oppose anatomical correctness, the
              objective is simply missing a term, and adding one is a cheap fix.
  CONFLICT    the frontier is steep everywhere -- every unit of joint accuracy costs real surface
              fit. Then surface geometry alone cannot identify the skeleton, which is a much
              deeper statement about registration from surface data than anything measured so far,
              and it would mean the skeleton must be supplied from OUTSIDE the surface objective
              (a predictor, extra observations) rather than recovered from it.

METHOD
Optimise `production objective + lambda * joint_target` from the production parameters, sweeping
lambda over decades. `betas` are projected into |z| <= 1 every step, because G6 showed the
anatomical accuracy is already available at the first standard deviation and it keeps every point
on the frontier plausible. Free parameters include `deform_verts`, matching what production itself
was allowed, so the lambda = 0 endpoint is a genuine control: starting at production and
re-optimising with lambda = 0 must stay near production, and if it does not, the sweep is not
measuring lambda.

Reported at each lambda: joint residual against the human annotations, chamfer, and the full
production objective. The frontier is (chamfer, joint residual).
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
N_IT = int(os.environ.get("G7_ITERS", "800"))
LAMBDAS = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 10.0]
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
        ann.append({"sid": sid, "obj": obj, "idx": idx, "bt": use.index("b_t"),
                    "ba3": use.index("b_a_3"),
                    "gt": torch.as_tensor(sim_fit(np.array([P[n] for n in use]), pj[sid][idx]),
                                          device=dev, dtype=torch.float32)})
    NS = len(ann)
    _, targets = load_meshes(mesh_files=[a["obj"] for a in ann], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in ann], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/g7")
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

    def jterm(J):
        return torch.stack([((J[i, ann[i]["idx"]] - ann[i]["gt"]) ** 2).sum(-1).mean()
                            for i in range(NS)]).mean()

    def report():
        with torch.no_grad():
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, comp = stage.forward(src, it=0)
            J = joints(v)
            out = []
            for i, a in enumerate(ann):
                r = (J[i, a["idx"]] - a["gt"]).norm(dim=-1)
                bl = float((a["gt"][a["ba3"]] - a["gt"][a["bt"]]).norm())
                out.append(float(torch.median(r)) / bl * 100)
        return float(np.median(out)), float(comp.get("chamfer", np.nan)), float(tot), \
            float((smal.betas.detach().abs() / sd_t).mean())

    setp(P0)
    a0, c0, t0, z0 = report()
    print("=" * 104)
    print("G7 -- Pareto frontier: production objective + lambda * joint target")
    print("=" * 104)
    print(f"{'lambda':>10}{'joint resid':>13}{'chamfer':>12}{'objective':>12}{'betas |z|':>11}"
          f"{'chamfer vs prod':>17}")
    print(f"{'(production)':>10}{a0:>12.2f}%{c0:>12.5f}{t0:>12.5f}{z0:>11.2f}{'1.00x':>17}")
    rows = [{"lambda": None, "anat": a0, "chamfer": c0, "total": t0, "z": z0}]

    for lam in LAMBDAS:
        setp(P0)
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            loss = tot + lam * jterm(joints(v))
            loss.backward()
            opt.step()
            with torch.no_grad():
                smal.betas.clamp_(min=-sd_t, max=sd_t)      # keep every point plausible
        a, c, t, z = report()
        rows.append({"lambda": lam, "anat": a, "chamfer": c, "total": t, "z": z})
        print(f"{lam:>10.4g}{a:>12.2f}%{c:>12.5f}{t:>12.5f}{z:>11.2f}{c/c0:>16.2f}x",
              flush=True)

    print("\n" + "=" * 104)
    sw = [r for r in rows if r["lambda"] is not None]
    ctrl = sw[0]
    print(f"[CONTROL] lambda=0 re-optimised from production: joints {ctrl['anat']:.2f}% "
          f"(production {a0:.2f}%), chamfer {ctrl['chamfer']:.5f} (production {c0:.5f})")
    ok = abs(ctrl["chamfer"] - c0) / c0 < 0.5
    print(f"          stays near production: {ok}  -- if False the sweep is not measuring lambda")

    # knee: best joint residual whose chamfer is within 2x of production
    cheap = [r for r in sw if r["chamfer"] <= 2 * c0]
    print()
    if cheap:
        b = min(cheap, key=lambda r: r["anat"])
        print(f"best joint residual at <=2x production chamfer: {b['anat']:.2f}% "
              f"(lambda {b['lambda']:.4g}, chamfer {b['chamfer']:.5f} = {b['chamfer']/c0:.2f}x)")
        if b["anat"] <= 0.6 * a0:
            print("\n=> COMPATIBLE. Anatomical accuracy is available at essentially no cost in")
            print("   surface fit. The objective is simply missing a joint-position term; adding")
            print("   one is a cheap and well-posed fix.")
        else:
            print("\n=> CONFLICT. Holding surface fit near production buys little joint accuracy.")
            print("   Surface geometry alone does not identify the skeleton; it must be supplied")
            print("   from outside the surface objective.")
    else:
        print("=> CONFLICT. No point on the sweep keeps chamfer within 2x of production.")
    print("=" * 104)
    json.dump(rows, open(os.path.join(HERE, "g7_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
