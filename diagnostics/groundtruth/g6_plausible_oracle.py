"""G6 -- how much anatomical accuracy is reachable INSIDE the model's own plausible shape space?

WHY G3 AND G4 DO NOT ANSWER THIS
G3 fitted the model's joints to the human annotations with no constraint on `betas` and reached a
3.7% residual, which was read as "the representation can express the truth" (Case B). G4 then
evaluated the production objective at that solution and found it scored far worse than the
production fit, which was read as "the objective is wrong" (B1).

Both readings are wrong, for the same reason: **the unconstrained oracle drives `betas` to |z| ~ 19**
-- nineteen prior standard deviations, against the production fit's |z| ~ 1.0. It matches the joint
targets by leaving the shape space, producing a mesh that is anatomically absurd. Chamfer at that
solution is 0.623 against production's 0.00094 in a space normalised to extent ~1, which is the
surface reporting exactly that. An oracle outside the prior is not a near-truth solution and cannot
adjudicate Case A vs Case B.

WHAT THIS DOES INSTEAD
Fit the joints to the annotations under a hard plausibility bound, projecting `betas` back into
|z| <= k after every step, and sweep k. Report BOTH the anatomical joint residual and the chamfer,
so a solution that buys joints by destroying the surface is visible rather than hidden.

    k = 1, 2, 3   inside the shape prior, i.e. shapes the model asserts are plausible
    k = inf       the G3/G4 regime, kept only to reproduce the artefact and show its cost

THE READING
  * a plausible-k oracle reaches a residual far below production  -> CASE B stands. The objective
    is not finding a solution that is both reachable and plausible.
  * plausible-k oracles cannot beat production                    -> CASE A. Within its own prior
    the model genuinely cannot express the observed anatomy, and G3's verdict is overturned.

Chamfer is reported at every k because it is the term that exposed the artefact, and because a
solution is only interesting if it is simultaneously good on joints and not absurd on the surface.
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
N_IT = int(os.environ.get("G6_ITERS", "2000"))
KS = [1.0, 2.0, 3.0, float("inf")]


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
    sd = np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64)))
    sd_t = torch.as_tensor(sd, device=dev, dtype=torch.float32)
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
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/g6")

    P0 = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in ann]), device=dev,
                             dtype=torch.float32) for k in prod[ann[0]["sid"]]}

    def setp(d):
        with torch.no_grad():
            for k, v in d.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def joints():
        v = smal()
        return (torch.einsum("jv,nvc->njc", smal.smal_model.J_regressor, v)
                if smal.smal_model.J_regressor.shape[1] == v.shape[1]
                else torch.einsum("vj,nvc->njc", smal.smal_model.J_regressor, v))

    def anat():
        with torch.no_grad():
            J = joints()
        out = []
        for i, a in enumerate(ann):
            r = (J[i, a["idx"]] - a["gt"]).norm(dim=-1)
            bl = float((a["gt"][a["ba3"]] - a["gt"][a["bt"]]).norm())
            out.append(float(torch.median(r)) / bl * 100)
        return float(np.median(out))

    def obj_terms():
        with torch.no_grad():
            nv = smal()
            src = stage.src_mesh.offset_verts((nv - stage.src_verts).view(-1, 3))
            loss, comp = stage.forward(src, it=0)
        return float(loss), {k: float(v) for k, v in comp.items()}

    setp(P0)
    p_anat = anat(); p_tot, p_comp = obj_terms()
    p_z = float((P0["betas"].abs() / sd_t).mean())
    print("=" * 100)
    print("G6 -- anatomical accuracy reachable inside the model's own shape prior")
    print("=" * 100)
    print(f"{'setting':<26}{'betas |z|':>11}{'joint resid':>13}{'chamfer':>12}{'objective':>12}")
    print(f"{'production fit':<26}{p_z:>11.2f}{p_anat:>12.2f}%{p_comp.get('chamfer', np.nan):>12.5f}"
          f"{p_tot:>12.5f}")

    rows = [{"setting": "production", "z": p_z, "anat": p_anat,
             "chamfer": p_comp.get("chamfer"), "total": p_tot}]
    FREE = ["global_rot", "joint_rot", "betas", "trans"]
    for k in KS:
        setp(P0)
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.02)
        for it in range(N_IT):
            opt.zero_grad()
            J = joints()
            loss = torch.stack([((J[i, ann[i]["idx"]] - ann[i]["gt"]) ** 2).sum(-1).mean()
                                for i in range(NS)]).mean()
            loss.backward(); opt.step()
            if np.isfinite(k):                      # hard projection back into the prior
                with torch.no_grad():
                    smal.betas.clamp_(min=-k * sd_t, max=k * sd_t)
        z = float((smal.betas.detach().abs() / sd_t).mean())
        a = anat(); tot, comp = obj_terms()
        lbl = f"oracle, |z| <= {k:.0f}" if np.isfinite(k) else "oracle, UNCONSTRAINED"
        print(f"{lbl:<26}{z:>11.2f}{a:>12.2f}%{comp.get('chamfer', np.nan):>12.5f}{tot:>12.5f}",
              flush=True)
        rows.append({"setting": lbl, "z": z, "anat": a, "chamfer": comp.get("chamfer"),
                     "total": tot})

    print("\n" + "=" * 100)
    plaus = [r for r in rows if r["setting"].startswith("oracle, |z|")]
    best = min(plaus, key=lambda r: r["anat"])
    print(f"best PLAUSIBLE oracle : {best['setting']}  joint residual {best['anat']:.2f}%")
    print(f"production fit        : {p_anat:.2f}%")
    if best["anat"] < 0.6 * p_anat:
        print("\n=> CASE B stands. A solution that is BOTH inside the shape prior and far more")
        print("   anatomically accurate than the production fit exists. The objective is not")
        print("   finding it.")
    elif best["anat"] > 0.9 * p_anat:
        print("\n=> CASE A. Inside its own prior the model cannot do materially better than the")
        print("   production fit. G3's Case B verdict is OVERTURNED: the 3.7% was bought with")
        print("   |z| ~ 19 betas, i.e. by leaving the shape space.")
    else:
        print("\n=> PARTIAL. Some accuracy is reachable within the prior, but far less than the")
        print("   unconstrained oracle suggested. Report the fraction; neither case is clean.")
    print("=" * 100)
    json.dump(rows, open(os.path.join(HERE, "g6_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
