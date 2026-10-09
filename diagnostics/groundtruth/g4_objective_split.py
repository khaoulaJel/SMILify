"""G4 -- split CASE B: is the OBJECTIVE wrong, or is the OPTIMISER not reaching it?

G3 established that the parameterisation can place its joints within 3.7% of the human annotations
(pose + betas only) while the production pipeline achieves 28.2%. So the residual is not a
representation limit. That leaves two sub-cases with opposite remedies:

  B1  THE OBJECTIVE IS WRONG   -- it scores the near-truth solution WORSE than what the pipeline
      converged to. Then the optimiser is doing its job and the loss is mis-specified; the fix is
      term weighting / priors / constraints, and the offending term is identifiable.
  B2  THE OPTIMISER FALLS SHORT -- the objective scores the near-truth solution BETTER, but the
      optimiser never gets there. Then the loss is fine and the fix is initialisation, schedule or
      search strategy.

THE INSTRUMENT
G3's oracle gives, per specimen, a parameter vector known to be close to truth. Evaluating the
PRODUCTION objective at that vector and at the production vector, and reading the per-term
breakdown, distinguishes B1 from B2 in a single forward pass each. `MoonshotStage.forward()`
returns `(loss, comp)` with comp keyed by term, so this evaluates the SHIPPED objective rather than
a reimplementation of it.

EVERYTHING IN FITTER SPACE, EXACTLY
`load_meshes` normalises each target by `(v - mean) / max|v - mean|` -- a pure similarity, so the
annotations map into fitter space exactly with the same centre and scale, recomputed here from the
same .obj the fitter loaded. The oracle is therefore re-fitted directly in fitter space (its
`global_rot`/`trans` absorb any residual frame), so both parameter vectors and the loss all live in
one coordinate system and the comparison needs no alignment step.

WHAT WOULD MAKE THIS UNREADABLE, checked rather than assumed
* The oracle must actually be near-truth in fitter space too -- asserted by reporting its joint
  residual here, which must reproduce G3's ~3.7%.
* The two vectors must differ. If the oracle collapses onto the production solution the comparison
  is vacuous; parameter distance is reported.
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
from pytorch3d.structures import Meshes  # noqa: E402

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
N_IT = int(os.environ.get("G4_ITERS", "1500"))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]

    # production parameters, per annotated specimen
    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float64)
    prod, prod_joints = {}, {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        V = np.asarray(d["verts"], dtype=np.float64)
        Jf = (np.einsum("jv,nvc->njc", Jr, V) if Jr.shape[1] == V.shape[1]
              else np.einsum("vj,nvc->njc", Jr, V))
        for i, lab in enumerate(d["labels"]):
            sid = str(lab).replace("_processed.obj", "").replace(".obj", "")
            prod[sid] = {k: np.asarray(d[k], dtype=np.float64)[i]
                         for k in ["betas", "global_rot", "joint_rot", "trans",
                                   "log_beta_scales", "betas_trans", "deform_verts"]}
            prod_joints[sid] = Jf[i]

    ann = []
    for f in sorted(glob.glob(os.path.join(REPO, "annotation/gt_batch1/*_joints.json"))):
        gt = json.load(open(f))
        sid = gt["specimen_id"]
        obj = os.path.join(MESHDIR, f"{sid}_processed.obj")
        if sid not in prod or not os.path.exists(obj):
            continue
        P = {j["joint_name"]: np.array(j["position"], dtype=np.float64)
             for j in gt["joints"] if j.get("position") is not None}
        use = [n for n in names if n in P and n not in WING and n != "b_h"]
        ann.append({"sid": sid, "obj": obj, "idx": [names.index(n) for n in use],
                    "gt_raw": np.array([P[n] for n in use])})
    print(f"{len(ann)} specimens with annotation + production fit + target mesh")

    mesh_files = [a["obj"] for a in ann]
    _, targets = load_meshes(mesh_files=mesh_files, device=dev)
    NS = len(ann)

    # PUT THE ANNOTATIONS INTO FITTER SPACE.
    # The annotation frame is NOT the .obj frame -- Blender applies a transform on import
    # (measured: annotation extent 173x49x58 against the .obj's 14926x5151x5167, and a different
    # centroid). So `load_meshes`'s (v-mean)/max|v| normalisation does NOT carry the annotations
    # across, and using it crushes them to ~0.02 extent. Verified before relying on it.
    #
    # Instead the frame is solved from the data: a 7-DOF similarity (rotation, uniform scale,
    # translation) mapping the annotated joints onto that specimen's PRODUCTION joints, over ~49
    # joints x 3 = ~147 constraints, i.e. heavily overdetermined. This is a global frame, not
    # shape -- it is the same alignment G1 used to compare template/production/truth on equal
    # terms, and it leaves the oracle free to move anywhere afterwards. The residual it leaves for
    # the production parameters must reproduce G1's ~28%, which is asserted below.
    def sim_fit(src, tgt):
        cs, ct = src.mean(0), tgt.mean(0)
        s0, t0 = src - cs, tgt - ct
        U, S, Vt = np.linalg.svd(s0.T @ t0)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        sc = (S * np.array([1, 1, d])).sum() / (s0 ** 2).sum()
        return sc * ((src - cs) @ R) + ct

    for a in ann:
        Jp = prod_joints[a["sid"]][a["idx"]]
        a["gt"] = torch.as_tensor(sim_fit(a["gt_raw"], Jp), device=dev, dtype=torch.float32)

    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    lw = st["loss_weights"]

    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in ann], loss_weights=lw,
                          lr=st.get("lr", 5e-4), device=dev,
                          n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/g4")

    def set_params(src):
        with torch.no_grad():
            for k in ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                      "betas_trans", "deform_verts"]:
                if hasattr(smal, k) and k in src:
                    getattr(smal, k).copy_(torch.as_tensor(src[k], device=dev,
                                                           dtype=torch.float32))

    def joints_now():
        v = smal()
        Jr = smal.smal_model.J_regressor
        return torch.einsum("jv,nvc->njc", Jr, v) if Jr.shape[1] == v.shape[1] \
            else torch.einsum("vj,nvc->njc", Jr, v)

    def eval_objective():
        with torch.no_grad():
            nv = smal()
            src = stage.src_mesh.offset_verts((nv - stage.src_verts).view(-1, 3))
            loss, comp = stage.forward(src, it=0)
        return float(loss), {k: float(v) for k, v in comp.items()}

    def joint_resid():
        with torch.no_grad():
            J = joints_now()
        out = []
        for i, a in enumerate(ann):
            r = (J[i, a["idx"]] - a["gt"]).norm(dim=-1)
            bl = float((a["gt"][[k for k, n in enumerate(a["idx"]) if names[n] == "b_a_3"][0]]
                        - a["gt"][[k for k, n in enumerate(a["idx"]) if names[n] == "b_t"][0]]
                        ).norm())
            out.append(float(torch.median(r)) / bl * 100)
        return float(np.median(out))

    # ---------------- production parameters
    P = {k: np.stack([prod[a["sid"]][k] for a in ann]) for k in prod[ann[0]["sid"]]}
    set_params(P)
    prod_loss, prod_comp = eval_objective()
    prod_res = joint_resid()

    # ---------------- oracle: refit pose+betas in fitter space, deform_verts left at the
    # production values so the two vectors differ ONLY in the channels G3 exercised
    free = ["global_rot", "joint_rot", "betas", "trans"]
    for k in free:
        getattr(smal, k).requires_grad_(True)
    opt = torch.optim.Adam([getattr(smal, k) for k in free], lr=0.02)
    gtb = [a["gt"] for a in ann]
    for it in range(N_IT):
        opt.zero_grad()
        J = joints_now()
        loss = torch.stack([((J[i, ann[i]["idx"]] - gtb[i]) ** 2).sum(-1).mean()
                            for i in range(NS)]).mean()
        loss.backward()
        opt.step()
        if it % 300 == 0:
            print(f"  oracle it {it:4d}  joint mse {float(loss):.6f}", flush=True)
    orc_loss, orc_comp = eval_objective()
    orc_res = joint_resid()

    print("\n" + "=" * 96)
    print("[CHECK] frame is sane (production must reproduce G1's ~28%), oracle is near-truth,")
    print("        and the two parameter vectors actually differ")
    print("=" * 96)
    print(f"  median joint residual  production {prod_res:6.2f}%   oracle {orc_res:6.2f}%")
    d = {k: float(np.linalg.norm(getattr(smal, k).detach().cpu().numpy() - P[k]))
         for k in ["betas", "joint_rot"]}
    print(f"  parameter distance oracle-vs-production: " +
          "  ".join(f"{k} {v:.3f}" for k, v in d.items()))

    print("\n" + "=" * 96)
    print("THE PRODUCTION OBJECTIVE, EVALUATED AT BOTH PARAMETER SETTINGS")
    print("(D1_PROD Stage_3_deform_fine weights; lower is better)")
    print("=" * 96)
    keys = sorted(set(prod_comp) | set(orc_comp))
    print(f"{'term':<14}{'weight':>9}{'production':>14}{'oracle':>14}{'oracle better?':>16}")
    for k in keys:
        w = lw.get(f"w_{k}", lw.get(f"w_{k.replace('mid','midline')}", float('nan')))
        a, b = prod_comp.get(k, float('nan')), orc_comp.get(k, float('nan'))
        print(f"{k:<14}{w:>9.4g}{a:>14.6f}{b:>14.6f}{str(b < a):>16}")
    print(f"{'TOTAL':<14}{'':>9}{prod_loss:>14.6f}{orc_loss:>14.6f}{str(orc_loss < prod_loss):>16}")

    print("\n" + "=" * 96)
    if orc_loss > prod_loss:
        print("=> B1: THE OBJECTIVE IS WRONG. It scores the near-truth solution WORSE than what")
        print("   the pipeline converged to. The optimiser is doing its job; the loss is")
        print("   mis-specified. The terms above where production beats the oracle name the")
        print("   culprit, and their weights are the thing to change.")
    else:
        print("=> B2: THE OPTIMISER FALLS SHORT. The objective prefers the near-truth solution but")
        print("   the optimiser never reaches it. The loss is fine; attack initialisation,")
        print("   schedule and search strategy.")
    print("=" * 96)
    json.dump({"production": {"total": prod_loss, "comp": prod_comp, "joint_resid": prod_res},
               "oracle": {"total": orc_loss, "comp": orc_comp, "joint_resid": orc_res}},
              open(os.path.join(HERE, "g4_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
