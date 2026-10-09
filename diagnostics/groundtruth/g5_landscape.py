"""G5 -- map the landscape between the production solution and the oracle.

G4 asks which of two points the objective prefers. That is a two-point comparison and it cannot
distinguish three different situations that have completely different remedies:

  MISALIGNED   anatomical error falls monotonically toward the oracle while the objective RISES.
               The loss is pointing the wrong way; no optimiser reaches a solution its objective
               dislikes. Fix the objective.
  BARRIER      both endpoints are good but the objective has a ridge between them. A good basin
               exists and is separated from where the optimiser starts. Fix initialisation,
               continuation, or the schedule.
  REACHABLE    the objective falls monotonically toward the oracle with no barrier. The good
               solution is downhill all the way and the optimiser still did not go there --
               which would point at the trajectory, the parameter-block freezing, or the
               learning-rate schedule rather than the landscape.

Only a path between the two points separates these, which is why this exists.

METHOD
Interpolate linearly in parameter space, production -> oracle, and at each step evaluate the
SHIPPED objective (`MoonshotStage.forward`, per-term) and the anatomical joint residual against the
human annotations. Both curves on one axis.

CAVEAT, stated rather than hidden: linear interpolation of axis-angle rotations is not the geodesic
path on SO(3), so the intermediate poses are not the shortest rotational route. For a landscape
probe this is acceptable and conventional -- the endpoints are exact, and a barrier detected on a
slightly non-geodesic path is still a barrier -- but the intermediate values are a section through
the landscape, not the optimiser's actual trajectory.
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
from pytorch3d.io import load_obj  # noqa: E402

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
N_IT = int(os.environ.get("G5_ITERS", "1500"))
STEPS = int(os.environ.get("G5_STEPS", "21"))
FREE = ["global_rot", "joint_rot", "betas", "trans"]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]

    prod = {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        for i, lab in enumerate(d["labels"]):
            sid = str(lab).replace("_processed.obj", "").replace(".obj", "")
            prod[sid] = {k: np.asarray(d[k], dtype=np.float64)[i] for k in
                         ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                          "betas_trans", "deform_verts"]}

    ann = []
    for f in sorted(glob.glob(os.path.join(REPO, "annotation/gt_batch1/*_joints.json"))):
        gt = json.load(open(f)); sid = gt["specimen_id"]
        obj = os.path.join(MESHDIR, f"{sid}_processed.obj")
        if sid not in prod or not os.path.exists(obj):
            continue
        P = {j["joint_name"]: np.array(j["position"], dtype=np.float64)
             for j in gt["joints"] if j.get("position") is not None}
        use = [n for n in names if n in P and n not in WING and n != "b_h"]
        v, _, _ = load_obj(obj, load_textures=False)
        c = v.mean(0); s = float((v - c).abs().max())
        ann.append({"sid": sid, "obj": obj, "idx": [names.index(n) for n in use],
                    "gt": torch.as_tensor((np.array([P[n] for n in use]) - c.numpy()) / s,
                                          device=dev, dtype=torch.float32),
                    "bt": use.index("b_t"), "ba3": use.index("b_a_3")})
    NS = len(ann)
    print(f"{NS} specimens")

    _, targets = load_meshes(mesh_files=[a["obj"] for a in ann], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]; lw = st["loss_weights"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in ann], loss_weights=lw,
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/g5")

    P = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in ann]), device=dev,
                            dtype=torch.float32) for k in prod[ann[0]["sid"]]}

    def setp(d):
        with torch.no_grad():
            for k, v in d.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def joints():
        v = smal()
        Jr = smal.smal_model.J_regressor
        return (torch.einsum("jv,nvc->njc", Jr, v) if Jr.shape[1] == v.shape[1]
                else torch.einsum("vj,nvc->njc", Jr, v))

    def anat():
        with torch.no_grad():
            J = joints()
        out = []
        for i, a in enumerate(ann):
            r = (J[i, a["idx"]] - a["gt"]).norm(dim=-1)
            bl = float((a["gt"][a["ba3"]] - a["gt"][a["bt"]]).norm())
            out.append(float(torch.median(r)) / bl * 100)
        return float(np.median(out))

    def objective():
        with torch.no_grad():
            nv = smal()
            src = stage.src_mesh.offset_verts((nv - stage.src_verts).view(-1, 3))
            loss, comp = stage.forward(src, it=0)
        return float(loss), {k: float(v) for k, v in comp.items()}

    # ---- oracle, refitted in fitter space (same protocol as G4)
    setp(P)
    for k in FREE:
        getattr(smal, k).requires_grad_(True)
    opt = torch.optim.Adam([getattr(smal, k) for k in FREE], lr=0.02)
    for it in range(N_IT):
        opt.zero_grad()
        J = joints()
        loss = torch.stack([((J[i, ann[i]["idx"]] - ann[i]["gt"]) ** 2).sum(-1).mean()
                            for i in range(NS)]).mean()
        loss.backward(); opt.step()
    O = {k: getattr(smal, k).detach().clone() for k in FREE}
    for k in P:
        if k not in O:
            O[k] = P[k].clone()          # deform_verts etc held at production values
    print(f"oracle fitted: anatomical residual {anat():.2f}%")

    print("\n" + "=" * 104)
    print("G5 -- section through the landscape, production (t=0) -> oracle (t=1)")
    print("=" * 104)
    terms = None
    rows = []
    for t in np.linspace(0.0, 1.0, STEPS):
        setp({k: (1 - t) * P[k] + t * O[k] for k in P})
        a = anat()
        tot, comp = objective()
        terms = terms or sorted(comp)
        rows.append({"t": float(t), "anat": a, "total": tot, **comp})
        print(f"  t={t:4.2f}  anat {a:6.2f}%   objective {tot:10.6f}", flush=True)

    an = np.array([r["anat"] for r in rows]); ob = np.array([r["total"] for r in rows])
    print("\n" + "=" * 104)
    print(f"anatomical error : {an[0]:.2f}% -> {an[-1]:.2f}%   (monotone down: "
          f"{bool(np.all(np.diff(an) <= 1e-9))})")
    print(f"objective        : {ob[0]:.6f} -> {ob[-1]:.6f}  (monotone down: "
          f"{bool(np.all(np.diff(ob) <= 1e-12))})")
    peak = int(np.argmax(ob))
    barrier = ob[peak] > max(ob[0], ob[-1]) * 1.001
    print(f"objective peak at t={rows[peak]['t']:.2f}, value {ob[peak]:.6f} -- "
          f"barrier between the endpoints: {barrier}")
    print()
    if ob[-1] > ob[0]:
        print("=> MISALIGNED. The objective RISES toward the anatomically correct solution.")
        print("   No optimiser will go somewhere its loss dislikes. Fix the objective; the")
        print("   per-term columns below identify which term is responsible.")
    elif barrier:
        print("=> BARRIER. Both endpoints score well but a ridge separates them: a good basin")
        print("   exists and the optimiser starts on the wrong side. Fix initialisation /")
        print("   continuation / schedule.")
    else:
        print("=> REACHABLE. The objective descends all the way to the oracle with no barrier,")
        print("   yet the optimiser did not get there -- so the fault is the trajectory itself")
        print("   (parameter-block freezing, learning rates, stage schedule), not the landscape.")
    print("=" * 104)

    sel = rows[::max(1, STEPS // 5)]
    hdr = "".join("%12s" % ("t=%.2f" % r["t"]) for r in sel)
    print("\n%-14s%s" % ("term", hdr))
    for k in terms:
        print("%-14s%s" % (k, "".join("%12.5f" % r[k] for r in sel)))

    json.dump(rows, open(os.path.join(HERE, "g5_results.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'g5_results.json')}")


if __name__ == "__main__":
    main()
