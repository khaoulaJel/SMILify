"""Reference analyses O0/O1/O2 (PREREGISTRATION.md §6.1). NOT deployable: they use the GT at fit time.

Starting point: arm A (D1_PROD) seed-0 final parameters. Objective: the D1_PROD Stage_3 objective
(same MoonshotStage.forward), plus lambda * mean squared distance between FK joints and the
fit-independent GT joints over a SUPERVISED subset. 800 iterations, Adam lr 0.005 on
global_rot, joint_rot, betas, trans, deform_verts (V9 arm-C protocol; lambda = 0.03 from V5b/V9).

  O0_control         lambda = 0     re-optimisation alone must not move joint error (voiding control)
  O1_all             all evaluated joints supervised           -> reachability ceiling (circular)
  O2_heldout_<r>     all regions except r supervised; scored ONLY on r -> does a partial skeleton transfer?

Writes a Stage_3_deform_fine.npz per arm under <runs>/<arm>_s0 in the standard format, so
score.py scores it with exactly the same code as every strategy arm.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(BENCH, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import torch  # noqa: E402
import yaml  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
import model_joints as MJ  # noqa: E402
from score import EXCL, region  # noqa: E402

PARAMS = ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales", "betas_trans", "deform_verts"]
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
REGIONS = ["body_axis", "coxa", "leg_proximal", "leg_distal", "mandible", "antenna"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/hpcwork/nao48500/jab_runs")
    ap.add_argument("--lam", type=float, default=0.03)
    ap.add_argument("--iters", type=int, default=800)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    G = json.load(open(os.path.join(BENCH, "data/gt_joints_fitframe.json")))
    names = MJ.joint_names()
    src = os.path.join(a.runs, "A_prod_s0", "Stage_3_deform_fine.npz")
    z = np.load(src, allow_pickle=True)
    labels = [MJ.clean_label(l) for l in z["labels"]]
    assert set(labels) == set(G), "reference fit does not cover the benchmark specimens"
    MJ.load_fit(src, wanted=set(G), device=dev)          # reproduction guard on the starting point

    mesh_files = [os.path.join("/hpcwork/nao48500/jab_stage", f"{s}_processed.obj") for s in labels]
    _, targets = load_meshes(mesh_files=mesh_files, device=dev)
    st = yaml.safe_load(open(os.path.join(BENCH, "cfg/A_prod.yaml")))["stages"]["Stage_3_deform_fine"]
    NS = len(labels)
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=labels, loss_weights=st["loss_weights"], lr=st["lr"], device=dev,
                          n_sample=st["n_sample"], edge_mode=st["edge_mode"], out_dir="/tmp/jab_oracle")

    gt = torch.zeros(NS, len(names), 3, device=dev)
    placed = torch.zeros(NS, len(names), dtype=torch.bool, device=dev)
    for i, s in enumerate(labels):
        for n, r in G[s]["joints"].items():
            if n not in EXCL:
                gt[i, names.index(n)] = torch.as_tensor(r["fit"], device=dev, dtype=torch.float32)
                placed[i, names.index(n)] = True
    reg_of = [region(n) if n not in EXCL else None for n in names]

    def reset():
        with torch.no_grad():
            for k in PARAMS:
                getattr(smal, k).copy_(torch.as_tensor(np.asarray(z[k], np.float32), device=dev))

    arms = [("O0_control", 0.0, None), ("O1_all", a.lam, None)] + \
           [(f"O2_heldout_{r}", a.lam, r) for r in REGIONS]
    log = {}
    for arm, lam, held in arms:
        reset()
        sup = placed.clone()
        if held is not None:
            for j, rg in enumerate(reg_of):
                if rg == held:
                    sup[:, j] = False
        opt = torch.optim.Adam([getattr(smal, k) for k in FREE], lr=0.005)
        for k in FREE:
            getattr(smal, k).requires_grad_(True)
        for it in range(a.iters):
            opt.zero_grad()
            v = smal()
            J = smal.smal_model.J_transformed + smal.trans.unsqueeze(1)     # FK pivots (§5.1)
            mesh = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            loss, _ = stage.forward(mesh, it=0)
            if lam > 0:
                d2 = ((J - gt) ** 2).sum(-1)
                loss = loss + lam * (d2 * sup).sum() / sup.sum()
            loss.backward()
            opt.step()
        out = os.path.join(a.runs, f"{arm}_s0")
        os.makedirs(out, exist_ok=True)
        with torch.no_grad():
            np.savez(os.path.join(out, "Stage_3_deform_fine.npz"),
                     **{k: getattr(smal, k).detach().cpu().numpy() for k in PARAMS},
                     verts=smal().detach().cpu().numpy(), faces=z["faces"], labels=z["labels"])
        log[arm] = dict(lam=lam, heldout_region=held, n_supervised=int(sup.sum()), iters=a.iters)
        print(f"[oracle] {arm} done ({int(sup.sum())} supervised joint instances)", flush=True)
    json.dump(log, open(os.path.join(BENCH, "data/oracle_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
