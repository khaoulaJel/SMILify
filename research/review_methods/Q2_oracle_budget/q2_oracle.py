"""Q2 oracle arms: from B1's final fit, re-optimise the PRODUCTION Stage_3 objective with one ground-truth
quantity restored (PREREGISTRATION.md; DEVIATIONS D6).

Production machinery: an OracleStage subclass of MoonshotStage built from the exact A_prod.yaml
Stage_3 options (scheme all, lr 0.0005, joint_rot lr 0.002, shrink edges, its weights), nits = 800,
with the B1 correspondence term (w_cse_corr = 1.0, as optimise_moonshot adds it). The ONLY addition
is the oracle term of each arm.

GT in the fitter's frame (D6): the root log-scale is not a global scale, so GT parameters cannot be
copied. Step 0 registers the model to each normalised GT mesh with known vertex identity (all
parameters except deform_verts, deform fixed at 0), initialised at the GT parameters, and reports the
residual. O-pose / O-shape use those frame-matched parameters. GT joints in the fitter frame are the
GT FK joints mapped by the same similarity as the mesh (exact; independent of step 0).

Arms (each starts from B1's reload-verified final parameters):
  O0       nothing added (voiding control: must not move error)
  O-part   correspondence term uses the oracle-part targets
  O-corr   correspondence term uses dense true targets
  O-joints + lam * mean sq. FK-joint error vs GT (lam = 0.03, JAB O1 value, not re-tuned)
  O-pose   joint_rot := frame-matched GT before re-optimisation
  O-shape  betas, log_beta_scales, betas_trans := frame-matched GT before re-optimisation
  O-all    O-corr + O-joints + O-pose + O-shape
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "joint_alignment_benchmark", "tools"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import model_joints as MJ  # noqa: E402
from score import EXCL  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

O = "/hpcwork/nao48500/review_methods/Q2"
YAML = os.path.join(REPO, "diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml")
PARAMS = ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales", "betas_trans", "deform_verts"]
SHAPE = ["betas", "log_beta_scales", "betas_trans"]
LAM = 0.03
ARMS = ["O0", "O-part", "O-corr", "O-joints", "O-pose", "O-shape", "O-all"]


class OracleStage(MoonshotStage):
    def __init__(self, *a, joints_gt=None, joints_use=None, lam=0.0, **k):
        super().__init__(*a, **k)
        self.joints_gt, self.joints_use, self.lam = joints_gt, joints_use, lam

    def forward(self, src_mesh, it=0):
        loss, comp = super().forward(src_mesh, it)
        if self.lam > 0:
            f = self.smal_3d_fitter
            J = f.smal_model.J_transformed + f.trans.unsqueeze(1)       # FK pivots of THIS step's forward
            d2 = ((J[:, self.joints_use] - self.joints_gt[:, self.joints_use]) ** 2).sum(-1).mean()
            comp["oracle_joints"] = d2
            loss = loss + self.lam * d2
        return loss, comp


def frame_register(gt, Vn, dev, iters=3000):
    """Model parameters reproducing the normalised GT meshes (identity known). Returns params, residual."""
    n = len(Vn)
    f = SMAL3DFitter(batch_size=n, device=dev)
    with torch.no_grad():
        for k in ("betas", "joint_rot", "log_beta_scales", "betas_trans"):
            getattr(f, k).copy_(torch.as_tensor(gt[k], dtype=torch.float32, device=dev))
        f.global_rot.zero_(); f.deform_verts.zero_()
        f.trans.copy_(torch.as_tensor(Vn.mean(1) - gt["verts"].mean(1), dtype=torch.float32, device=dev))
    P = [getattr(f, k) for k in ("global_rot", "trans", "joint_rot", "betas", "log_beta_scales", "betas_trans")]
    opt = torch.optim.Adam(P, lr=0.003)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    tgt = torch.as_tensor(Vn, dtype=torch.float32, device=dev)
    for _ in range(iters):
        opt.zero_grad()
        ((f() - tgt) ** 2).sum(-1).mean().backward()
        opt.step(); sch.step()
    with torch.no_grad():
        r = (f() - tgt).norm(dim=-1)
    out = {k: getattr(f, k).detach().clone() for k in PARAMS}
    return out, dict(median_vertex_err=float(r.median()), p99=float(r.quantile(0.99)), max=float(r.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--nits", type=int, default=800)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    base = os.path.join(O, a.regime)
    mdir = os.path.join(base, "meshes")
    gt = np.load(os.path.join(mdir, "ground_truth.npz"), allow_pickle=True)
    src = os.path.join(base, "runs", f"B1_s{a.seed}", "Stage_3_deform_fine.npz")
    z = np.load(src, allow_pickle=True)
    labels = [MJ.clean_label(x) for x in z["labels"]]
    MJ.load_fit(src, wanted=set(labels), device=dev)            # reproduction guard on the start point
    gi = [list(gt["names"]).index(s) for s in labels]
    mesh_files = [os.path.join(mdir, f"{s}.obj") for s in labels]
    _, targets = load_meshes(mesh_files=mesh_files, device=dev)
    Vn = np.stack([v.cpu().numpy() for v in targets.verts_list()])
    V = gt["verts"][gi].astype(np.float64)
    c = V.mean(1, keepdims=True); sc = np.abs(V - c).max((1, 2), keepdims=True)
    assert np.abs((V - c) / sc - Vn).max() < 1e-4, "GT mesh normalisation differs from load_meshes"
    Jgt = torch.as_tensor((gt["joints"][gi] - c) / sc, dtype=torch.float32, device=dev)
    names = MJ.joint_names()
    use = torch.as_tensor([k for k, n in enumerate(names) if n not in EXCL], device=dev)

    reg, resid = frame_register({k: gt[k][gi] for k in ("betas", "joint_rot", "log_beta_scales", "betas_trans", "verts")}, Vn, dev)
    print(f"[q2] {a.regime} s{a.seed} frame registration residual: {resid}", flush=True)

    tg = {}
    for k in ("deployed", "opart", "ocorr"):
        t = np.load(os.path.join(base, f"targets_{k}.npz"))
        ti = [list(t["names"]).index(s) for s in labels]
        tg[k] = (torch.as_tensor(t["verts"][ti], dtype=torch.float32, device=dev),
                 torch.as_tensor(t["mask"][ti], dtype=torch.bool, device=dev))
    st = dict(yaml.safe_load(open(YAML))["stages"]["Stage_3_deform_fine"])
    st["nits"] = a.nits
    st["loss_weights"] = dict(st["loss_weights"], w_cse_corr=1.0)

    smal = SMAL3DFitter(batch_size=len(labels), device=dev)
    log = dict(frame_registration=resid, start=src, stage_options=st)
    for arm in ARMS:
        with torch.no_grad():
            for k in PARAMS:
                getattr(smal, k).copy_(torch.as_tensor(np.asarray(z[k], np.float32), device=dev))
            if arm in ("O-pose", "O-all"):
                smal.joint_rot.copy_(reg["joint_rot"])
            if arm in ("O-shape", "O-all"):
                for k in SHAPE:
                    getattr(smal, k).copy_(reg[k])
        tkey = {"O-part": "opart", "O-corr": "ocorr", "O-all": "ocorr"}.get(arm, "deployed")
        lam = LAM if arm in ("O-joints", "O-all") else 0.0
        out = os.path.join(base, "runs", f"{arm}_s{a.seed}")
        os.makedirs(out, exist_ok=True)
        stage = OracleStage(name="Stage_3_deform_fine", target_meshes=targets, smal_3d_fitter=smal,
                            out_dir=out, device=dev, mesh_names=labels, cse_corr_verts=tg[tkey][0],
                            cse_corr_mask=tg[tkey][1], joints_gt=Jgt, joints_use=use, lam=lam, **st)
        stage.run()
        stage.save_npz(labels=z["labels"])
        log[arm] = dict(targets=tkey, lam=lam, set_pose=arm in ("O-pose", "O-all"),
                        set_shape=arm in ("O-shape", "O-all"))
        print(f"[q2] {a.regime} s{a.seed} {arm} done", flush=True)
    json.dump(log, open(os.path.join(base, "runs", f"oracle_log_s{a.seed}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
