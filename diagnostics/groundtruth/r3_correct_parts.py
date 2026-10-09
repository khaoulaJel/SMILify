"""R3 -- can the fitter put its NAMED anatomical parts where the anatomy actually is?

R2 measured that the model's own named parts sit 27-81% of Weber's length from the corresponding
human landmark on the anterior structures: the fitted `ma_r` is 50% of a body length from the real
right mandible tip, `an_1_r` is 74% from the real scape apex. The registration places *a* surface
near the anatomy while placing the *named* structures far from it.

R3 asks whether that is fixable by supervision. Each landmark is constrained to its ANATOMICALLY
CORRECT part -- assigned from anatomy, not from a consensus vertex -- so the loss can only be
satisfied by moving the right structure:

    L = mean_p  min over v in CORRECT_PART(p)  || p - v ||^2

THREE ARMS, same targets, differing only in which vertices may satisfy them:
  CORRECT   the anatomically correct part          -- can the named structure get there?
  CONSENSUS R2's assignment (10 of 12 = the head)  -- the loose comparison
  FREE      any vertex (= V12's S12)               -- the upper bound, 0.1% at 1.10x

If CORRECT reaches the landmarks at a comparable chamfer, the parts are merely mis-positioned and
supervision relocates them. If it cannot -- high residual, or a large chamfer penalty for trying --
then the model's part layout cannot be reconciled with the real anatomy by fitting alone, which is a
statement about the template rather than the objective.
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
sys.path.insert(0, HERE)

import yaml  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
import trait_extract as TX  # noqa: E402

MESHDIR = "/hpcwork/nao48500/worker_alt_data"
LAM = 1.0
N_IT = int(os.environ.get("R3_ITERS", "800"))
FREE_P = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
NEWLM = json.load(open(os.path.join(REPO, "annotation/landmark_indices_recalibrated.json")))

# ANATOMICALLY CORRECT part per landmark, assigned from anatomy rather than from any fitted vertex.
CORRECT = {"clypeal_ant_mid": ["b_h"], "cephalic_post_mid": ["b_h"],
           "head_width_r": ["b_h"], "head_width_l": ["b_h"],
           "mandibular_apex_r": ["ma_r"], "mandibular_apex_l": ["ma_l"],
           "antennal_insertion_r": ["an_1_r"], "antennal_insertion_l": ["an_1_l"],
           "scape_apex_r": ["an_1_r", "an_2_r"], "scape_apex_l": ["an_1_l", "an_2_l"],
           "wl_anterior_r": ["b_t"], "wl_posterior_r": ["b_t"]}


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M = TX.load_model(); dom = M["dominant"]; jn = M["jnames"]
    prod = {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        for i, lab in enumerate(d["labels"]):
            prod[str(lab).replace("_processed.obj", "")] = {
                k: np.asarray(d[k], np.float32)[i] for k in
                ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                 "betas_trans", "deform_verts"]}
    spec = []
    for p in sorted(glob.glob(os.path.join(REPO, "annotation/landmarks/*_traits.json"))):
        d = json.load(open(p)); sid = d["specimen_id"]
        if sid not in prod:
            continue
        o = readobj(os.path.join(MESHDIR, f"{sid}_processed.obj"))
        c, s = o.mean(0), np.abs(o - o.mean(0)).max()
        L = {k: (np.array(v["original"], float) - c) / s for k, v in d["landmarks"].items()}
        spec.append(dict(sid=sid, obj=os.path.join(MESHDIR, f"{sid}_processed.obj"), L=L,
                         WL=float(np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"]))))
    NS = len(spec); ALL = list(spec[0]["L"])

    REG = {"CORRECT": {}, "CONSENSUS": {}, "FREE": {k: None for k in ALL}}
    for k in ALL:
        idx = np.where(np.isin(dom, [jn.index(p) for p in CORRECT[k]]))[0]
        REG["CORRECT"][k] = torch.as_tensor(idx, device=dev)
        j = int(dom[NEWLM[k]["vertex"]])
        REG["CONSENSUS"][k] = torch.as_tensor(np.where(dom == j)[0], device=dev)
    print(f"{NS} specimens\n\nlandmark -> anatomically correct part:")
    for k in ALL:
        print(f"   {k:22s} {'+'.join(CORRECT[k]):14s} {len(REG['CORRECT'][k]):>5} verts")

    _, targets = load_meshes(mesh_files=[a["obj"] for a in spec], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in spec], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/r3")
    P0 = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in spec]), device=dev)
          for k in prod[spec[0]["sid"]]}
    LM = {k: torch.as_tensor(np.stack([a["L"][k] if k in a["L"] else np.full(3, np.nan)
                                       for a in spec]), device=dev, dtype=torch.float32)
          for k in ALL}
    WL = torch.as_tensor([a["WL"] for a in spec], device=dev)

    def setp():
        with torch.no_grad():
            for k, v in P0.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def loss(v, reg):
        tot, n = 0.0, 0
        for k in ALL:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            sub = v[ok] if reg[k] is None else v[ok][:, reg[k]]
            tot = tot + (torch.cdist(p[ok].unsqueeze(1), sub).squeeze(1)
                         .min(-1).values ** 2).mean(); n += 1
        return tot / max(n, 1)

    def errs(v):
        """(error to the CORRECT part, error to the nearest surface) per landmark, median."""
        a, b = [], []
        for k in ALL:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            dc = torch.cdist(p[ok].unsqueeze(1),
                             v[ok][:, REG["CORRECT"][k]]).squeeze(1).min(-1).values
            dn = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1).min(-1).values
            a += list((dc / WL[ok] * 100).cpu().numpy())
            b += list((dn / WL[ok] * 100).cpu().numpy())
        return float(np.median(a)), float(np.median(b))

    setp()
    with torch.no_grad():
        v = smal()
        src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
        _, comp = stage.forward(src, it=0)
        bc, bn = errs(v); base_c = float(comp.get("chamfer", np.nan))
    print("\n" + "=" * 86)
    print("R3 -- can the NAMED anatomical parts be moved onto the real anatomy?")
    print("=" * 86)
    print(f"{'arm':14s}{'err to CORRECT part':>21}{'err to any surface':>20}{'chamfer':>10}")
    print(f"{'production':14s}{bc:>20.1f}%{bn:>19.1f}%{1.0:>9.2f}x")
    rows = [dict(arm="production", correct=bc, nearest=bn, chamfer=1.0)]
    for name in ("CORRECT", "CONSENSUS", "FREE"):
        setp()
        for nm in FREE_P:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE_P], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            (tot + LAM * loss(v, REG[name])).backward()
            opt.step()
        with torch.no_grad():
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            _, comp = stage.forward(src, it=0)
            c, n_ = errs(v); ch = float(comp.get("chamfer", np.nan)) / base_c
        rows.append(dict(arm=name, correct=c, nearest=n_, chamfer=ch))
        print(f"{name:14s}{c:>20.1f}%{n_:>19.1f}%{ch:>9.2f}x", flush=True)

    R = {r["arm"]: r for r in rows}
    print(f"\n  production: the named parts sit {R['production']['correct']:.1f}% from the anatomy "
          f"while some surface is {R['production']['nearest']:.1f}% away")
    print(f"  supervised on the CORRECT part: {R['CORRECT']['correct']:.1f}% at "
          f"{R['CORRECT']['chamfer']:.2f}x chamfer")
    ok = R["CORRECT"]["correct"] < 2.0
    print("\n  => THE PARTS ARE MERELY MIS-POSITIONED. Supervision relocates the named structures"
          if ok else
          "\n  => THE NAMED PARTS CANNOT REACH THE ANATOMY. This is a template/skinning limit,")
    print("     onto the real anatomy, so this is an objective problem, not a template one."
          if ok else "     not something the objective can fix.")
    json.dump(rows, open(os.path.join(HERE, "r3_results.json"), "w"), indent=2)
    print("=" * 86)


if __name__ == "__main__":
    main()
