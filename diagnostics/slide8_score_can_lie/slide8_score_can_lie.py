"""Slide 8, "THE SCORE CAN LIE" -- single-specimen reproduction of R9/R3 for one gt_expert
specimen, with saved meshes/params + rendered stills/animation for a presentation slide.

Methodology copied exactly from diagnostics/groundtruth/r9_objective_audit.py and
r3_correct_parts.py (same production checkpoints, same D1_PROD.yaml Stage_3_deform_fine
objective, same anatomically-CORRECT landmark supervision), restricted to ONE specimen so a
concrete mesh+params pair can be saved to disk for rendering (R9/R3 only ever kept aggregate
per-arm numbers in a 12-specimen batch; per-specimen meshes were never written out).

Specimen chosen: Odontomachus_bauri_CASENT0878072 -- a trap-jaw ant with unusually elongated,
articulated mandibles, one of the 12 R9/R3 gt_expert specimens (all 12 gt_expert specimens were
already used by R9; there is no 13th annotated specimen to be "fresh" with -- see RESULTS md).
This morphology is exactly the kind R9 flagged as most likely to show the effect ("named parts
drift furthest on ... long mandibles"), which makes any anatomical mismatch visually legible in
a still frame, not just in the numbers.
"""
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
sys.path.insert(0, os.path.join(REPO, "diagnostics/groundtruth"))

import glob  # noqa: E402
import yaml  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
import trait_extract as TX  # noqa: E402

MESHDIR = "/hpcwork/nao48500/worker_alt_data"
LAM = 1.0
N_IT = int(os.environ.get("SLIDE8_ITERS", "800"))
FREE_P = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
NEWLM = json.load(open(os.path.join(REPO, "annotation/landmark_indices_recalibrated.json")))
SID = os.environ.get("SLIDE8_SID", "Odontomachus_bauri_CASENT0878072")
OUT = HERE

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
    print(f"device={dev}")
    M = TX.load_model(); dom = M["dominant"]; jn = M["jnames"]

    prod = {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        for i, lab in enumerate(d["labels"]):
            sid = str(lab).replace("_processed.obj", "")
            prod[sid] = {k: np.asarray(d[k], np.float32)[i] for k in
                         ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                          "betas_trans", "deform_verts"]}
    assert SID in prod, f"{SID} not found in production checkpoints; available={list(prod)[:5]}..."
    print(f"specimen={SID}  found in production checkpoint: yes")

    ltrait = os.path.join(REPO, f"annotation/landmarks/{SID}_traits.json")
    assert os.path.exists(ltrait), ltrait
    d = json.load(open(ltrait))
    assert d["specimen_id"] == SID
    objf = os.path.join(MESHDIR, f"{SID}_processed.obj")
    assert os.path.exists(objf), objf
    o = readobj(objf)
    c, s = o.mean(0), np.abs(o - o.mean(0)).max()
    print(f"target mesh: {objf}  n_verts={len(o)}  centroid={c}  scale={s:.4f}")
    L = {k: (np.array(v["original"], float) - c) / s for k, v in d["landmarks"].items()}
    ALL = list(L)
    WL = float(np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"]))
    print(f"landmarks available: {ALL}")
    print(f"Weber's length (normalised) = {WL:.4f}")

    REG_CORRECT = {}
    for k in ALL:
        idx = np.where(np.isin(dom, [jn.index(p) for p in CORRECT[k]]))[0]
        REG_CORRECT[k] = torch.as_tensor(idx, device=dev)
        print(f"  landmark {k:22s} -> CORRECT part {'+'.join(CORRECT[k]):14s} ({len(idx)} verts)")

    _, targets = load_meshes(mesh_files=[objf], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    lw = st["loss_weights"]
    smal = SMAL3DFitter(batch_size=1, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[SID], loss_weights=lw, lr=st.get("lr", 5e-4), device=dev,
                          n_sample=st.get("n_sample", 6000), edge_mode=st.get("edge_mode", "shrink"),
                          out_dir="/tmp/slide8")
    P0 = {k: torch.as_tensor(v[None], device=dev) for k, v in prod[SID].items()}
    LM = {k: torch.as_tensor(L[k][None], device=dev, dtype=torch.float32) for k in ALL}

    def setp():
        with torch.no_grad():
            for k, v in P0.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def loss(v, reg):
        tot, n = 0.0, 0
        for k in ALL:
            p = LM[k]
            sub = v if reg is None else v[:, reg[k]]
            tot = tot + (torch.cdist(p.unsqueeze(1), sub).squeeze(1).min(-1).values ** 2).mean()
            n += 1
        return tot / max(n, 1)

    def joint_err(v):
        """median % of WL, landmark to its anatomically-CORRECT part."""
        errs = []
        for k in ALL:
            p = LM[k]
            dc = torch.cdist(p.unsqueeze(1), v[:, REG_CORRECT[k]]).squeeze(1).min(-1).values
            errs.append(float(dc.item()) / WL * 100)
        return float(np.median(errs)), dict(zip(ALL, errs))

    def save_state(tag, params_dict, verts, comp, tot_val, cerr, cerrs):
        np.savez(os.path.join(OUT, f"fit_{tag}.npz"),
                 **{k: v.detach().cpu().numpy() for k, v in params_dict.items()})
        faces = smal.faces[0].detach().cpu().numpy() if hasattr(smal, "faces") else stage.src_mesh.faces_padded()[0].cpu().numpy()
        vv = verts[0].detach().cpu().numpy()
        with open(os.path.join(OUT, f"mesh_{tag}.obj"), "w") as f:
            for row in vv:
                f.write(f"v {row[0]} {row[1]} {row[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        return dict(tag=tag, total=float(tot_val), chamfer_raw=float(comp.get("chamfer", float("nan"))),
                    joint_err_median_pct_WL=cerr, joint_err_per_landmark_pct_WL=cerrs)

    # ---- arm (a): production fit, as-is ----
    setp()
    with torch.no_grad():
        v = smal()
        src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
        tot, comp = stage.forward(src, it=0)
        cerr, cerrs = joint_err(v)
    prod_summary = save_state("production", {k: getattr(smal, k) for k in FREE_P}, v, comp, tot, cerr, cerrs)
    print(f"\nPRODUCTION  total={float(tot):.6f}  chamfer={float(comp.get('chamfer', float('nan'))):.6f}  "
          f"joint_err_median={cerr:.2f}% WL")

    # ---- arm (b): anatomically-CORRECT fit (R3/R9 CORRECT arm, LAM=1.0, 800 Adam its) ----
    setp()
    for nm in FREE_P:
        getattr(smal, nm).requires_grad_(True)
    opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE_P], lr=0.005)
    for it in range(N_IT):
        opt.zero_grad()
        v = smal()
        src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
        t_, _ = stage.forward(src, it=0)
        (t_ + LAM * loss(v, REG_CORRECT)).backward()
        opt.step()
        if it % 200 == 0:
            print(f"  CORRECT it={it} tot={float(t_):.6f}", flush=True)
    with torch.no_grad():
        v = smal()
        src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
        tot, comp = stage.forward(src, it=0)
        cerr, cerrs = joint_err(v)
    corr_summary = save_state("corrected", {k: getattr(smal, k) for k in FREE_P}, v, comp, tot, cerr, cerrs)
    print(f"\nCORRECTED   total={float(tot):.6f}  chamfer={float(comp.get('chamfer', float('nan'))):.6f}  "
          f"joint_err_median={cerr:.2f}% WL")

    # save target mesh too, for overlay rendering
    tv = targets.verts_padded()[0].cpu().numpy()
    tf = targets.faces_padded()[0].cpu().numpy()
    with open(os.path.join(OUT, "mesh_target.obj"), "w") as f:
        for row in tv:
            f.write(f"v {row[0]} {row[1]} {row[2]}\n")
        for face in tf:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    result = dict(specimen=SID, obj_file=objf, weber_length_norm=WL,
                  production=prod_summary, corrected=corr_summary,
                  chamfer_ratio_corrected_over_production=(
                      corr_summary["chamfer_raw"] / prod_summary["chamfer_raw"]),
                  total_ratio_corrected_over_production=(
                      corr_summary["total"] / prod_summary["total"]),
                  n_iters=N_IT, lam=LAM, free_params=FREE_P)
    json.dump(result, open(os.path.join(OUT, "slide8_numeric_results.json"), "w"), indent=2)
    print("\n" + "=" * 90)
    print(f"chamfer ratio (corrected/production) = {result['chamfer_ratio_corrected_over_production']:.3f}x")
    print(f"total objective ratio (corrected/production) = {result['total_ratio_corrected_over_production']:.3f}x")
    print(f"joint error: production={prod_summary['joint_err_median_pct_WL']:.2f}%  "
          f"corrected={corr_summary['joint_err_median_pct_WL']:.2f}%")
    print("wrote slide8_numeric_results.json")
    print("=" * 90)


if __name__ == "__main__":
    main()
