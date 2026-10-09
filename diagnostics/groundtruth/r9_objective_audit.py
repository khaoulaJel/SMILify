"""R9 -- does the production objective prefer the anatomically CORRECT solution, or production's?

Extends r3_correct_parts.py's exact machinery (same 12 specimens, same production-parameter
source, same optimization) to capture and report the FULL per-term `comp` dict at each arm's
converged state, not just chamfer. See PREREGISTRATION_R9_objective_audit.md for the decision
rule, fixed before this ran.
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
N_IT = int(os.environ.get("R9_ITERS", "800"))
FREE_P = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
NEWLM = json.load(open(os.path.join(REPO, "annotation/landmark_indices_recalibrated.json")))

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

    _, targets = load_meshes(mesh_files=[a["obj"] for a in spec], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    lw = st["loss_weights"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in spec], loss_weights=lw,
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/r9")
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

    ALL_TERMS = ["chamfer", "edge", "normal", "laplacian", "offset", "sym", "mid",
                 "scale", "allo", "limit", "beta_prior", "trans", "jres", "dsym",
                 "cse_corr", "nrm_align"]

    def full_report(comp, tot_val):
        row = {"total": tot_val}
        for t in ALL_TERMS:
            raw = float(comp[t]) if t in comp else None
            w = lw.get(f"w_{t}", 0.0) if t != "mid" else lw.get("w_midline", 0.0)
            if t == "allo":
                w = lw.get("w_allo", 0.0)
            row[f"{t}_raw"] = raw
            row[f"{t}_weighted"] = (raw * w) if raw is not None else None
        return row

    setp()
    with torch.no_grad():
        v = smal()
        src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
        tot, comp = stage.forward(src, it=0)
        bc, bn = errs(v)
    prod_row = full_report(comp, float(tot))
    prod_row.update(arm="production", correct=bc, nearest=bn, chamfer_ratio=1.0)

    results = [prod_row]
    base_c = prod_row["chamfer_raw"]
    for name in ("CORRECT", "CONSENSUS", "FREE"):
        setp()
        for nm in FREE_P:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE_P], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            t_, _ = stage.forward(src, it=0)
            (t_ + LAM * loss(v, REG[name])).backward()
            opt.step()
        with torch.no_grad():
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, comp = stage.forward(src, it=0)
            c, n_ = errs(v)
        row = full_report(comp, float(tot))
        row.update(arm=name, correct=c, nearest=n_,
                   chamfer_ratio=row["chamfer_raw"] / base_c if base_c else None)
        results.append(row)
        print(f"{name:12s} total={row['total']:.6f}  chamfer_ratio={row['chamfer_ratio']:.3f}x  "
              f"correct_err={c:.1f}%", flush=True)

    print("\n" + "=" * 100)
    print("R9 -- full production-objective decomposition, production vs CORRECT/CONSENSUS/FREE")
    print("=" * 100)
    hdr = f"{'arm':12s}{'TOTAL':>12s}"
    for t in ALL_TERMS:
        hdr += f"{t:>12s}"
    print(hdr)
    for row in results:
        line = f"{row['arm']:12s}{row['total']:>12.5f}"
        for t in ALL_TERMS:
            wv = row.get(f"{t}_weighted")
            line += f"{wv:>12.5f}" if wv is not None else f"{'--':>12s}"
        print(line)

    R = {r["arm"]: r for r in results}
    delta = R["CORRECT"]["total"] - R["production"]["total"]
    print(f"\nTOTAL production objective: production={R['production']['total']:.6f}  "
          f"CORRECT={R['CORRECT']['total']:.6f}  delta(CORRECT-production)={delta:+.6f}")
    if delta < 0:
        print("=> CASE 1: CORRECT's objective is LOWER. The objective is compatible with the "
              "anatomically correct solution; the remaining problem is search/discovery.")
    elif delta > 0:
        print("=> CASE 2: CORRECT's objective is HIGHER. The objective itself disfavours the "
              "anatomically correct solution -- redesign candidate, not just a search problem.")
    else:
        print("=> tie (should not happen with float totals)")

    print("\nPer-term delta (CORRECT_weighted - production_weighted), positive = CORRECT pays more:")
    for t in ALL_TERMS:
        pw = R["production"].get(f"{t}_weighted")
        cw = R["CORRECT"].get(f"{t}_weighted")
        if pw is None or cw is None:
            continue
        print(f"  {t:12s} production={pw:>10.6f}  CORRECT={cw:>10.6f}  delta={cw-pw:>+10.6f}")

    json.dump(results, open(os.path.join(HERE, "r9_results.json"), "w"), indent=2)
    print("\nwrote r9_results.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
