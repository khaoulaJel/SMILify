"""V5 -- the annotated pilot. Can joint supervision improve REAL annotation error without
destroying surface fit?

TWO BLOCKERS FOUND BEFORE WRITING THIS, both recorded in RESULTS_V5:

1. `Eciton_burchellii` -- the only annotated specimen that is also trait-flagged -- has **zero
   anterior joints annotated** (35 joints: body axis + legs only). Its violation is HW/HL = 2.96,
   a HEAD trait. So Eciton alone cannot test the anterior pathology. Every other expert specimen
   has 8-9 anterior joints, so the pilot runs on all 12 rather than on Eciton alone, with Eciton
   reported separately.
2. **No surface-landmark annotations exist for any specimen.** `04_blender_trait_annotator.py` is
   written and ready but has never been run; `trait_session/` holds only joint files. The
   "+landmarks" and "+both" arms are therefore NOT RUNNABLE and are not faked.

Scored against `annotation/gt_expert/` -- the CORRECTED annotations. Every previous G-series run
used the superseded `gt_batch1`, whose mandibles and coxae are ~20% of a mesosoma off.
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

import pickle  # noqa: E402
import yaml  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
import trait_extract as TX  # noqa: E402

NEWLM = json.load(open(os.path.join(REPO, "annotation/landmark_indices_recalibrated.json")))

MESHDIR = "/hpcwork/nao48500/worker_alt_data"
ANTERIOR = {"b_h", "ma_r", "ma_l", "an_1_r", "an_2_r", "an_3_r", "an_1_l", "an_2_l", "an_3_l"}
WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
LAMBDAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
N_IT = int(os.environ.get("V5_ITERS", "800"))
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
BOUNDS = {"HW/HL": (0.40, 2.00), "SL/HL": (0.20, 3.00), "PetL/WL": (0.02, 0.80),
          "ML/HL": (0.10, 2.00), "GL/WL": (0.50, 4.00)}


def norm_id(s):
    s = s.replace("_edited", "")
    h = len(s) // 2
    return s[:h] if len(s) % 2 == 0 and s[:h] == s[h:] else s


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
    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float64)
    M = TX.load_model(); lm = TX.load_landmarks(M)

    prod, pj = {}, {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        V = np.asarray(d["verts"], dtype=np.float64)
        Jf = np.einsum("jv,nvc->njc", Jr, V)
        for i, lab in enumerate(d["labels"]):
            sid = str(lab).replace("_processed.obj", "").replace(".obj", "")
            prod[sid] = {k: np.asarray(d[k], dtype=np.float32)[i] for k in
                         ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                          "betas_trans", "deform_verts"]}
            pj[sid] = Jf[i]

    ann = []
    for f in sorted(glob.glob(os.path.join(REPO, "annotation/gt_expert/*_joints.json"))):
        gt = json.load(open(f)); sid = norm_id(gt["specimen_id"])
        obj = os.path.join(MESHDIR, f"{sid}_processed.obj")
        if sid not in prod or not os.path.exists(obj):
            print(f"  SKIP {sid}: prod={sid in prod} obj={os.path.exists(obj)}")
            continue
        P = {j["joint_name"]: np.array(j["position"], float)
             for j in gt["joints"] if j.get("position") is not None}
        use = [n for n in names if n in P and n not in WING]
        idx = [names.index(n) for n in use]
        ann.append({"sid": sid, "obj": obj, "idx": idx, "use": use,
                    "ant": [k for k, n in enumerate(use) if n in ANTERIOR],
                    "post": [k for k, n in enumerate(use) if n not in ANTERIOR],
                    "bt": use.index("b_t"), "ba3": use.index("b_a_3"),
                    "gt": torch.as_tensor(sim_fit(np.array([P[n] for n in use]), pj[sid][idx]),
                                          device=dev, dtype=torch.float32)})
    NS = len(ann)
    print(f"{NS} expert-annotated specimens; anterior joints per specimen: "
          f"{[len(a['ant']) for a in ann]}", flush=True)

    _, targets = load_meshes(mesh_files=[a["obj"] for a in ann], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in ann], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/v5")
    P0 = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in ann]), device=dev)
          for k in prod[ann[0]["sid"]]}

    def setp():
        with torch.no_grad():
            for k, v in P0.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def joints(v):
        # J_regressor is stored (verts, joints) here, not (joints, verts) -- guard on the shape
        # rather than assuming, as G7 does. Dropping this guard is what void-ran job 3811983.
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
            _, comp = stage.forward(src, it=0)
            J = joints(v)
            allr, antr, postr, per = [], [], [], {}
            for i, a in enumerate(ann):
                r = (J[i, a["idx"]] - a["gt"]).norm(dim=-1)
                bl = float((a["gt"][a["ba3"]] - a["gt"][a["bt"]]).norm())
                tot = float(torch.median(r)) / bl * 100
                allr.append(tot); per[a["sid"]] = tot
                if a["ant"]:
                    antr.append(float(torch.median(r[a["ant"]])) / bl * 100)
                postr.append(float(torch.median(r[a["post"]])) / bl * 100)
            # V6 showed the OLD template landmark indices were 4-21% of a body length off, and
            # that ML/HL in particular read ~2.4x long. Score the violations on the RECALIBRATED
            # indices, otherwise the "supervision damages trait validity" trend is measured with
            # the broken ruler that produced it.
            vv = v.cpu().numpy().astype(np.float64)
            nd = lambda a, b: np.linalg.norm(vv[:, NEWLM[a]["vertex"]] - vv[:, NEWLM[b]["vertex"]],
                                             axis=1)
            HLn = nd("clypeal_ant_mid", "cephalic_post_mid")
            R = {"HW/HL": nd("head_width_r", "head_width_l") / HLn,
                 "ML/HL": nd("mandibular_apex_r", "clypeal_ant_mid") / HLn,
                 "SL/HL": nd("scape_apex_r", "antennal_insertion_r") / HLn}
            viol = sum(int(((R[k] < lo) | (R[k] > hi)).sum())
                       for k, (lo, hi) in BOUNDS.items() if k in R)
        return (float(np.median(allr)), float(np.median(antr)), float(np.median(postr)),
                float(comp.get("chamfer", np.nan)), viol, per)

    setp()
    a0, an0, po0, c0, v0, per0 = report()
    print("\n" + "=" * 92)
    print("V5 -- joint supervision on the 12 expert-annotated specimens (CORRECTED annotations)")
    print("=" * 92)
    hdr = (f"{'lambda':>9}{'joint err':>11}{'anterior':>10}{'posterior':>11}"
           f"{'chamfer':>10}{'x prod':>8}{'viol':>6}{'Eciton':>9}")
    print(hdr)
    ec = "Eciton_burchellii_CASENT0744558"
    print(f"{'production':>9}{a0:>10.2f}%{an0:>9.2f}%{po0:>10.2f}%{c0:>10.5f}"
          f"{1.0:>7.2f}x{v0:>6d}{per0.get(ec, float('nan')):>8.2f}%", flush=True)
    rows = [{"lambda": None, "joint": a0, "ant": an0, "post": po0, "chamfer": c0,
             "viol": v0, "per_specimen": per0}]

    for lam in LAMBDAS:
        setp()
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            (tot + lam * jterm(joints(v))).backward()
            opt.step()
        a, an, po, c, vi, per = report()
        rows.append({"lambda": lam, "joint": a, "ant": an, "post": po, "chamfer": c,
                     "viol": vi, "per_specimen": per})
        print(f"{lam:>9.3g}{a:>10.2f}%{an:>9.2f}%{po:>10.2f}%{c:>10.5f}"
              f"{c/c0:>7.2f}x{vi:>6d}{per.get(ec, float('nan')):>8.2f}%", flush=True)

    sw = [r for r in rows if r["lambda"] is not None]
    ctrl = sw[0]
    ok = abs(ctrl["chamfer"] - c0) / c0 < 0.5 and abs(ctrl["joint"] - a0) < 5
    print(f"\n[VOIDING CONTROL] lambda=0: joints {ctrl['joint']:.2f}% (prod {a0:.2f}%), chamfer "
          f"{ctrl['chamfer']:.5f} (prod {c0:.5f})  ->  "
          f"{'PASS' if ok else 'FAIL -- ROWS ARE VOID'}")
    cheap = [r for r in sw if r["chamfer"] <= 1.5 * c0]
    if cheap:
        b = min(cheap, key=lambda r: r["joint"])
        print(f"[DECISION] best joint error at <=1.5x production chamfer: {b['joint']:.2f}% "
              f"vs production {a0:.2f}% (lambda {b['lambda']}, chamfer {b['chamfer']/c0:.2f}x)")
        print("  => anatomical accuracy IS improvable without destroying surface fit"
              if b["joint"] < 0.8 * a0 else
              "  => NO arm improves anatomy at acceptable surface cost -- rethink the direction")
    json.dump(rows, open(os.path.join(HERE, "v5_results.json"), "w"), indent=2)
    print("=" * 92)


if __name__ == "__main__":
    main()
