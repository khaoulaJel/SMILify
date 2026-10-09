"""V1 -- validate the G7 joint-position term on real annotated scans, on the SURFACE and SHAPE
endpoints G7 never measured, and test whether free deformation is what lets the surface objective
ignore the skeleton.

See PREREGISTRATION_V1_real_scan_validation.md for the arms, the metrics and the bars.
Machinery is G7's, reused verbatim where possible so the two are directly comparable.
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

sys.path.insert(0, HERE)
import trait_extract as TX  # noqa: E402

WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
N_IT = int(os.environ.get("V1_ITERS", "800"))
FREE_ALL = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
FREE_NOD = ["global_rot", "joint_rot", "betas", "trans"]

# arm name -> (lambda, deform mode). deform: "free" | "prod" (frozen at production) | "zero"
ARMS = [
    ("L0_free",   0.0, "free"),
    ("L01_free",  0.1, "free"),
    ("L03_free",  0.3, "free"),
    ("L10_free",  1.0, "free"),
    ("L0_dfroz",  0.0, "prod"),
    ("L01_dfroz", 0.1, "prod"),
    ("L0_dzero",  0.0, "zero"),
    ("L01_dzero", 0.1, "zero"),
]

# T1's pre-set biological bounds. Wide by design: values no ant has.
BOUNDS = {"ML/HL": (0.10, 2.00), "HW/HL": (0.40, 2.00), "SL/HL": (0.20, 3.00),
          "GL/WL": (0.50, 4.00), "PetL/WL": (0.02, 0.80)}
RATIOS = ["HW/HL", "ML/HL", "SL/HL", "GL/WL", "PetL/WL", "HL/WL", "HW/WL"]


def sim_fit(src, tgt):
    cs, ct = src.mean(0), tgt.mean(0)
    s0, t0 = src - cs, tgt - ct
    U, S, Vt = np.linalg.svd(s0.T @ t0)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    sc = (S * np.array([1, 1, d])).sum() / (s0 ** 2).sum()
    return sc * ((src - cs) @ R) + ct


def ratios_of(verts, M, lm):
    """GLAD ratio table (n_specimens, ) per ratio name, plus bilateral asymmetry medians."""
    t = TX.traits(verts, M, lm)
    out = {}
    for r in RATIOS:
        a, b = r.split("/")
        out[r] = t[a] / np.maximum(t[b], 1e-12)
    asym = {k: float(np.median(v)) for k, v in TX.asymmetry(t).items()}
    return out, asym


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    sd_t = torch.as_tensor(np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64))),
                           device=dev, dtype=torch.float32)
    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float64)

    M = TX.load_model()
    lm = TX.load_landmarks(M)
    print(f"landmark set {TX.landmark_version()}", flush=True)

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
    print(f"{NS} annotated specimens", flush=True)
    _, targets = load_meshes(mesh_files=[a["obj"] for a in ann], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in ann], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/v1")
    P0 = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in ann]), device=dev,
                             dtype=torch.float32) for k in prod[ann[0]["sid"]]}
    # mesosoma length per specimen, in the annotation-aligned frame: the normaliser used everywhere
    meso = np.array([float((a["gt"][a["ba3"]] - a["gt"][a["bt"]]).norm()) for a in ann])

    def setp(deform_mode):
        with torch.no_grad():
            for k, v in P0.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)
            if deform_mode == "zero":
                smal.deform_verts.zero_()

    def joints(v):
        return (torch.einsum("jv,nvc->njc", smal.smal_model.J_regressor, v)
                if smal.smal_model.J_regressor.shape[1] == v.shape[1]
                else torch.einsum("vj,nvc->njc", smal.smal_model.J_regressor, v))

    def jterm(J):
        return torch.stack([((J[i, ann[i]["idx"]] - ann[i]["gt"]) ** 2).sum(-1).mean()
                            for i in range(NS)]).mean()

    def report(arm):
        with torch.no_grad():
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, comp = stage.forward(src, it=0)
            J = joints(v)
            per = []
            for i, a in enumerate(ann):
                r = (J[i, a["idx"]] - a["gt"]).norm(dim=-1)
                bl = float((a["gt"][a["ba3"]] - a["gt"][a["bt"]]).norm())
                per.append(float(torch.median(r)) / bl * 100)
            verts = v.detach().cpu().numpy().astype(np.float64)
            dv = smal.deform_verts.detach().cpu().numpy()
            # deform magnitude in the SAME frame the verts are in, then / mesosoma of that frame
            mfr = np.array([np.linalg.norm(
                np.einsum("jv,vc->jc", Jr, verts[i])[names.index("b_a_3")] -
                np.einsum("jv,vc->jc", Jr, verts[i])[names.index("b_t")])
                if Jr.shape[1] == verts.shape[1] else 1.0 for i in range(NS)])
            drms = float(np.median(np.sqrt((dv ** 2).sum(-1)).mean(-1) / np.maximum(mfr, 1e-12)) * 100)
        rat, asym = ratios_of(verts, M, lm)
        viol = 0
        for r, (lo, hi) in BOUNDS.items():
            viol += int(((rat[r] < lo) | (rat[r] > hi)).sum())
        return {
            "arm": arm,
            "joint_resid": float(np.median(per)), "joint_per_specimen": per,
            "chamfer": float(comp.get("chamfer", np.nan)), "objective": float(tot),
            "betas_z": float((smal.betas.detach().abs() / sd_t).mean()),
            "deform_rms_pct": drms,
            "log_beta_scales_abs": float(smal.log_beta_scales.detach().abs().mean()),
            "ratios_median": {r: float(np.median(rat[r])) for r in RATIOS},
            "ratios_all": {r: [float(x) for x in rat[r]] for r in RATIOS},
            "impossible": viol,
            "asymmetry": asym,
        }

    rows = []
    setp("free")
    base = report("production")
    rows.append(base)
    hdr = (f"{'arm':>11}{'joints':>9}{'chamfer':>11}{'x prod':>8}{'|z|':>6}"
           f"{'deform%':>9}{'impossible':>12}{'asym ML':>9}")
    print("=" * len(hdr)); print("V1 -- joint term on real scans: surface, shape and deformation")
    print("=" * len(hdr)); print(hdr, flush=True)

    def show(r):
        print(f"{r['arm']:>11}{r['joint_resid']:>8.2f}%{r['chamfer']:>11.5f}"
              f"{r['chamfer']/base['chamfer']:>7.2f}x{r['betas_z']:>6.2f}"
              f"{r['deform_rms_pct']:>9.3f}{r['impossible']:>12d}"
              f"{r['asymmetry'].get('ML', float('nan')):>9.3f}", flush=True)
    show(base)

    for arm, lam, dmode in ARMS:
        setp(dmode)
        free = FREE_ALL if dmode == "free" else FREE_NOD
        for nm in FREE_ALL:
            getattr(smal, nm).requires_grad_(nm in free)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in free], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            loss = tot + lam * jterm(joints(v))
            loss.backward()
            opt.step()
            with torch.no_grad():
                smal.betas.clamp_(min=-sd_t, max=sd_t)
        r = report(arm); r["lambda"] = lam; r["deform"] = dmode
        rows.append(r); show(r)

    by = {r["arm"]: r for r in rows}
    json.dump(rows, open(os.path.join(HERE, "v1_results.json"), "w"), indent=2)

    print("\n" + "=" * 78)
    print("PRE-REGISTERED BARS")
    print("=" * 78)
    c = by["L0_free"]
    ctrl_ok = (abs(c["chamfer"] - base["chamfer"]) / base["chamfer"] < 0.5
               and abs(c["joint_resid"] - base["joint_resid"]) < 5.0)
    print(f"[VOIDING CONTROL] L0_free chamfer {c['chamfer']:.5f} vs prod {base['chamfer']:.5f}, "
          f"joints {c['joint_resid']:.2f}% vs {base['joint_resid']:.2f}%  ->  "
          f"{'PASS' if ctrl_ok else 'FAIL -- EVERY ROW BELOW IS VOID'}")

    j = by["L01_free"]
    a_p = base["asymmetry"].get("ML", float("nan"))
    a_j = j["asymmetry"].get("ML", float("nan"))
    h1 = (j["impossible"] <= base["impossible"]) and (a_j <= 1.5 * a_p)
    print(f"[H1 surface not damaged] impossible {base['impossible']} -> {j['impossible']}, "
          f"ML asymmetry {a_p:.3f} -> {a_j:.3f}  ->  {'PASS' if h1 else 'FAIL'}")

    h2 = (j["deform_rms_pct"] <= 1.5 * base["deform_rms_pct"]
          and j["log_beta_scales_abs"] <= 1.5 * base["log_beta_scales_abs"])
    print(f"[H2 satisfied honestly] deform RMS {base['deform_rms_pct']:.3f}% -> "
          f"{j['deform_rms_pct']:.3f}%, |log_beta_scales| {base['log_beta_scales_abs']:.4f} -> "
          f"{j['log_beta_scales_abs']:.4f}  ->  {'PASS' if h2 else 'FAIL'}")

    tr = (by["L01_free"]["joint_resid"] - by["L01_dfroz"]["joint_resid"]) / by["L01_free"]["joint_resid"]
    cr = abs(by["L0_free"]["joint_resid"] - by["L0_dfroz"]["joint_resid"]) / by["L0_free"]["joint_resid"]
    h3 = tr >= 0.25 and cr < 0.25
    print(f"[H3 deform absorbs error] freezing deform at lam=0.1: {by['L01_free']['joint_resid']:.2f}%"
          f" -> {by['L01_dfroz']['joint_resid']:.2f}% ({tr*100:+.1f}% relative); "
          f"control at lam=0 moves {cr*100:.1f}%  ->  {'PASS' if h3 else 'FAIL'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
