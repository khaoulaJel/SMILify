"""V9 -- the four-arm cross-modality test. Does SKELETAL supervision propagate to the SKIN, and
are the two anatomical channels complementary?

V5b established joints -> joints (32.7% -> 15.2%, 1.54x chamfer). V8 established surface -> surface
(held-out 12.9% -> 6.9%, 1.07x chamfer) and, importantly, surface -> joints is ABSENT (33.22% ->
33.00%). The missing arrow is joints -> surface, and it is the arm that decides whether the model's
articulation is anatomically meaningful or merely a convenient parameterisation.

ARMS, all judged on the SAME four never-supervised surface landmarks:
  A  D1                              baseline
  B  D1 + surface landmarks          the V8 mechanism
  C  D1 + joints                     does skeletal correction reach the skin?
  D  D1 + joints + surface landmarks are the channels complementary?

In arm C the surface landmarks enter no objective at all, so its held-out column cannot be
memorised -- the evaluation quantity is a different KIND of observation from the supervision.

`deform_verts` RMS is monitored throughout: V1 found the joint term inflates it 2.4x, which is the
mechanism by which the surface could absorb a skeletal correction instead of following it.
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

MESHDIR = "/hpcwork/nao48500/worker_alt_data"
HELDOUT = ["mandibular_apex_l", "antennal_insertion_l", "scape_apex_l", "wl_posterior_r"]
LAM_J, LAM_L = 0.03, 1.0          # operating points from V5b and V8
ARMS = [("A  D1", 0.0, 0.0), ("B  +landmarks", 0.0, LAM_L),
        ("C  +joints", LAM_J, 0.0), ("D  +both", LAM_J, LAM_L)]
N_IT = int(os.environ.get("V9_ITERS", "800"))
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
BOUNDS = {"HW/HL": (0.40, 2.00), "SL/HL": (0.20, 3.00), "ML/HL": (0.10, 2.00)}
NEWLM = json.load(open(os.path.join(REPO, "annotation/landmark_indices_recalibrated.json")))


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


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
    jn = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float64)
    M = TX.load_model(); lm = TX.load_landmarks(M)

    prod, pj = {}, {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        V = np.asarray(d["verts"], dtype=np.float64)
        Jf = (np.einsum("jv,nvc->njc", Jr, V) if Jr.shape[1] == V.shape[1]
              else np.einsum("vj,nvc->njc", Jr, V))
        for i, lab in enumerate(d["labels"]):
            sid = str(lab).replace("_processed.obj", "")
            prod[sid] = {k: np.asarray(d[k], np.float32)[i] for k in
                         ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                          "betas_trans", "deform_verts"]}
            pj[sid] = Jf[i]

    spec = []
    for p in sorted(glob.glob(os.path.join(REPO, "annotation/landmarks/*_traits.json"))):
        d = json.load(open(p)); sid = d["specimen_id"]
        gj = os.path.join(REPO, "annotation/gt_expert", f"{sid}_edited_joints.json")
        if sid not in prod or not os.path.exists(gj):
            continue
        o = readobj(os.path.join(MESHDIR, f"{sid}_processed.obj"))
        c, s = o.mean(0), np.abs(o - o.mean(0)).max()
        L = {k: (np.array(v["original"], float) - c) / s for k, v in d["landmarks"].items()}
        gt = json.load(open(gj))
        P = {j["joint_name"]: np.array(j["position"], float)
             for j in gt["joints"] if j.get("position") is not None}
        use = [n for n in jn if n in P and n not in WING]
        idx = [jn.index(n) for n in use]
        spec.append(dict(sid=sid, obj=os.path.join(MESHDIR, f"{sid}_processed.obj"), L=L,
                         WL=float(np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"])),
                         jidx=idx, jbt=use.index("b_t"), jba3=use.index("b_a_3"),
                         jgt=torch.as_tensor(sim_fit(np.array([P[n] for n in use]), pj[sid][idx]),
                                             device=dev, dtype=torch.float32)))
    NS = len(spec)
    sup = [k for k in spec[0]["L"] if k not in HELDOUT]
    print(f"{NS} specimens | supervised landmarks {len(sup)} | held out {HELDOUT}", flush=True)

    _, targets = load_meshes(mesh_files=[a["obj"] for a in spec], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in spec], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/v9")
    P0 = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in spec]), device=dev)
          for k in prod[spec[0]["sid"]]}
    LM = {k: torch.as_tensor(np.stack([a["L"][k] if k in a["L"] else np.full(3, np.nan)
                                       for a in spec]), device=dev, dtype=torch.float32)
          for k in spec[0]["L"]}

    def setp():
        with torch.no_grad():
            for k, v in P0.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def p2s(v, keys):
        tot, n = 0.0, 0
        for k in keys:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            d = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1)
            tot = tot + (d.min(-1).values ** 2).mean(); n += 1
        return tot / max(n, 1)

    def joints(v):
        return (torch.einsum("jv,nvc->njc", smal.smal_model.J_regressor, v)
                if smal.smal_model.J_regressor.shape[1] == v.shape[1]
                else torch.einsum("vj,nvc->njc", smal.smal_model.J_regressor, v))

    def jterm(J):
        return torch.stack([((J[i, spec[i]["jidx"]] - spec[i]["jgt"]) ** 2).sum(-1).mean()
                            for i in range(NS)]).mean()

    def report():
        with torch.no_grad():
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            _, comp = stage.forward(src, it=0)
            out = {}
            for grp, keys in (("sup", sup), ("held", HELDOUT)):
                e = []
                for k in keys:
                    p = LM[k]; ok = ~torch.isnan(p[:, 0])
                    if not ok.any():
                        continue
                    d = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1).min(-1).values
                    wl = torch.as_tensor([a["WL"] for i, a in enumerate(spec) if ok[i]], device=dev)
                    e += list((d / wl * 100).cpu().numpy())
                out[grp] = float(np.median(e))
            J = joints(v); je = []
            for i, a in enumerate(spec):
                r = (J[i, a["jidx"]] - a["jgt"]).norm(dim=-1)
                bl = float((a["jgt"][a["jba3"]] - a["jgt"][a["jbt"]]).norm())
                je.append(float(torch.median(r)) / bl * 100)
            vv = v.cpu().numpy().astype(np.float64)
            nd = lambda a, b: np.linalg.norm(vv[:, NEWLM[a]["vertex"]] - vv[:, NEWLM[b]["vertex"]],
                                             axis=1)
            HL = nd("clypeal_ant_mid", "cephalic_post_mid")
            R = {"HW/HL": nd("head_width_r", "head_width_l") / HL,
                 "ML/HL": nd("mandibular_apex_r", "clypeal_ant_mid") / HL,
                 "SL/HL": nd("scape_apex_r", "antennal_insertion_r") / HL}
            viol = sum(int(((R[k] < lo) | (R[k] > hi)).sum()) for k, (lo, hi) in BOUNDS.items())
            asym = float(np.median(TX.asymmetry(TX.traits(vv, M, lm))["ML"]))
            dv = smal.deform_verts.detach().cpu().numpy()
            drms = float(np.median(np.sqrt((dv ** 2).sum(-1)).mean(-1)) * 1e3)
        return dict(sup=out["sup"], held=out["held"], joint=float(np.median(je)),
                    chamfer=float(comp.get("chamfer", np.nan)), viol=viol, asym=asym, deform=drms)

    setp()
    base = report(); base["arm"] = "production"
    print("\n" + "=" * 104)
    print("V9 -- four arms, all judged on the SAME four never-supervised surface landmarks")
    print("=" * 104)
    print(f"{'arm':16s}{'HELD-OUT surf':>15}{'sup surf':>10}{'joint err':>11}{'chamfer':>10}"
          f"{'x prod':>8}{'viol':>6}{'asym':>7}{'deform':>8}")

    def show(r):
        print(f"{r['arm']:16s}{r['held']:>14.1f}%{r['sup']:>9.1f}%{r['joint']:>10.2f}%"
              f"{r['chamfer']:>10.5f}{r['chamfer']/base['chamfer']:>7.2f}x{r['viol']:>6d}"
              f"{r['asym']:>7.3f}{r['deform']:>8.2f}", flush=True)
    show(base)
    rows = [base]
    for name, lj, ll in ARMS:
        setp()
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            loss = tot
            if lj:
                loss = loss + lj * jterm(joints(v))
            if ll:
                loss = loss + ll * p2s(v, sup)
            loss.backward(); opt.step()
        r = report(); r["arm"] = name; rows.append(r); show(r)

    R = {r["arm"]: r for r in rows}
    A, B, C, D = R["A  D1"], R["B  +landmarks"], R["C  +joints"], R["D  +both"]
    print(f"\n[VOIDING CONTROL] arm A (no supervision) vs production: held-out {A['held']:.1f}% vs "
          f"{base['held']:.1f}%, chamfer {A['chamfer']/base['chamfer']:.2f}x  ->  "
          f"{'PASS' if abs(A['held']-base['held'])<3 and A['chamfer']<1.5*base['chamfer'] else 'FAIL'}")
    print("\nTHE MISSING ARROW -- does skeletal supervision reach the skin?")
    print(f"  C vs A on held-out surface: {A['held']:.1f}% -> {C['held']:.1f}% "
          f"({100*(1-C['held']/A['held']):+.0f}%), joints {A['joint']:.1f}% -> {C['joint']:.1f}%")
    print("\nCOMPLEMENTARITY -- does adding joints to landmarks buy anything?")
    print(f"  B {B['held']:.1f}%  ->  D {D['held']:.1f}% ({100*(1-D['held']/B['held']):+.0f}%), "
          f"joints B {B['joint']:.1f}% -> D {D['joint']:.1f}%, chamfer "
          f"{B['chamfer']/base['chamfer']:.2f}x -> {D['chamfer']/base['chamfer']:.2f}x")
    json.dump(rows, open(os.path.join(HERE, "v9_results.json"), "w"), indent=2)
    print("=" * 104)


if __name__ == "__main__":
    main()
