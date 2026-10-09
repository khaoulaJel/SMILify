"""V10 -- is V8 robust to WHICH landmarks are held out?

V8 held out one set (left side + rear) and reported held-out landmark error 12.9% -> 6.9%. If that
is generalisation rather than a property of that particular split, it must survive holding out the
right side, the anterior, the head midline, and a mixed draw. Same loss, same lambda, four splits.

Original V8 docstring follows.

V8 -- can SPARSE SURFACE observations pull the fit toward real anatomy without the surface
distortion that interior joint supervision causes?

WHY THE LOSS IS POINT-TO-SURFACE AND NOT POINT-TO-VERTEX. V7 measured the held-out error of a
recalibrated template landmark at 44.2% of Weber's length while the NEAREST POINT on the same
fitted surface sits at 10.1% -- 4.4x closer. 77% of that error is the landmark index, not the fit:
a single template vertex cannot represent an anatomical point across these species, because the 12
specimens' independent votes disagree by 6-17% of a body length. Writing the loss as "template
vertex v belongs at human point p" would therefore train the fit to satisfy a correspondence the
data says is wrong.

So the term asks only that the SURFACE PASS THROUGH the human point, with no claim about which
vertex does it:

    L_landmark = mean_p  min_v || p - v ||^2

The assignment is recomputed every step, exactly as chamfer does, so it asserts nothing.

HELD-OUT BY CONSTRUCTION. Four landmarks are never supervised -- one from each region, all on the
left or the rear -- and the result is read on those. Supervising a point and then reporting that
the surface reaches it would only show reachability, which is the objection G7 and V5 both carry.
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
# V10: rotate WHICH landmarks are held out. V8 used one split (left + rear); if the 47% gain is
# real it must survive splits that hold out the right side, the anterior, and a random draw.
SPLITS = {
    "S1 left+rear (V8)": ["mandibular_apex_l", "antennal_insertion_l", "scape_apex_l",
                          "wl_posterior_r"],
    "S2 right+front":    ["mandibular_apex_r", "antennal_insertion_r", "scape_apex_r",
                          "wl_anterior_r"],
    "S3 head midline":   ["clypeal_ant_mid", "cephalic_post_mid", "head_width_r", "head_width_l"],
    "S4 mixed":          ["mandibular_apex_r", "antennal_insertion_l", "scape_apex_r",
                          "cephalic_post_mid"],
}
HELDOUT = SPLITS["S1 left+rear (V8)"]
LAMBDAS = [1.0]
N_IT = int(os.environ.get("V8_ITERS", "800"))
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
        gj = os.path.join(REPO, "annotation/gt_expert",
                          f"{sid}_edited_joints.json")
        if sid not in prod or not os.path.exists(gj):
            continue
        obj = os.path.join(MESHDIR, f"{sid}_processed.obj")
        o = readobj(obj); c, s = o.mean(0), np.abs(o - o.mean(0)).max()
        L = {k: (np.array(v["original"], float) - c) / s for k, v in d["landmarks"].items()}
        WL = float(np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"]))
        gt = json.load(open(gj))
        P = {j["joint_name"]: np.array(j["position"], float)
             for j in gt["joints"] if j.get("position") is not None}
        use = [n for n in jn if n in P and n not in WING]
        idx = [jn.index(n) for n in use]
        spec.append(dict(sid=sid, obj=obj, L=L, WL=WL, jidx=idx, jbt=use.index("b_t"),
                         jba3=use.index("b_a_3"),
                         jgt=torch.as_tensor(sim_fit(np.array([P[n] for n in use]), pj[sid][idx]),
                                             device=dev, dtype=torch.float32)))
    NS = len(spec)
    sup = [k for k in spec[0]["L"] if k not in HELDOUT]
    print(f"{NS} specimens | supervised {len(sup)} landmarks | held out {HELDOUT}", flush=True)

    _, targets = load_meshes(mesh_files=[a["obj"] for a in spec], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in spec], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/v8")
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
        """mean over landmarks of the squared distance to the NEAREST VERTEX. The assignment is
        recomputed every call, so no correspondence is asserted."""
        tot, n = 0.0, 0
        for k in keys:
            p = LM[k]
            ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            d = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1)     # (n, nverts)
            tot = tot + (d.min(-1).values ** 2).mean(); n += 1
        return tot / max(n, 1)

    def report_with(sup_keys, held_keys):
        with torch.no_grad():
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            _, comp = stage.forward(src, it=0)
            out = {}
            for grp, keys in (("supervised", sup_keys), ("HELD OUT", held_keys)):
                e = []
                for k in keys:
                    p = LM[k]; ok = ~torch.isnan(p[:, 0])
                    if not ok.any():
                        continue
                    d = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1).min(-1).values
                    wl = torch.as_tensor([a["WL"] for i, a in enumerate(spec) if ok[i]],
                                         device=dev)
                    e += list((d / wl * 100).cpu().numpy())
                out[grp] = float(np.median(e))
            J = (torch.einsum("jv,nvc->njc", smal.smal_model.J_regressor, v)
                 if smal.smal_model.J_regressor.shape[1] == v.shape[1]
                 else torch.einsum("vj,nvc->njc", smal.smal_model.J_regressor, v))
            je = []
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
            asym = TX.asymmetry(TX.traits(vv, M, lm))
        return (out["supervised"], out["HELD OUT"], float(np.median(je)),
                float(comp.get("chamfer", np.nan)), viol, float(np.median(asym["ML"])))

    import copy
    allres = {}
    print("\n" + "=" * 92)
    print("V10 -- does V8's held-out gain survive changing WHICH landmarks are held out?")
    print("=" * 92)
    print(f"{'split':22s}{'held out (prod)':>17}{'held out (sup)':>16}{'change':>9}"
          f"{'chamfer':>9}{'joints':>9}")
    rows = []
    for name, ho in SPLITS.items():
        globals()["HELDOUT"] = ho
        sup2 = [k for k in spec[0]["L"] if k not in ho]
        setp()
        s_p, h_p, j_p, c_p, v_p, a_p = report_with(sup2, ho)
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src_m = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src_m, it=0)
            (tot + LAMBDAS[0] * p2s(v, sup2)).backward()
            opt.step()
        s_s, h_s, j_s, c_s, v_s, a_s = report_with(sup2, ho)
        rows.append(dict(split=name, held_prod=h_p, held_sup=h_s, chamfer=c_s / c_p,
                         joint_prod=j_p, joint_sup=j_s, n_sup=len(sup2)))
        print(f"{name:22s}{h_p:>16.1f}%{h_s:>15.1f}%{100*(1-h_s/h_p):>8.0f}%"
              f"{c_s/c_p:>8.2f}x{j_s:>8.1f}%", flush=True)
    ch = [100 * (1 - r["held_sup"] / r["held_prod"]) for r in rows]
    print(f"\n  improvement across the four splits: {min(ch):.0f}% to {max(ch):.0f}%  "
          f"(median {np.median(ch):.0f}%)")
    print(f"  -> {'ROBUST: the gain does not depend on the split' if min(ch) > 20 else 'FRAGILE: at least one split fails'}")
    json.dump(rows, open(os.path.join(HERE, "v10_results.json"), "w"), indent=2)
    print("=" * 96)


if __name__ == "__main__":
    main()
