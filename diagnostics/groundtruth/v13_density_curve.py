"""V13 -- how much anatomical information does SMILify need per specimen?

V12 settled the two structural questions: capacity is not the bottleneck (all 12 landmarks held
simultaneously at 0.2%, inside the prior, for 1.09x chamfer) and correspondence identity is not the
lever (oracle vertex->point 7.7% vs free assignment 7.3%). What remains is that supervising 8 of 12
landmarks leaves 7.3% on the other 4 -- sparse anatomical information corrects what it touches and
little else, the same coverage effect G8 found on the skeletal channel.

This measures the curve. Supervise k landmarks, evaluate the 12-k that were never supervised, for
k = 2, 4, 6, 8, 10. The output is a quantitative specification for whatever supplies anatomy
automatically: how many constraints per specimen are needed, and where the returns flatten.

WHY MULTIPLE DRAWS PER k. V10 showed a single held-out split is not representative -- the same
intervention read 3% to 42% depending on which landmarks were held out, because the PRODUCTION
baseline differs per landmark. Each k is therefore run with several independent random draws and
reported as a median with its spread. A single draw per k would produce a curve shaped mostly by
which landmarks happened to be chosen.
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

MESHDIR = "/hpcwork/nao48500/worker_alt_data"
KS = [2, 4, 6, 8, 10]
DRAWS = int(os.environ.get("V13_DRAWS", "3"))
LAM = 1.0
N_IT = int(os.environ.get("V13_ITERS", "800"))
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
SEED = 20260908


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
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
    NS = len(spec)
    ALL = list(spec[0]["L"])
    print(f"{NS} specimens, {len(ALL)} landmarks, {DRAWS} draws per k", flush=True)

    _, targets = load_meshes(mesh_files=[a["obj"] for a in spec], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in spec], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/v13")
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

    def p2s(v, keys):
        tot, n = 0.0, 0
        for k in keys:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            d = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1)
            tot = tot + (d.min(-1).values ** 2).mean(); n += 1
        return tot / max(n, 1)

    def err(v, keys):
        e = []
        for k in keys:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            d = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1).min(-1).values
            e += list((d / WL[ok] * 100).cpu().numpy())
        return float(np.median(e)) if e else float("nan")

    def run(sup):
        setp()
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            (tot + LAM * p2s(v, sup)).backward()
            opt.step()
        with torch.no_grad():
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            _, comp = stage.forward(src, it=0)
            held = [k for k in ALL if k not in sup]
            return err(v, held), err(v, sup), float(comp.get("chamfer", np.nan))

    setp()
    with torch.no_grad():
        v = smal()
        src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
        _, comp = stage.forward(src, it=0)
        base_all, base_c = err(v, ALL), float(comp.get("chamfer", np.nan))
    print(f"\nproduction, all 12 landmarks: {base_all:.1f}%\n")
    print("=" * 84)
    print("V13 -- how many anatomical constraints does a specimen need?")
    print("=" * 84)
    print(f"{'k supervised':>13}{'held out':>10}{'UNSEEN err (median)':>21}{'spread':>16}"
          f"{'chamfer':>10}")
    rows = []
    for k in KS:
        got = []
        for d in range(DRAWS):
            sup = list(rng.choice(ALL, size=k, replace=False))
            h, s, c = run(sup)
            got.append((h, s, c, sup))
        hs = [g[0] for g in got]
        rows.append(dict(k=k, held_median=float(np.median(hs)),
                         held_min=float(min(hs)), held_max=float(max(hs)),
                         chamfer=float(np.median([g[2] for g in got])),
                         draws=[{"sup": g[3], "held": g[0], "sup_err": g[1]} for g in got]))
        print(f"{k:>13}{12-k:>10}{np.median(hs):>20.1f}%{min(hs):>8.1f}-{max(hs):<7.1f}"
              f"{np.median([g[2] for g in got])/base_c:>9.2f}x", flush=True)
    print(f"{12:>13}{0:>10}{'0.2% (V12, no held-out)':>21}{'':>16}{'1.09x':>10}")
    print(f"{0:>13}{12:>10}{base_all:>20.1f}%{'(production)':>16}{'1.00x':>10}")
    json.dump({"production_all": base_all, "rows": rows},
              open(os.path.join(HERE, "v13_results.json"), "w"), indent=2)
    print("=" * 84)


if __name__ == "__main__":
    main()
