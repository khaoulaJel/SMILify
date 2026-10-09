"""V12 -- what prevents the surface landmark from reaching the anatomy, given that we tell the
optimizer where the anatomy is?

V11 established the ~7% held-out floor is the MODEL, not the annotator (human repeatability 2.4%,
every V10 split floors 2-3x above its own noise). Three explanations remain:

  A  CORRESPONDENCE  -- the optimizer moves the wrong template surface toward the landmark
  B  CAPACITY        -- the 25-PC shape space cannot place these points without distorting others
  C  OPTIMISATION    -- it could, but the objective settles elsewhere

ARMS
  P        production
  S8       supervise 8, measure the 4 held out          -- replicates V8/V9, the ~7% floor
  S12      supervise ALL 12, measure all 12             -- the CAPACITY ceiling. If this reaches
                                                           ~0.1% then B is refuted outright: the
                                                           model CAN hold every landmark at once,
                                                           and the held-out gap is inference.
  S8_ORACLE  supervise 8 by VERTEX-to-point, using the per-specimen vertex that was nearest the
             human landmark in the production fit, instead of the free point-to-surface
             assignment -- gives the optimizer the correct correspondence identity and asks
             whether the free assignment was the problem.

ALSO MEASURED: oracle-vertex drift. For each held-out landmark, the distance from the human point
to the vertex that was nearest it in the PRODUCTION fit. If that grows while the nearest-surface
distance shrinks, the optimisation is breaking correspondence while improving geometry -- direct
evidence for A.
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
HELDOUT = ["mandibular_apex_l", "antennal_insertion_l", "scape_apex_l", "wl_posterior_r"]
LAM = 1.0
N_IT = int(os.environ.get("V12_ITERS", "800"))
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
    sup8 = [k for k in ALL if k not in HELDOUT]
    print(f"{NS} specimens | all {len(ALL)} landmarks | supervised-8 {len(sup8)} | "
          f"held out {HELDOUT}", flush=True)

    _, targets = load_meshes(mesh_files=[a["obj"] for a in spec], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in spec], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/v12")
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

    # per-specimen ORACLE correspondence: the vertex nearest the human landmark in production
    setp()
    with torch.no_grad():
        v0 = smal()
    ORACLE = {}
    for k in ALL:
        p = LM[k]; idx = torch.full((NS,), -1, dtype=torch.long, device=dev)
        ok = ~torch.isnan(p[:, 0])
        if ok.any():
            idx[ok] = torch.cdist(p[ok].unsqueeze(1), v0[ok]).squeeze(1).argmin(-1)
        ORACLE[k] = idx

    def p2s(v, keys):
        tot, n = 0.0, 0
        for k in keys:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            d = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1)
            tot = tot + (d.min(-1).values ** 2).mean(); n += 1
        return tot / max(n, 1)

    def v2p(v, keys):
        """vertex-to-point: the ORACLE vertex must go to the human point."""
        tot, n = 0.0, 0
        for k in keys:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            got = v[ok, ORACLE[k][ok]]
            tot = tot + ((got - p[ok]) ** 2).sum(-1).mean(); n += 1
        return tot / max(n, 1)

    def err(v, keys, mode):
        e = []
        for k in keys:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            if mode == "surface":
                d = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1).min(-1).values
            else:                                   # track the production-oracle vertex
                d = (v[ok, ORACLE[k][ok]] - p[ok]).norm(dim=-1)
            e += list((d / WL[ok] * 100).cpu().numpy())
        return float(np.median(e))

    def report():
        with torch.no_grad():
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            _, comp = stage.forward(src, it=0)
            return dict(held=err(v, HELDOUT, "surface"), sup8=err(v, sup8, "surface"),
                        all12=err(v, ALL, "surface"),
                        held_oracle=err(v, HELDOUT, "oracle"),
                        chamfer=float(comp.get("chamfer", np.nan)))

    def run(keys, loss_fn):
        setp()
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            (tot + LAM * loss_fn(v, keys)).backward()
            opt.step()
        return report()

    setp(); base = report()
    print("\n" + "=" * 96)
    print("V12 -- where does the remaining landmark error come from?")
    print("=" * 96)
    print(f"{'arm':26s}{'HELD-OUT':>11}{'supervised':>12}{'all 12':>9}"
          f"{'oracle-vtx (held)':>19}{'chamfer':>10}")

    def show(n, r):
        print(f"{n:26s}{r['held']:>10.1f}%{r['sup8']:>11.1f}%{r['all12']:>8.1f}%"
              f"{r['held_oracle']:>18.1f}%{r['chamfer']/base['chamfer']:>9.2f}x", flush=True)
    show("P  production", base)
    rows = {"P": base}
    rows["S8"] = run(sup8, p2s);            show("S8  supervise 8 (p2s)", rows["S8"])
    rows["S12"] = run(ALL, p2s);            show("S12 supervise ALL 12", rows["S12"])
    rows["S8o"] = run(sup8, v2p);           show("S8_ORACLE  vertex->point", rows["S8o"])

    print("\n" + "-" * 96)
    print("B  CAPACITY -- can the model hold every landmark at once?")
    print(f"   S12 all-12 error {rows['S12']['all12']:.1f}%  (production {base['all12']:.1f}%)")
    print("   => CAPACITY IS NOT THE BOTTLENECK; the held-out gap is inference, not expressiveness"
          if rows["S12"]["all12"] < 1.0 else
          "   => the model CANNOT place all 12 simultaneously -- capacity IS a real limit")
    print("\nA  CORRESPONDENCE -- does the free assignment beat a fixed oracle identity?")
    print(f"   S8 held-out {rows['S8']['held']:.1f}%   vs   S8_ORACLE held-out "
          f"{rows['S8o']['held']:.1f}%")
    print(f"   oracle-vertex drift on held-out: production {base['held_oracle']:.1f}% -> "
          f"S8 {rows['S8']['held_oracle']:.1f}%")
    json.dump(rows, open(os.path.join(HERE, "v12_results.json"), "w"), indent=2)
    print("=" * 96)


if __name__ == "__main__":
    main()
