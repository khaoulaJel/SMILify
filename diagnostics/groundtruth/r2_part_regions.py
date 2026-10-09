"""R2 -- is ANATOMICAL PART IDENTITY a sufficient region? The coarsest possible semantic output.

R1 swept Euclidean balls on the template and found exact vertex identity is reachable but costs
~20% more surface distortion, with the cost saturating once regions reach ~5% of the diagonal
(337 vertices). R2 replaces the ball with the model's OWN anatomical part -- the set of vertices
whose dominant skinning weight belongs to that joint -- which is anatomically bounded rather than
geometric, and mostly LARGER than 5% (the head part is 2174 vertices, 21% of the mesh).

WHY THIS IS THE DECISIVE SPEC. If part identity suffices, an automatic anatomical layer only has to
answer "which part is this?" -- a 51-way labelling problem -- rather than produce dense
correspondence. That is dramatically weaker than what W3/W5 were asked for and is the cheapest
possible interface between a semantic model and the fitter.

Each landmark is assigned to the part of its V6 recalibrated consensus vertex, so the mapping is
derived, not hand-written.

Original R1 docstring follows.

R1 -- can the fitter consume REGION-level anatomical constraints instead of exact vertices?

The V-series closed with three independent findings that a biological landmark is not a template
vertex: V7 (a held-out recalibrated vertex sits at 44.2% of Weber's length against a 10.1%
nearest-surface floor; the 12 specimens' votes disagree by 6-17% of a body length), V6 (the
rule-computed indices were 4-21% of a body length wrong), and V11b (a human annotator legitimately
chose two points ~38% apart on the same antennal socket).

If a landmark is a REGION, the supervision should say "some vertex of the correct anatomical region
must reach this point" rather than "vertex 6125 must reach this point".

    L = mean_p  min over v in REGION(p)  || p - v ||^2

REGION radius interpolates continuously between the two formulations already measured:
    radius 0    -> a single vertex     = V12's S8_ORACLE / v2p, which does not work
    radius inf  -> every vertex        = V12's S12, which reaches 0.2%
So the sweep answers, in one experiment, HOW MUCH correspondence precision the fitter needs -- and
whether a semantically meaningful region is enough.

All 12 landmarks are supervised and all 12 measured: this is a REACHABILITY test by design, which
is the correct form for "can the representation consume this constraint". It makes no
generalisation claim -- V13 showed those do not hold at this sample size.
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
RADII = ["part", 0.05, None]      # "part" = the anatomical part; 0.05 = R1's saturation point
LAM = 1.0
N_IT = int(os.environ.get("R1_ITERS", "800"))
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
NEWLM = json.load(open(os.path.join(REPO, "annotation/landmark_indices_recalibrated.json")))


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M = TX.load_model(); T = M["v_template"]
    diag = float(np.linalg.norm(np.ptp(T, 0)))
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

    # REGIONS on the template: vertices within r*diagonal of the recalibrated consensus vertex
    REG, PARTOF = {}, {}
    dom = M["dominant"]
    for r in RADII:
        if r is None:
            REG[r] = {k: None for k in ALL}          # unrestricted
            continue
        REG[r] = {}
        for k in ALL:
            if r == "part":
                # the anatomical part containing the recalibrated consensus vertex -- derived
                # from the model's own skinning weights, not hand-assigned
                j = int(dom[NEWLM[k]["vertex"]])
                idx = np.where(dom == j)[0]
                PARTOF[k] = M["jnames"][j]
            else:
                c0 = T[NEWLM[k]["vertex"]]
                idx = np.where(np.linalg.norm(T - c0, axis=1) <= r * diag)[0]
            if len(idx) == 0:
                idx = np.array([NEWLM[k]["vertex"]])
            REG[r][k] = torch.as_tensor(idx, device=dev)
    print(f"{NS} specimens, {len(ALL)} landmarks, template diagonal {diag:.3f}")
    print("region sizes (vertices per landmark, median over the 12):")
    for r in RADII:
        n = ("all 10235" if r is None else
             f"{int(np.median([len(REG[r][k]) for k in ALL]))}")
        lbl = ("anatomical part" if r == "part" else
               "inf" if r is None else f"{100*r:.1f}% of diagonal")
        print(f"   {lbl:>20}  ->  {n}")
    if PARTOF:
        print("\nlandmark -> anatomical part (derived from the consensus vertex):")
        for k in ALL:
            print(f"   {k:24s} {PARTOF[k]:10s} ({len(REG['part'][k])} verts)")

    _, targets = load_meshes(mesh_files=[a["obj"] for a in spec], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in spec], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/r1")
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

    def loss(v, regions):
        tot, n = 0.0, 0
        for k in ALL:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            sub = v[ok] if regions[k] is None else v[ok][:, regions[k]]
            d = torch.cdist(p[ok].unsqueeze(1), sub).squeeze(1)
            tot = tot + (d.min(-1).values ** 2).mean(); n += 1
        return tot / max(n, 1)

    def err_free(v):
        """true surface-to-landmark distance, unrestricted -- comparable to V12's S12"""
        e = []
        for k in ALL:
            p = LM[k]; ok = ~torch.isnan(p[:, 0])
            if not ok.any():
                continue
            d = torch.cdist(p[ok].unsqueeze(1), v[ok]).squeeze(1).min(-1).values
            e += list((d / WL[ok] * 100).cpu().numpy())
        return float(np.median(e))

    setp()
    with torch.no_grad():
        v = smal()
        src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
        _, comp = stage.forward(src, it=0)
        base, base_c = err_free(v), float(comp.get("chamfer", np.nan))
    print("\n" + "=" * 80)
    print("R1 -- how precise must the correspondence be?")
    print("=" * 80)
    print(f"{'region radius':>22}{'verts':>9}{'landmark err':>15}{'chamfer':>10}")
    print(f"{'production':>22}{'-':>9}{base:>14.1f}%{1.0:>9.2f}x")
    rows = []
    for r in RADII:
        setp()
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            (tot + LAM * loss(v, REG[r])).backward()
            opt.step()
        with torch.no_grad():
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            _, comp = stage.forward(src, it=0)
            e, c = err_free(v), float(comp.get("chamfer", np.nan))
        nv = "all" if r is None else str(int(np.median([len(REG[r][k]) for k in ALL])))
        lbl = ("ANATOMICAL PART" if r == "part" else
               "inf (= S12)" if r is None else f"{100*r:.1f}% of diagonal")
        rows.append(dict(radius=r, n_verts=nv, err=e, chamfer=c / base_c))
        print(f"{lbl:>22}{nv:>9}{e:>14.1f}%{c/base_c:>9.2f}x", flush=True)
    json.dump({"production": base, "rows": rows},
              open(os.path.join(HERE, "r2_results.json"), "w"), indent=2)
    P = next(r for r in rows if r["radius"] == "part")
    F = next(r for r in rows if r["radius"] is None)
    print()
    print(f"=> PART-level: landmark error {P['err']:.1f}%, chamfer {P['chamfer']:.2f}x")
    print(f"   Free assignment: {F['err']:.1f}%, {F['chamfer']:.2f}x")
    print("   => PART IDENTITY IS A SUFFICIENT INTERFACE. An automatic layer need only answer"
          if P["err"] < 1.0 and P["chamfer"] < 1.15 * F["chamfer"] else
          "   => part identity is NOT sufficient -- finer localisation is required")
    print('      "which anatomical part is this?" -- a 51-way labelling problem.'
          if P["err"] < 1.0 and P["chamfer"] < 1.15 * F["chamfer"] else "")
    print("=" * 80)


if __name__ == "__main__":
    main()
