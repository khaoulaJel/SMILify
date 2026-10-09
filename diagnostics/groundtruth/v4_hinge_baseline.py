"""V4 -- the weak hinge prior BASELINE. The number the real method must beat.

Not proposed as the solution. A hinge can only push a value back inside a hand-drawn box; it
cannot make a fit anatomically correct, and it may buy a lower violation count by COMPRESSING
legitimate anterior variation -- which a bound-based flag is constitutionally unable to see.
So the question here is never "does 9.1% fall" but "does it fall WITHOUT flattening real biology".

    L = L_production + w * mean_ratios,specimens [ relu(lo-r)^2 + relu(r-hi)^2 ] / (hi-lo)^2

The traits must be DIFFERENTIABLE, and `trait_extract` is numpy with a Kabsch alignment inside.
They are reimplemented in torch here, and a MANDATORY equivalence check against `trait_extract`
at the production parameters gates the whole run.
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
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))

import yaml  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
import trait_extract as TX  # noqa: E402
from measure import body_frame  # noqa: E402

MESHDIR = "/hpcwork/nao48500/worker_alt_data"
BOUNDS = {"HW/HL": (0.40, 2.00), "SL/HL": (0.20, 3.00), "PetL/WL": (0.02, 0.80),
          "ML/HL": (0.10, 2.00), "GL/WL": (0.50, 4.00)}
WEIGHTS = [0.0, 1.0, 10.0, 100.0]
N_IT = int(os.environ.get("V4_ITERS", "400"))
N_UNFLAGGED = int(os.environ.get("V4_UNFLAGGED", "200"))
CHUNK = int(os.environ.get("V4_CHUNK", "54"))
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
SEED = 20260907


class TorchTraits:
    """The five bounded GLAD ratios, differentiable. Mirrors trait_extract exactly; the
    equivalence check in main() is what licenses using it."""

    def __init__(self, M, lm, device):
        self.d = device
        self.pair = {}
        for name, (kind, spec, _s, _n) in TX.TRAITS.items():
            if kind == "pair":
                self.pair[name] = (lm[spec[0]][0], lm[spec[1]][0])
        self.head = torch.as_tensor(
            np.where(M["dominant"] == M["jnames"].index("b_h"))[0], device=device)
        V0 = M["v_template"][self.head.cpu().numpy()]
        self.T = torch.as_tensor(V0 - V0.mean(0), device=device, dtype=torch.float32)
        self.ax = torch.as_tensor(body_frame(M)[0], device=device, dtype=torch.float32)

    def _hw(self, v):
        P = v[:, self.head]
        Pc = P - P.mean(1, keepdim=True)
        H = Pc.transpose(1, 2) @ self.T                      # (n,3,3)
        U, _, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
        R = U @ D @ Vt                                        # T ~ Pc @ R, as measure.kabsch
        proj = (Pc @ R) @ self.ax
        return proj.max(1).values - proj.min(1).values

    def __call__(self, v):
        def p(name):
            a, b = self.pair[name]
            return (v[:, a] - v[:, b]).norm(dim=-1)
        HL, ML, SL = p("HL"), p("ML"), p("SL")
        WL, PetL, GL = p("WL"), p("PetL"), p("GL")
        return {"HW/HL": self._hw(v) / HL, "SL/HL": SL / HL, "PetL/WL": PetL / WL,
                "ML/HL": ML / HL, "GL/WL": GL / WL}


def hinge(R):
    t = 0.0
    for k, (lo, hi) in BOUNDS.items():
        r = R[k]
        t = t + ((torch.relu(lo - r) ** 2 + torch.relu(r - hi) ** 2) / (hi - lo) ** 2).mean()
    return t / len(BOUNDS)


def severity_np(R):
    s = None
    for k, (lo, hi) in BOUNDS.items():
        o = np.maximum(np.maximum(lo - R[k], R[k] - hi), 0) / (hi - lo)
        s = o if s is None else np.maximum(s, o)
    return s


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    M = TX.load_model(); lm = TX.load_landmarks(M)
    TT = TorchTraits(M, lm, dev)

    labels, PAR = [], None
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        labels += [str(x) for x in d["labels"]]
        cur = {k: np.asarray(d[k], dtype=np.float32) for k in
               ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                "betas_trans", "deform_verts"]}
        PAR = cur if PAR is None else {k: np.concatenate([PAR[k], cur[k]]) for k in PAR}
    v2 = json.load(open(os.path.join(HERE, "v2_results.json")))
    flagged = set(v2["flagged_labels_with"])
    fl = np.array([l in flagged for l in labels])

    sel = np.concatenate([np.where(fl)[0],
                          rng.choice(np.where(~fl)[0], size=N_UNFLAGGED, replace=False)])
    print(f"{len(sel)} specimens: {int(fl.sum())} flagged + {N_UNFLAGGED} sampled unflagged",
          flush=True)

    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]

    out = {w: {"ratios": {k: [] for k in BOUNDS}, "chamfer": [], "labels": []}
           for w in WEIGHTS}
    checked = False
    for c0 in range(0, len(sel), CHUNK):
        ci = sel[c0:c0 + CHUNK]
        nb = len(ci)
        _, targets = load_meshes(mesh_files=[os.path.join(MESHDIR, labels[i]) for i in ci],
                                 device=dev)
        smal = SMAL3DFitter(batch_size=nb, device=dev)
        stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                              mesh_names=[labels[i] for i in ci],
                              loss_weights=st["loss_weights"], lr=st.get("lr", 5e-4), device=dev,
                              n_sample=st.get("n_sample", 6000),
                              edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/v4")
        P0 = {k: torch.as_tensor(PAR[k][ci], device=dev) for k in PAR}

        def setp():
            with torch.no_grad():
                for k, v in P0.items():
                    if hasattr(smal, k):
                        getattr(smal, k).copy_(v)

        # ---- MANDATORY EQUIVALENCE CHECK, once, at the production parameters ----------
        if not checked:
            setp()
            with torch.no_grad():
                v = smal()
            Rt = {k: val.cpu().numpy().astype(np.float64) for k, val in TT(v).items()}
            tn = TX.traits(v.cpu().numpy().astype(np.float64), M, lm)
            Rn = {k: tn[k.split("/")[0]] / np.maximum(tn[k.split("/")[1]], 1e-12) for k in BOUNDS}
            worst = max(float(np.max(np.abs(Rt[k] - Rn[k]) / np.maximum(np.abs(Rn[k]), 1e-12)))
                        for k in BOUNDS)
            ok = worst < 1e-4
            print(f"[EQUIVALENCE CHECK] torch traits vs trait_extract: max relative deviation "
                  f"{worst:.3e} (bar <1e-4)  ->  "
                  f"{'PASS' if ok else 'FAIL -- THE HINGE WOULD PENALISE THE WRONG QUANTITY'}",
                  flush=True)
            if not ok:
                sys.exit(1)
            checked = True

        for w in WEIGHTS:
            setp()
            for nm in FREE:
                getattr(smal, nm).requires_grad_(True)
            opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
            for it in range(N_IT):
                opt.zero_grad()
                v = smal()
                src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
                tot, _ = stage.forward(src, it=0)
                (tot + w * hinge(TT(v))).backward()
                opt.step()
            with torch.no_grad():
                v = smal()
                src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
                _, comp = stage.forward(src, it=0)
                R = {k: val.cpu().numpy().astype(np.float64) for k, val in TT(v).items()}
            for k in BOUNDS:
                out[w]["ratios"][k].append(R[k])
            out[w]["chamfer"].append(float(comp.get("chamfer", np.nan)))
            out[w]["labels"] += [labels[i] for i in ci]
        print(f"  chunk {c0//CHUNK + 1}/{-(-len(sel)//CHUNK)} done", flush=True)

    # ---------------- report ------------------------------------------------------------
    res = {}
    for w in WEIGHTS:
        R = {k: np.concatenate(out[w]["ratios"][k]) for k in BOUNDS}
        sev = severity_np(R)
        f = sev > 0
        lab = np.array(out[w]["labels"])
        isf = np.array([l in flagged for l in lab])
        res[w] = {"R": R, "sev": sev, "flag": f, "isf": isf,
                  "chamfer": float(np.mean(out[w]["chamfer"])), "labels": lab}

    base = res[0.0]
    print("\n" + "=" * 96)
    print("V4 -- weak hinge prior baseline. The number the real method must beat.")
    print("=" * 96)
    ctrl_broken = int(base["flag"][base["isf"]].sum())
    print(f"[CONTROL] w=0 re-optimised from production: {ctrl_broken}/{int(base['isf'].sum())} "
          f"originally-flagged specimens still violate, chamfer {base['chamfer']:.5f}")
    print(f"\n{'w':>7}{'flagged still broken':>22}{'newly broken':>14}{'corpus rate':>13}"
          f"{'median sev':>12}{'chamfer':>10}{'x w=0':>8}")
    for w in WEIGHTS:
        r = res[w]
        still = int(r["flag"][r["isf"]].sum())
        new = int(r["flag"][~r["isf"]].sum())
        rate = (still + new / max((~r["isf"]).sum(), 1) * (757 - 69)) / 757
        sv = r["sev"][r["flag"]]
        print(f"{w:>7.0f}{still:>16d}/{int(r['isf'].sum()):<5d}"
              f"{new:>8d}/{int((~r['isf']).sum()):<5d}{100*rate:>12.1f}%"
              f"{(np.median(sv) if len(sv) else 0):>12.3f}{r['chamfer']:>10.5f}"
              f"{r['chamfer']/base['chamfer']:>7.2f}x")

    print(f"\n=== does the hinge FLATTEN real biology? sd of log ratio, UNFLAGGED specimens only ===")
    print(f"{'ratio':>9}" + "".join(f"{'w='+str(int(w)):>10}" for w in WEIGHTS))
    comp = {}
    for k in BOUNDS:
        row = [float(np.std(np.log(np.maximum(res[w]["R"][k][~res[w]["isf"]], 1e-12))))
               for w in WEIGHTS]
        comp[k] = row
        print(f"{k:>9}" + "".join(f"{x:>10.4f}" for x in row))
    print(f"{'retained':>9}" + "".join(f"{'':>10}" if w == 0 else
          f"{np.mean([comp[k][WEIGHTS.index(w)]/max(comp[k][0],1e-12) for k in BOUNDS]):>9.2f}x"
          for w in WEIGHTS))

    json.dump({"weights": WEIGHTS, "n_iters": N_IT,
               "rows": {str(w): {"still_broken": int(res[w]["flag"][res[w]["isf"]].sum()),
                                 "newly_broken": int(res[w]["flag"][~res[w]["isf"]].sum()),
                                 "n_flagged": int(res[w]["isf"].sum()),
                                 "n_unflagged": int((~res[w]["isf"]).sum()),
                                 "median_severity": float(np.median(res[w]["sev"][res[w]["flag"]]))
                                 if res[w]["flag"].any() else 0.0,
                                 "chamfer": res[w]["chamfer"]} for w in WEIGHTS},
               "sd_log_unflagged": comp},
              open(os.path.join(HERE, "v4_results.json"), "w"), indent=2)
    print("=" * 96)


if __name__ == "__main__":
    main()
