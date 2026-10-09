"""V6 -- the first direct measurement of SURFACE accuracy in this project.

Every accuracy number to date has been joint-based (G1/G3/G6/G7/V5, against interior rotation
pivots) or self-referential (T1/V4, trait values against bounds we chose). The 12 annotated
surface landmarks make a third thing possible: how far the fitted SURFACE lands from an
anatomically defined point a human placed on the scan.

Frames: landmarks are exported in the .obj frame; `load_meshes` maps .obj -> fitter frame by
(v - mean)/max|v - mean|, so the inverse brings the fit to the landmarks. Nothing is fitted here.

Baseline: the same params with betas = 0 and deform = 0 -- the template in the production pose.
The gap between the two is what the shape space plus the deformation field actually buy on the
surface, which is G1's design applied to landmarks instead of joints.
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

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
import trait_extract as TX  # noqa: E402

# annotator name -> template landmark name (left variants come from the mirror map)
MAP = {"clypeal_ant_mid": ("clypeal_margin_ant_mid", 0),
       "cephalic_post_mid": ("cephalic_margin_post_mid", 0),
       "mandibular_apex_r": ("mandibular_apex_r", 0),
       "mandibular_apex_l": ("mandibular_apex_r", 1),
       "antennal_insertion_r": ("antennal_insertion_r", 0),
       "antennal_insertion_l": ("antennal_insertion_r", 1),
       "scape_apex_r": ("scape_apex_r", 0),
       "scape_apex_l": ("scape_apex_r", 1),
       "wl_anterior_r": ("wl_anterior_r", 0),
       "wl_posterior_r": ("wl_posterior_r", 0)}


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


def main():
    dev = "cpu"
    M = TX.load_model()
    lm = TX.load_landmarks(M)          # name -> (right_idx, left_idx)
    prod, order = {}, []
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        for i, lab in enumerate(d["labels"]):
            sid = str(lab).replace("_processed.obj", "")
            prod[sid] = {k: np.asarray(d[k], dtype=np.float32)[i] for k in
                         ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                          "betas_trans", "deform_verts"]}

    ann = {}
    for p in sorted(glob.glob(os.path.join(REPO, "annotation/landmarks/*_traits.json"))):
        d = json.load(open(p))
        ann[d["specimen_id"]] = {k: np.array(v["original"], float)
                                 for k, v in d["landmarks"].items()}
    sids = [s for s in ann if s in prod]
    print(f"{len(sids)} specimens with both landmarks and a production fit", flush=True)

    smal = SMAL3DFitter(batch_size=len(sids), device=dev)
    P0 = {k: torch.as_tensor(np.stack([prod[s][k] for s in sids])) for k in prod[sids[0]]}

    def verts(zero_shape):
        with torch.no_grad():
            for k, v in P0.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)
            if zero_shape:
                smal.betas.zero_(); smal.deform_verts.zero_()
            return smal().cpu().numpy().astype(np.float64)

    V_fit, V_tpl = verts(False), verts(True)

    per = {}
    for arm, V in (("production", V_fit), ("template-in-pose", V_tpl)):
        rows = {}
        for i, sid in enumerate(sids):
            L = ann[sid]
            obj = readobj(os.path.join("/hpcwork/nao48500/worker_alt_data",
                                       f"{sid}_processed.obj"))
            c, sc = obj.mean(0), np.abs(obj - obj.mean(0)).max()
            WL = np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"]) / sc   # fitter units
            for k, (tname, side) in MAP.items():
                if k not in L:
                    continue
                tgt = (L[k] - c) / sc
                got = V[i][lm[tname][side]]
                rows.setdefault(k, []).append(float(np.linalg.norm(got - tgt) / WL * 100))
        per[arm] = rows

    print("\n" + "=" * 78)
    print("V6 -- distance from the FITTED SURFACE to the human landmark, % of Weber's length")
    print("=" * 78)
    print(f"{'landmark':24s}{'n':>4}{'production':>13}{'template':>11}{'fit helps by':>14}")
    allp, allt = [], []
    for k in MAP:
        if k not in per["production"]:
            continue
        a = np.array(per["production"][k]); b = np.array(per["template-in-pose"][k])
        allp += list(a); allt += list(b)
        print(f"{k:24s}{len(a):>4}{np.median(a):>12.1f}%{np.median(b):>10.1f}%"
              f"{100*(1-np.median(a)/np.median(b)):>13.0f}%")
    ap, at = np.array(allp), np.array(allt)
    print(f"{'ALL':24s}{len(ap):>4}{np.median(ap):>12.1f}%{np.median(at):>10.1f}%"
          f"{100*(1-np.median(ap)/np.median(at)):>13.0f}%")

    print("\nper specimen (production, median over its landmarks):")
    for i, sid in enumerate(sids):
        v = [per["production"][k][ [s for s in sids].index(sid) ] for k in per["production"]
             if len(per["production"][k]) == len(sids)]
        print(f"  {sid[:44]:46s}{np.median(v):>7.1f}%")

    json.dump({"per_landmark": {a: {k: v for k, v in r.items()} for a, r in per.items()},
               "specimens": sids}, open(os.path.join(HERE, "v6_results.json"), "w"), indent=1)
    print("=" * 78)


if __name__ == "__main__":
    main()
