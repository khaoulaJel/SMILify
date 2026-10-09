"""V7 -- does a recalibrated landmark GENERALISE to a specimen it was not calibrated on?

V6 recalibrated the template landmark indices from 12 human-annotated specimens, then applied them
to all 757. The calibration set and the evaluation set were the same, so the improvement it
reported is reachability, not generalisation -- the same objection G7 carries about joint targets.

Leave-one-specimen-out fixes that: calibrate each landmark on 11 specimens, then measure the error
on the 12th, which contributed nothing to it.

Three references per landmark, all in % of Weber's length:
  OLD        the September rule-computed index -- what the trait layer has been using
  LOO        calibrated on the other 11
  FLOOR      the nearest point on the fitted surface -- the best any vertex index could do,
             i.e. the part of the error that belongs to the FIT rather than to the landmark
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

from scipy.spatial import cKDTree  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
import trait_extract as TX  # noqa: E402

OLD = {"clypeal_ant_mid": ("clypeal_margin_ant_mid", 0),
       "cephalic_post_mid": ("cephalic_margin_post_mid", 0),
       "mandibular_apex_r": ("mandibular_apex_r", 0), "mandibular_apex_l": ("mandibular_apex_r", 1),
       "antennal_insertion_r": ("antennal_insertion_r", 0),
       "antennal_insertion_l": ("antennal_insertion_r", 1),
       "scape_apex_r": ("scape_apex_r", 0), "scape_apex_l": ("scape_apex_r", 1),
       "wl_anterior_r": ("wl_anterior_r", 0), "wl_posterior_r": ("wl_posterior_r", 0)}


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


def main():
    M = TX.load_model(); lm = TX.load_landmarks(M); T = M["v_template"]
    prod = {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        for i, l in enumerate(d["labels"]):
            prod[str(l).replace("_processed.obj", "")] = {
                k: np.asarray(d[k], np.float32)[i] for k in
                ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                 "betas_trans", "deform_verts"]}
    ann = {}
    for p in sorted(glob.glob(os.path.join(REPO, "annotation/landmarks/*_traits.json"))):
        d = json.load(open(p))
        ann[d["specimen_id"]] = {k: np.array(v["original"], float)
                                 for k, v in d["landmarks"].items()}
    sids = [s for s in ann if s in prod]
    smal = SMAL3DFitter(batch_size=len(sids), device="cpu")
    with torch.no_grad():
        for k in prod[sids[0]]:
            if hasattr(smal, k):
                getattr(smal, k).copy_(torch.as_tensor(np.stack([prod[s][k] for s in sids])))
        V = smal().numpy().astype(np.float64)

    # per specimen: the vertex nearest its own human landmark, plus the floor distance
    pick, floor, tgt = {}, {}, {}
    for i, sid in enumerate(sids):
        o = readobj(os.path.join("/hpcwork/nao48500/worker_alt_data", f"{sid}_processed.obj"))
        c, sc = o.mean(0), np.abs(o - o.mean(0)).max()
        L = ann[sid]
        WL = np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"]) / sc
        tree = cKDTree(V[i])
        for k, v in L.items():
            t = (v - c) / sc
            dist, idx = tree.query(t)
            pick.setdefault(k, {})[sid] = int(idx)
            floor.setdefault(k, {})[sid] = float(dist / WL * 100)
            tgt.setdefault(k, {})[sid] = (t, WL)

    print(f"{len(sids)} specimens, leave-one-out\n")
    print(f"{'landmark':22s}{'n':>3}{'OLD':>9}{'LOO':>9}{'FLOOR':>9}{'LOO beats OLD':>15}")
    res, tot = {}, {"old": [], "loo": [], "floor": []}
    for k in pick:
        eo, el, ef = [], [], []
        for i, sid in enumerate(sids):
            if sid not in pick[k]:
                continue
            t, WL = tgt[k][sid]
            others = [pick[k][s] for s in sids if s in pick[k] and s != sid]
            if not others:
                continue
            P = T[others]
            cons = others[int(np.argmin(np.linalg.norm(P[:, None] - P[None], axis=-1).sum(1)))]
            el.append(float(np.linalg.norm(V[i][cons] - t) / WL * 100))
            ef.append(floor[k][sid])
            if k in OLD:
                eo.append(float(np.linalg.norm(V[i][lm[OLD[k][0]][OLD[k][1]]] - t) / WL * 100))
        mo = np.median(eo) if eo else float("nan")
        ml, mf = np.median(el), np.median(ef)
        res[k] = {"old": mo if mo == mo else None, "loo": ml, "floor": mf, "n": len(el)}
        tot["old"] += eo; tot["loo"] += el; tot["floor"] += ef
        print(f"{k:22s}{len(el):>3}{mo:>8.1f}%{ml:>8.1f}%{mf:>8.1f}%"
              f"{('yes' if ml < mo else 'no') if mo == mo else '-':>15}")
    MO, ML, MF = (np.median(tot["old"]), np.median(tot["loo"]), np.median(tot["floor"]))
    print(f"{'ALL':22s}{len(tot['loo']):>3}{MO:>8.1f}%{ML:>8.1f}%{MF:>8.1f}%")
    print(f"\n  OLD {MO:.1f}%  ->  LOO {ML:.1f}%   ({100*(1-ML/MO):.0f}% better on held-out data)")
    print(f"  FLOOR {MF:.1f}% is what the FIT costs -- no landmark index can do better.")
    print(f"  LOO is {ML/MF:.1f}x the floor, so {100*(1-MF/ML):.0f}% of the remaining error is "
          f"still the landmark, not the fit.")
    json.dump(res, open(os.path.join(HERE, "v7_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
