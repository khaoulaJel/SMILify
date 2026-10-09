"""Reproduce the real V6/V7 mandible-landmark measurement on one specimen and dump everything
needed to render it: fitted mesh, target scan mesh, human landmark, nearest-surface point,
and the OLD indexed 'mandibular_apex' vertex the pipeline names as that landmark.

Same data paths and same computation as diagnostics/groundtruth/v6_surface_accuracy.py and
v7_landmark_heldout.py -- nothing here is invented, this just picks one specimen and saves
the geometry instead of only printing the aggregate percentages.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = "/rwthfs/rz/cluster/home/nao48500/SMILify"
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
sys.path.insert(0, os.path.join(REPO, "diagnostics/groundtruth"))
os.chdir(REPO)

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
import trait_extract as TX  # noqa: E402

OLD_MAP = {"mandibular_apex_r": ("mandibular_apex_r", 0),
           "mandibular_apex_l": ("mandibular_apex_r", 1)}


def readobj(p):
    V = []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
    return np.array(V)


def main():
    M = TX.load_model()
    lm = TX.load_landmarks(M)

    prod = {}
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

    sids = [s for s in ann if s in prod and "mandibular_apex_r" in ann[s]]
    print(f"{len(sids)} candidate specimens")

    smal = SMAL3DFitter(batch_size=len(sids), device="cpu")
    P0 = {k: torch.as_tensor(np.stack([prod[s][k] for s in sids])) for k in prod[sids[0]]}
    with torch.no_grad():
        for k, v in P0.items():
            if hasattr(smal, k):
                getattr(smal, k).copy_(v)
        V_fit = smal().cpu().numpy().astype(np.float64)
        F = smal.faces.cpu().numpy() if hasattr(smal, "faces") else M["f"]

    # rank specimens by how close this reproduces the V6 headline ratio (~48.8 vs 7.7, ~6.4x)
    scored = []
    for i, sid in enumerate(sids):
        obj = readobj(os.path.join("/hpcwork/nao48500/worker_alt_data", f"{sid}_processed.obj"))
        c, sc = obj.mean(0), np.abs(obj - obj.mean(0)).max()
        L = ann[sid]
        WL = np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"]) / sc

        tree = cKDTree(V_fit[i])
        for side_key, (tname, side) in OLD_MAP.items():
            if side_key not in L:
                continue
            tgt = (L[side_key] - c) / sc
            old_idx = lm[tname][side]
            old_pt = V_fit[i][old_idx]
            d_old = float(np.linalg.norm(old_pt - tgt) / WL * 100)
            d_floor, floor_idx = tree.query(tgt)
            d_floor = float(d_floor / WL * 100)
            ratio = d_old / d_floor if d_floor > 0 else 0
            scored.append((abs(ratio - 6.4) + abs(d_old - 48.8) / 10, sid, side_key, {
                "sid": sid, "side_key": side_key, "old_idx": int(old_idx),
                "floor_idx": int(floor_idx), "d_old_pct_WL": d_old, "d_floor_pct_WL": d_floor,
                "ratio": ratio, "human_landmark_raw": L[side_key].tolist(),
                "human_landmark_fitter_frame": tgt.tolist(), "center": c.tolist(),
                "scale": float(sc), "WL_fitter_units": float(WL),
                "specimen_index": i}))

    scored.sort(key=lambda x: x[0])
    print("\ntop 8 closest to the 48.8% / 7.7% / 6.4x headline:")
    for score, sid, side_key, rec in scored[:8]:
        print(f"  {sid[:40]:42s} {side_key:20s} old={rec['d_old_pct_WL']:6.1f}%  "
              f"floor={rec['d_floor_pct_WL']:5.1f}%  ratio={rec['ratio']:.1f}x")

    best = scored[0][3]
    sid, i = best["sid"], best["specimen_index"]
    obj_path = os.path.join("/hpcwork/nao48500/worker_alt_data", f"{sid}_processed.obj")
    target_obj = readobj(obj_path)

    out_dir = "/w0/tmp/nao48500/login23-1_353989/claude-54434/-rwthfs-rz-cluster-home-nao48500-SMILify/9f42922d-7ac6-494a-aafb-70b3f65325c7/scratchpad/mesh_case"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "case.npz"),
              fitted_verts=V_fit[i], faces=F,
              target_verts_raw=target_obj,
              **{k: v for k, v in best.items() if k not in ("sid", "side_key")})
    json.dump({k: v for k, v in best.items() if k not in ("human_landmark_fitter_frame",)},
               open(os.path.join(out_dir, "case_meta.json"), "w"), indent=1)
    print(f"\nSAVED case: {sid} / {best['side_key']}  -> {out_dir}/case.npz")
    print(json.dumps({k: v for k, v in best.items()
                       if k not in ("human_landmark_fitter_frame", "human_landmark_raw")},
                      indent=1))


if __name__ == "__main__":
    main()
