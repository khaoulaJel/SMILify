"""M08 tuning (PREREGISTRATION.md): one configuration of the pre-registered grid over the tuning set.

Tuning set: 48 synthetic specimens, REALPOSE construction with articulation from P0 `pose_train`
fits only (genera never in JAB or pose_eval), seed 781000; built once and cached. Criterion: mean
leg-correctness of the per-template-vertex targets (true identity), plus geodesic error and coverage.

Usage: python m08_tune.py --config <index 0..29>     (grid order printed by --list)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import anatomy_proxy as ap  # noqa: E402
import geodesic_full as gf  # noqa: E402
import m08_register as reg  # noqa: E402

OUT = "/hpcwork/nao48500/review_methods/M08"
TUNESET = os.path.join(OUT, "tuneset_pose_train48.npz")


def grid():
    g = []
    for b in (0.3, 1.0, 2.0):
        for l in (1.0, 10.0, 100.0):
            g.append(dict(method="BCPD", beta=b, lam=l))
    for b in (0.3, 1.0, 2.0):
        for l in (1.0, 10.0, 100.0):
            for t in (0.2, 0.5):
                g.append(dict(method="GBCPD", beta=b, lam=l, tau=t))
    for s in (0.5, 1.0, 2.0):
        g.append(dict(method="NICP", scale=s))
    return g


def tuneset():
    if os.path.isfile(TUNESET):
        return np.load(TUNESET, allow_pickle=True)
    import torch
    import make_shift_sets as mss
    split = json.load(open(os.path.join(HERE, "..", "P0_real_pose_corpus", "out", "split.json")))
    fits = sorted(glob.glob("/hpcwork/nao48500/review_methods/P0/runs/c??_s0/Stage_3_deform_fine.npz"))
    rng = np.random.default_rng(781000)
    wanted = set(rng.choice(sorted(split["pose_train"]), 48, replace=False))
    d = mss.real_param_set("REALPOSE", 781000, "cuda" if torch.cuda.is_available() else "cpu",
                           fit_paths=fits, wanted=wanted)
    os.makedirs(OUT, exist_ok=True)
    np.savez(TUNESET, verts=d["verts"], source_labels=d["source_labels"])
    assert len(d["verts"]) == 48, len(d["verts"])
    return np.load(TUNESET, allow_pickle=True)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--config", type=int, default=-1)
    ap_.add_argument("--list", action="store_true")
    a = ap_.parse_args()
    G = grid()
    if a.list or a.config < 0:
        for i, c in enumerate(G):
            print(i, c)
        return
    cfg = G[a.config]
    dd = ap.load_dd()
    Y, F = reg.template(dd)
    D, sa = gf.load(dd)
    legc = ap.coarse_vec(ap.template_vertex_labels(dd), "leg")
    leg = legc != "other"
    ts = tuneset()
    rows = []
    for i, V in enumerate(ts["verts"]):
        Xv = reg.normalise(V.astype(np.float64))
        X = reg.sample_surface(Xv, F, 8000, np.random.default_rng(1000 + i))
        t0 = time.time()
        if cfg["method"] == "NICP":
            T, m = reg.run_nicp(Y, F, Xv, F, scale=cfg["scale"])
        else:
            T, m = reg.run_bcpd(Y, X, F_src=F if cfg["method"] == "GBCPD" else None,
                                beta=cfg["beta"], lam=cfg["lam"], tau=cfg.get("tau"))
        _, nn = cKDTree(Xv).query(T)            # true identity of the scan point each target hits
        ok = m & leg
        rows.append(dict(leg_correct=float((legc[nn][ok] == legc[ok]).mean()) if ok.any() else float("nan"),
                         geo_median=float(np.median(gf.normalised_error(D, sa, np.where(m)[0], nn[m]))),
                         coverage=float(m.mean()), seconds=time.time() - t0))
        print(f"[tune] cfg {a.config} {cfg} specimen {i}: {rows[-1]}", flush=True)
    summ = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    os.makedirs(os.path.join(OUT, "tune"), exist_ok=True)
    json.dump(dict(config=cfg, index=a.config, summary=summ, per_specimen=rows),
              open(os.path.join(OUT, "tune", f"cfg{a.config:02d}.json"), "w"), indent=1)
    print(f"[tune] cfg {a.config} {cfg} SUMMARY {summ}", flush=True)


if __name__ == "__main__":
    main()
