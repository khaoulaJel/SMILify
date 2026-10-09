"""Q3b S2: run the C3 CSE network on every shift set and measure the real-failure signature.

Network loading, point sampling, retrieval and cycle distance are imported unchanged from Q3a A4,
so synthetic and real numbers come from identical code. Per specimen: per-point leg accuracy on
retrieved tr/fe/ti/ta (+pt) vertices, error types (wrong side / wrong leg number / non-leg), cycle
median. DEG conditions are C0 with a scan degradation applied before sampling:
  DEG_noise  Gaussian noise, sigma = 0.5% of the normalised extent (extent is 2 after normalising)
  DEG_holes  10% of surface area removed in 10 patches (nearest face centroids to random seeds)
  DEG_distal ta+pt faces removed on 2 random legs
Decision rule in PREREGISTRATION.md.
"""
import glob
import zlib
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "Q3a_cse_transfer_case_study"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import anatomy_proxy as ap  # noqa: E402
import geodesic_full as gf  # noqa: E402
from a4_cycle_signal import load_net, retrieve, N_POINTS, N_REP  # noqa: E402
from a2_real_correspondence import seg_of  # noqa: E402

SETS = os.environ.get("S2_SETS", "/hpcwork/nao48500/review_methods/shift_sets")   # S2-confirm points this at P0 sets
OUT_NAME = os.environ.get("S2_OUT", "S2_shift_battery.json")
DO_DEG = os.environ.get("S2_DEG", "1") == "1"
SEGS = ("tr", "fe", "ti", "ta")


def face_area(V, F):
    return np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def degrade_faces(V, F, kind, rng, vlab):
    if kind == "holes":
        cen = V[F].mean(1)
        a = face_area(V, F)
        tree = cKDTree(cen)
        drop = np.zeros(len(F), bool)
        target = 0.10 * a.sum()
        for s in rng.choice(len(F), 10, replace=False):
            _, idx = tree.query(cen[s], k=min(len(F), 4000))
            cum = np.cumsum(a[idx])
            drop[idx[: np.searchsorted(cum, target / 10) + 1]] = True
        return F[~drop]
    if kind == "distal":
        legs = rng.choice([f"{k}_{s}" for k in (1, 2, 3) for s in ("r", "l")], 2, replace=False)
        fl = vlab[F[:, 0]]
        bad = np.array([str(n).startswith("l_") and str(n).split("_")[2] in ("ta", "pt") and
                        f"{str(n).split('_')[1]}_{str(n).split('_')[-1]}" in legs for n in fl])
        return F[~bad]
    return F


def sample(V, F, n, rng, noise=0.0):
    a = face_area(V, F)
    fi = rng.choice(len(F), n, p=a / a.sum())
    b = rng.dirichlet([1, 1, 1], n)
    P = (b[:, :, None] * V[F[fi]]).sum(1)
    if noise:
        P = P + rng.normal(0, noise, P.shape)
    return P, F[fi][np.arange(n), b.argmax(1)]


def score_specimen(net, keys, V, F, vlab, rng, device, deg=None, geo=None):
    c = V.mean(0)
    V = (V - c) / np.abs(V - c).max()
    Fd = degrade_faces(V, F, deg, rng, vlab) if deg in ("holes", "distal") else F
    acc = dict(correct=[], side=[], num=[], nonleg=[], cyc=[], geo=[])
    for _ in range(N_REP):
        P, corner = sample(V, Fd, N_POINTS, rng, noise=0.01 if deg == "noise" else 0.0)
        r, cyc = retrieve(net, keys, P, device)
        true, ret = vlab[corner], vlab[r]
        m = np.isin([seg_of(n) for n in true], SEGS)          # points truly on tr/fe/ti/ta/pt
        tl, rl = ap.coarse_vec(true[m], "leg"), ap.coarse_vec(ret[m], "leg")
        acc["correct"].append(tl == rl)
        acc["nonleg"].append(rl == "other")
        acc["side"].append((rl != "other") & (np.array([x[-1] for x in rl]) != np.array([x[-1] for x in tl]))
                           & (np.array([x[1] for x in rl]) == np.array([x[1] for x in tl])))
        acc["num"].append((rl != "other") & (np.array([x[1] for x in rl]) != np.array([x[1] for x in tl])))
        acc["cyc"].append(cyc)
        acc["geo"].append(gf.normalised_error(geo[0], geo[1], corner[m], r[m]))
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    return dict(leg_acc=float(cat["correct"].mean()), wrong_side=float(cat["side"].mean()),
                wrong_num=float(cat["num"].mean()), nonleg=float(cat["nonleg"].mean()),
                cycle_median=float(np.median(cat["cyc"])),
                geo_median=float(np.median(cat["geo"])), geo_frac_le_0p05=float((cat["geo"] <= 0.05).mean()))


def boot_ci(x, B=5000, rng=np.random.default_rng(1)):
    x = np.asarray(x)
    m = rng.choice(x, (B, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    net, _ = load_net(device)
    keys = net.vertex_embeddings()
    dd = ap.load_dd()
    F = np.asarray(dd["f"]).astype(np.int64)
    vlab = ap.template_vertex_labels(dd)
    geo = gf.load(dd)

    jobs = [(os.path.basename(p)[:-4], p, None) for p in sorted(glob.glob(os.path.join(SETS, "*.npz")))]
    c0 = os.path.join(SETS, "C0.npz")
    if DO_DEG:
        jobs += [(f"DEG_{k}", c0, k) for k in ("noise", "holes", "distal")]

    res = {}
    for name, path, deg in jobs:
        z = np.load(path)
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        per = [score_specimen(net, keys, z["verts"][i].astype(float), F, vlab, rng, device, deg, geo)
               for i in range(len(z["verts"]))]
        summ = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        summ["leg_acc_ci95"] = boot_ci([p["leg_acc"] for p in per])
        summ["measured_bend"] = float(z["measured_bend"])
        res[name] = dict(summary=summ, per_specimen=per)
        print(f"{name:<12} bend {summ['measured_bend']:5.1f}  leg_acc {summ['leg_acc']:.3f} "
              f"[{summ['leg_acc_ci95'][0]:.3f},{summ['leg_acc_ci95'][1]:.3f}]  side {summ['wrong_side']:.3f} "
              f"num {summ['wrong_num']:.3f} nonleg {summ['nonleg']:.3f}  cyc {summ['cycle_median']:.4f}  geo {summ['geo_median']:.3f} (<=.05 {summ['geo_frac_le_0p05']:.2f})", flush=True)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "out", OUT_NAME), "w"), indent=1)


if __name__ == "__main__":
    main()
