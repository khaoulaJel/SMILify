"""Q3b S2 test sets: 48 synthetic specimens per condition, one factor changed from the training
generator (correlated sampler, rho 0.6, pose_scale 0.25, shape_scale 1.0, scale_scale 0.10).

Seeds 777000 + condition index, disjoint from the training corpus (20260826-20260833).

ART conditions: pose_scale is calibrated by bisection so that the median over specimens of the
per-specimen median leg bend vs template rest (Q3a A7 statistic, from FK joints) hits the target.
ARTLIM: the ART-real pose_scale, then joint_rot clamped to the model's authored joint limits
(anatomically allowed directions), then re-measured.
ROT: rigid rotation of the posed specimen about its centroid, random axis, fixed angle.
Scan degradations (DEG) are applied at point-sampling time by the evaluator, not here.

Every set stores verts, FK joints and all parameters, and records its measured bend statistic.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import anatomy_proxy as ap  # noqa: E402
from fitter_3d.pointcloud2smil.sample_smil_model import generate_correlated_chain_parameters  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

PARAMS = ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales", "betas_trans"]
BASE = dict(pose_scale=0.25, shape_scale=1.0, scale_scale=0.10, rho=0.6)
REAL_BEND = 29.6          # median of real expert-skeleton per-specimen median bends (Q3a A7)

DD = ap.load_dd()
NAMES, PARENT = ap.joint_tree(DD)
CHILD = {p: n for n, p in PARENT.items() if p is not None and n.startswith("l_")}
LEGJ = [n for n in NAMES if n.startswith("l_") and n.split("_")[2] in ("tr", "fe", "ti", "ta")]
REST = None


def _bends(J):
    out = []
    for j in LEGJ:
        p, c = PARENT[j], CHILD.get(j)
        if c is None:
            continue
        a, b, d = J[NAMES.index(p)], J[NAMES.index(j)], J[NAMES.index(c)]
        u, v = b - a, d - b
        out.append(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1))))
    return np.array(out)


def bend_stat(joints):
    global REST
    if REST is None:
        REST = _bends(np.asarray(DD["J"]))
    return float(np.median([np.median(np.abs(_bends(J) - REST)) for J in joints]))


def generate(n, seed, device, pose_scale, shape_scale, scale_scale, rho, clamp_limits=False):
    f = SMAL3DFitter(batch_size=n, device=device, shape_family=-1)
    generate_correlated_chain_parameters(f, seed=seed, random_dist="normal", shape_scale=shape_scale,
                                         pose_scale=pose_scale, trans_scale=0.0, scale_scale=scale_scale,
                                         global_rot_scale=0.0, rho=rho)
    if clamp_limits:
        lim = torch.as_tensor(np.asarray(DD["joint_limits"])[1:], dtype=torch.float32, device=device)
        f.joint_rot.data = torch.max(torch.min(f.joint_rot.data, lim[None, :, :, 1]), lim[None, :, :, 0])
    with torch.no_grad():
        v = f()
        j = f.smal_model.J_transformed + f.trans.unsqueeze(1)
    out = dict(verts=v.cpu().numpy(), joints=j.cpu().numpy())
    for k in PARAMS:
        out[k] = getattr(f, k).detach().cpu().numpy()
    return out


def calibrate(target, device, clamp=False, n=96, seed=123):
    lo, hi = 0.05, 2.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        b = bend_stat(generate(n, seed, device, mid, BASE["shape_scale"], BASE["scale_scale"], BASE["rho"], clamp)["joints"])
        if b < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rotate(d, deg, seed):
    rng = np.random.default_rng(seed)
    V, J = d["verts"].copy(), d["joints"].copy()
    for i in range(len(V)):
        ax = rng.normal(size=3)
        R = Rot.from_rotvec(np.radians(deg) * ax / np.linalg.norm(ax)).as_matrix()
        c = V[i].mean(0)
        V[i] = (V[i] - c) @ R.T + c
        J[i] = (J[i] - c) @ R.T + c
    d = dict(d)
    d["verts"], d["joints"] = V, J
    return d


REAL_FITS = [f"/hpcwork/nao48500/jab_runs/A_prod_s{s}/Stage_3_deform_fine.npz" for s in (0, 1, 2)]
REAL_FACTORS = {"REALPOSE": ["joint_rot"], "REALSHAPE": ["betas"],
                "REALSCALE": ["log_beta_scales", "betas_trans"],
                "REALALL": ["betas", "joint_rot", "log_beta_scales", "betas_trans"]}


def real_param_set(name, seed, device, fit_paths=None, wanted=None):
    """DEVIATIONS D3: real fitted parameters (factor subset) on clean model surfaces; the other
    parameters from the training generator. global_rot 0, trans 0, no deform."""
    fits = [np.load(p, allow_pickle=True) for p in (fit_paths or REAL_FITS)]
    keep = [np.array([wanted is None or str(l).replace("_processed", "").replace(".obj", "") in wanted for l in f["labels"]]) for f in fits]
    real = {k: np.concatenate([f[k][m] for f, m in zip(fits, keep)]) for k in ("betas", "joint_rot", "log_beta_scales", "betas_trans")}
    n = len(real["betas"])
    f = SMAL3DFitter(batch_size=n, device=device, shape_family=-1)
    generate_correlated_chain_parameters(f, seed=seed, random_dist="normal", shape_scale=BASE["shape_scale"],
                                         pose_scale=BASE["pose_scale"], trans_scale=0.0,
                                         scale_scale=BASE["scale_scale"], global_rot_scale=0.0, rho=BASE["rho"])
    with torch.no_grad():
        for k in REAL_FACTORS[name]:
            src = torch.as_tensor(real[k], dtype=torch.float32, device=device)
            assert tuple(getattr(f, k).shape) == tuple(src.shape), (k, getattr(f, k).shape, src.shape)
            getattr(f, k).copy_(src)
        v = f()
        j = f.smal_model.J_transformed + f.trans.unsqueeze(1)
    out = dict(verts=v.cpu().numpy(), joints=j.cpu().numpy())
    for k in PARAMS:
        out[k] = getattr(f, k).detach().cpu().numpy()
    out["source_labels"] = np.concatenate([[str(x) for x in np.asarray(fit["labels"])[m]] for fit, m in zip(fits, keep)])
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--n", type=int, default=48)
    ap_.add_argument("--out_dir", default="/hpcwork/nao48500/review_methods/shift_sets")
    a = ap_.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    conds = [("C0", {}, None)]
    for t in (15, 20, 25, REAL_BEND, 35):
        conds.append((f"ART{t:g}", {"_bend": t}, None))
    conds.append(("ARTLIM_real", {"_bend": REAL_BEND, "_clamp": True}, None))
    for deg in (10, 20, 30):
        conds.append((f"ROT{deg}", {}, deg))
    for s in (0.2, 0.3):
        conds.append((f"SCL{s}", {"scale_scale": s}, None))
    for s in (1.5, 2.0):
        conds.append((f"SHP{s}", {"shape_scale": s}, None))
    conds.append(("REAL", {"_bend": REAL_BEND, "scale_scale": 0.25}, 15))

    cal = {}
    for i, (name, over, rot) in enumerate(conds):
        p = dict(BASE)
        clamp = over.get("_clamp", False)
        if "_bend" in over:
            key = (over["_bend"], clamp)
            if key not in cal:
                cal[key] = calibrate(over["_bend"], dev, clamp)
            p["pose_scale"] = cal[key]
        p.update({k: v for k, v in over.items() if not k.startswith("_")})
        d = generate(a.n, 777000 + i, dev, p["pose_scale"], p["shape_scale"], p["scale_scale"], p["rho"], clamp)
        if rot:
            d = rotate(d, rot, 777000 + i)
        b = bend_stat(d["joints"])
        np.savez(os.path.join(a.out_dir, f"{name}.npz"), **d, cond=name, rot_deg=rot or 0,
                 measured_bend=b, **{f"gen_{k}": v for k, v in p.items()}, clamp_limits=clamp)
        print(f"[shift] {name:<12} pose_scale {p['pose_scale']:.3f} shape {p['shape_scale']} scale {p['scale_scale']} "
              f"rot {rot or 0} clamp {clamp} -> measured median bend {b:.1f} deg", flush=True)

    for j, name in enumerate(REAL_FACTORS):
        d = real_param_set(name, 778000 + j, dev)
        b = bend_stat(d["joints"])
        np.savez(os.path.join(a.out_dir, f"{name}.npz"), **d, cond=name, rot_deg=0, measured_bend=b,
                 factors=np.array(REAL_FACTORS[name]))
        print(f"[shift] {name:<12} real factors {REAL_FACTORS[name]} (n={len(d['verts'])}) "
              f"-> measured median bend {b:.1f} deg", flush=True)


if __name__ == "__main__":
    main()
