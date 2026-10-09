"""Q3d readout against PREREGISTRATION.md.

Final-stage (S3) per-specimen seed-mean: joint error / body-axis length (reload-verified fits),
fit-implied identity with true labels (Q3c definition), chamfer. lambda 0 = S1 `prod`; lambda 1 x
{rho0.0, rho0.2, rho0.4} = S1 `full_*` (same code state); everything else from Q3d runs.
"""
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "joint_alignment_benchmark", "tools"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import anatomy_proxy as ap  # noqa: E402
import model_joints as MJ  # noqa: E402
import synthetic_skeleton_score as sss  # noqa: E402

S1 = "/hpcwork/nao48500/review_methods/Q3b_S1/runs"
Q3D = "/hpcwork/nao48500/review_methods/Q3d/runs"
QUALS = ["rho0.0", "rho0.2", "rho0.4", "nonleg0.2"]
LAMS = ["0", "0.1", "0.25", "0.5", "1", "2"]


def run_dir(q, lam, s):
    if lam == "0":
        return f"{S1}/prod_s{s}"
    if lam == "1" and q.startswith("rho"):
        return f"{S1}/full_{q}_s{s}"
    return f"{Q3D}/{q}_lam{lam}_s{s}"


def main():
    dd = ap.load_dd()
    F = np.asarray(dd["f"]).astype(np.int64)
    legc = ap.coarse_vec(ap.template_vertex_labels(dd), "leg")
    z = np.load(os.path.join(REPO, "diagnostics/moonshot/synth_power48/ground_truth.npz"))
    rng = np.random.default_rng(0)
    samp = {}
    for i, s in enumerate(z["names"]):
        V = z["verts"][i].astype(np.float64); c = V.mean(0); sc = np.abs(V - c).max(); Vn = (V - c) / sc
        a = np.linalg.norm(np.cross(Vn[F[:, 1]] - Vn[F[:, 0]], Vn[F[:, 2]] - Vn[F[:, 0]]), axis=1)
        fi = rng.choice(len(F), 4000, p=a / a.sum()); b = rng.dirichlet([1, 1, 1], 4000)
        samp[str(s)] = ((b[:, :, None] * Vn[F[fi]]).sum(1), F[fi][np.arange(4000), b.argmax(1)], Vn)
    res = {}
    for q in QUALS:
        for lam in LAMS:
            per = {}
            for s in (0, 1, 2):
                p = os.path.join(run_dir(q, lam, s), "Stage_3_deform_fine.npz")
                sk = sss.score_fit(p)
                zz = np.load(p, allow_pickle=True)
                labs = [MJ.clean_label(x) for x in zz["labels"]]
                for k, sid in enumerate(labs):
                    P, tv, Vn = samp[sid]
                    _, m = cKDTree(zz["verts"][k]).query(P)
                    leg = legc[tv] != "other"
                    d1, _ = cKDTree(Vn).query(zz["verts"][k]); d2, _ = cKDTree(zz["verts"][k]).query(Vn)
                    r = per.setdefault(sid, {"S": [], "I": [], "G": []})
                    r["S"].append(sk[sid]["med"]); r["I"].append(float((legc[m][leg] == legc[tv][leg]).mean()))
                    r["G"].append(float((d1 ** 2).mean() + (d2 ** 2).mean()))
            sids = sorted(per)
            res[(q, lam)] = {k: np.array([np.mean(per[s][k]) for s in sids]) for k in ("S", "I", "G")}
            m = res[(q, lam)]
            print(f"{q:<10} lam {lam:<5} joint {100*np.median(m['S']):5.2f}% L  identity {m['I'].mean():.3f}  "
                  f"chamfer {np.median(m['G']):.5f}", flush=True)

    out = {"lambda_star": {}, "tests": {}}
    print("\nPRE-REGISTERED readings")
    for q in QUALS:
        means = {lam: res[(q, lam)]["S"].mean() for lam in LAMS}
        ls = min(means, key=means.get)
        out["lambda_star"][q] = ls
        d1 = res[(q, ls)]["S"] - res[(q, "1")]["S"]
        d0 = res[(q, ls)]["S"] - res[(q, "0")]["S"]
        p1 = stats.binomtest(int((d1 < 0).sum()), len(d1)).pvalue if ls != "1" else 1.0
        p0 = stats.binomtest(int((d0 < 0).sum()), len(d0)).pvalue if ls != "0" else 1.0
        out["tests"][q] = dict(lambda_star=ls, mean_by_lambda={k: float(v) for k, v in means.items()},
                               vs_lam1_better=int((d1 < 0).sum()), p_vs_lam1=float(p1),
                               vs_lam0_better=int((d0 < 0).sum()), p_vs_lam0=float(p0))
        print(f"  {q:<10} lambda* = {ls:<5} mean joint by lambda: " +
              " ".join(f"{k}:{100*v:.2f}" for k, v in means.items()) +
              f" | vs lam1 better {int((d1 < 0).sum())}/48 p {p1:.2g} | vs lam0 better {int((d0 < 0).sum())}/48 p {p0:.2g}")
    order = [float(out["lambda_star"][q]) for q in ("rho0.0", "rho0.2", "rho0.4")]
    print(f"  ordering lambda*(correct) >= lambda*(wrong20) >= lambda*(wrong40): {order} -> "
          f"{'holds' if order[0] >= order[1] >= order[2] else 'violated'}")
    # useful window at wrong20: any lambda > 0 beating lambda 0?
    best_w = None
    for lam in LAMS[1:]:
        d = res[("rho0.2", lam)]["S"] - res[("rho0.2", "0")]["S"]
        p = stats.binomtest(int((d < 0).sum()), len(d)).pvalue
        print(f"  wrong20 lambda {lam:<5} vs lambda 0: better {int((d < 0).sum())}/48 p {p:.2g} mean d {100*d.mean():+.2f}% L")
        if (d < 0).sum() > 24 and p < 0.05:
            best_w = lam
    out["useful_window_wrong20"] = best_w
    print("  error type at 20%: wrong-leg vs non-leg, mean joint error by lambda")
    for lam in LAMS:
        print(f"    lambda {lam:<5} wrong-leg {100*res[('rho0.2', lam)]['S'].mean():.2f}  non-leg {100*res[('nonleg0.2', lam)]['S'].mean():.2f}")
    out["table"] = {f"{q}|{l}": {k: v.tolist() for k, v in m.items()} for (q, l), m in res.items()}
    json.dump(out, open(os.path.join(HERE, "out", "Q3d_readout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
