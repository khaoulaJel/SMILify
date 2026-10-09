"""Q2 readout (PREREGISTRATION.md endpoints and decision rule).

S: per-specimen median FK joint error / body-axis length L, overall and by JAB region; GT joints mapped
   into the fitter frame by the mesh's own similarity (exact). Fits reload-verified (JAB loader).
C: post-fit correspondence: 4000 area-sampled points per GT mesh carrying their true dominant-joint
   label; each matched to the nearest FITTED vertex; leg-level accuracy on leg points and median
   geodesic error / sqrt(area) (common/geodesic_full.py, validated).
G: chamfer (symmetric, 10k samples).
Gap fraction per O-arm: (B1 - O) / (B1 - O_all) of the seed-mean specimen error, with a specimen
bootstrap 95% CI; paired tests vs O0. Decision rules applied verbatim and printed.
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
import geodesic_full as gf  # noqa: E402
import model_joints as MJ  # noqa: E402
from score import EXCL, region  # noqa: E402

O = "/hpcwork/nao48500/review_methods/Q2"
ARMS = ["B0", "B1", "O0", "O-part", "O-corr", "O-joints", "O-pose", "O-shape", "O-all"]
AXIS = ["b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]


def fit_path(regime, arm, s):
    return os.path.join(O, regime, "runs", f"{arm}_s{s}", "Stage_3_deform_fine.npz")


def score_arm(regime, arm, gt, dd, D, sa):
    names = MJ.joint_names()
    use = [k for k, n in enumerate(names) if n not in EXCL]
    regs = np.array([region(names[k]) for k in use])
    ax = [names.index(n) for n in AXIS]
    F = np.asarray(dd["f"]).astype(np.int64)
    vlab = ap.template_vertex_labels(dd)
    legc = ap.coarse_vec(vlab, "leg")
    per_seed = []
    for s in (0, 1, 2):
        p = fit_path(regime, arm, s)
        z = np.load(p, allow_pickle=True)
        labs = [MJ.clean_label(x) for x in z["labels"]]
        fits = MJ.load_fit(p, set(labs))
        rng = np.random.default_rng(0)
        rows = {}
        for sid in labs:
            i = list(gt["names"]).index(sid)
            V = gt["verts"][i].astype(np.float64); c = V.mean(0); sc = np.abs(V - c).max()
            Jg = (gt["joints"][i] - c) / sc
            L = sum(np.linalg.norm(Jg[ax[k + 1]] - Jg[ax[k]]) for k in range(len(ax) - 1))
            e = np.linalg.norm(fits[sid]["FK"][use] - Jg[use], axis=1) / L
            r = dict(med=float(np.median(e)), mean=float(e.mean()))
            for rg in np.unique(regs):
                r[f"med_{rg}"] = float(np.median(e[regs == rg]))
            Vn = (V - c) / sc
            a = np.linalg.norm(np.cross(Vn[F[:, 1]] - Vn[F[:, 0]], Vn[F[:, 2]] - Vn[F[:, 0]]), axis=1)
            fi = rng.choice(len(F), 4000, p=a / a.sum())
            b = rng.dirichlet([1, 1, 1], 4000)
            P = (b[:, :, None] * Vn[F[fi]]).sum(1)
            true_v = F[fi][np.arange(4000), b.argmax(1)]
            _, match = cKDTree(fits[sid]["verts"]).query(P)
            leg = legc[true_v] != "other"
            r["corr_leg_acc"] = float((legc[match][leg] == legc[true_v][leg]).mean())
            r["corr_geo_med"] = float(np.median(gf.normalised_error(D, sa, true_v, match)))
            Pf = fits[sid]["verts"]
            d1, _ = cKDTree(Vn).query(Pf); d2, _ = cKDTree(Pf).query(Vn)
            r["chamfer"] = float((d1 ** 2).mean() + (d2 ** 2).mean())
            rows[sid] = r
        per_seed.append(rows)
    sids = sorted(per_seed[0])
    keys = list(per_seed[0][sids[0]])
    return sids, {k: np.array([np.mean([ps[sid][k] for ps in per_seed]) for sid in sids]) for k in keys}, \
        float(np.mean(np.std([[ps[sid]["med"] for sid in sids] for ps in per_seed], axis=0)))


def boot_frac(b1, o, oall, B=5000, rng=np.random.default_rng(1)):
    idx = rng.integers(0, len(b1), (B, len(b1)))
    num = b1[idx].mean(1) - o[idx].mean(1)
    den = b1[idx].mean(1) - oall[idx].mean(1)
    f = num / np.where(np.abs(den) < 1e-12, np.nan, den)
    return float(np.nanpercentile(f, 2.5)), float(np.nanpercentile(f, 97.5))


def main():
    dd = ap.load_dd()
    D, sa = gf.load(dd)
    out = {}
    for regime in ("REALPOSE", "REALSCALE"):
        gt = np.load(os.path.join(O, regime, "meshes", "ground_truth.npz"), allow_pickle=True)
        res = {}
        for arm in ARMS:
            sids, m, seed_sd = score_arm(regime, arm, gt, dd, D, sa)
            res[arm] = m
            print(f"{regime:<10} {arm:<9} joint {100*np.median(m['med']):5.2f}% L (seed SD {100*seed_sd:.2f})  "
                  f"distal {100*np.median(m['med_leg_distal']):5.2f}  corr_leg {np.mean(m['corr_leg_acc']):.3f}  "
                  f"geo {np.median(m['corr_geo_med']):.3f}  chamfer {np.median(m['chamfer']):.5f}", flush=True)
        b1, oall = res["B1"]["med"], res["O-all"]["med"]
        gap = {}
        print(f"\n{regime}: fraction of the B1 -> O-all gap closed (mean over specimens), 95% CI; vs O0 paired")
        for arm in ("O-part", "O-corr", "O-joints", "O-pose", "O-shape"):
            o = res[arm]["med"]
            frac = float((b1.mean() - o.mean()) / (b1.mean() - oall.mean()))
            ci = boot_frac(b1, o, oall)
            d = o - res["O0"]["med"]
            sg = stats.binomtest(int((d < 0).sum()), len(d)).pvalue
            gap[arm] = dict(frac=frac, ci95=ci, better_than_O0=int((d < 0).sum()), n=len(d), sign_p=float(sg),
                            wil_p=float(stats.wilcoxon(d).pvalue))
            print(f"  {arm:<9} {100*frac:6.1f}%  [{100*ci[0]:.1f}, {100*ci[1]:.1f}]   better than O0 on "
                  f"{(d < 0).sum()}/{len(d)} (sign {sg:.2g})")
        o0d = res["O0"]["med"] - b1
        print(f"  O0 control: mean change vs B1 {100*o0d.mean():+.3f}% L, {(o0d < 0).sum()}/{len(o0d)} lower")
        f = {k: v["frac"] for k, v in gap.items()}
        rules = {
            "M01": (f["O-joints"] >= 0.4 or f["O-pose"] >= 0.4) and max(f["O-joints"], f["O-pose"]) > f["O-part"],
            "M05": f["O-pose"] >= 0.4,
            "M04_oracle_part_or_corr": f["O-part"] >= 0.4 or f["O-corr"] >= 0.4,
            "M02_oracle_corr": f["O-corr"] >= 0.4,
        }
        print(f"  decision inputs (rules need S1 too for M04/M02; S1: H-absorb NOT SUPPORTED as registered): {rules}")
        out[regime] = dict(gap=gap, rules_oracle_part=rules,
                           arms={a: {k: v.tolist() for k, v in m.items()} for a, m in res.items()})
    json.dump(out, open(os.path.join(HERE, "out", "Q2_readout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
