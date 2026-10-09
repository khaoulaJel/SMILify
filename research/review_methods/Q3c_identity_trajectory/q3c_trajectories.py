"""Q3c: identity, geometry and skeleton through the five optimisation stages (PREREGISTRATION.md).

Synthetic (S1 fits on P48, true labels) and real (JAB arms, expert-skeleton proxy labels). Existing
fits only; each loaded through JAB's model_joints.load_fit (refuses non-reproducing fits).
Writes out/Q3c_synthetic.json, out/Q3c_real.json and prints the pre-registered tests.
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
sys.path.insert(0, os.path.join(HERE, "..", "Q3a_cse_transfer_case_study"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "joint_alignment_benchmark", "tools"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import anatomy_proxy as ap  # noqa: E402
import geodesic_full as gf  # noqa: E402
import model_joints as MJ  # noqa: E402
import synthetic_skeleton_score as sss  # noqa: E402
from a2_real_correspondence import scan_samples, seg_of, EXCL, JAB  # noqa: E402

STAGES = [("H0", "_hier", "H0_body"), ("H1", "_hier", "H1_legs"), ("H2", "_hier", "H2_joint"),
          ("S2", "", "Stage_2_deform_coarse"), ("S3", "", "Stage_3_deform_fine")]
SYN_RUNS = "/hpcwork/nao48500/review_methods/Q3b_S1/runs"
SYN_ARMS = ["prod", "full_rho0.0", "full_rho0.2", "full_rho0.4", "hier_rho0.0", "hier_rho0.2", "hier_rho0.4"]
REAL_RUNS = "/hpcwork/nao48500/jab_runs"
REAL_ARMS = ["A_prod", "I_cse", "J_nooffnorm"]


def chamfer(A, B):
    d1, _ = cKDTree(B).query(A); d2, _ = cKDTree(A).query(B)
    return float((d1 ** 2).mean() + (d2 ** 2).mean())


def synthetic():
    dd = ap.load_dd()
    D, sa = gf.load(dd)
    F = np.asarray(dd["f"]).astype(np.int64)
    legc = ap.coarse_vec(ap.template_vertex_labels(dd), "leg")
    z = np.load(os.path.join(REPO, "diagnostics/moonshot/synth_power48/ground_truth.npz"))
    rng = np.random.default_rng(0)
    samples = {}
    for i, s in enumerate(z["names"]):                         # fixed labelled samples per specimen
        V = z["verts"][i].astype(np.float64); c = V.mean(0); sc = np.abs(V - c).max(); Vn = (V - c) / sc
        a = np.linalg.norm(np.cross(Vn[F[:, 1]] - Vn[F[:, 0]], Vn[F[:, 2]] - Vn[F[:, 0]]), axis=1)
        fi = rng.choice(len(F), 4000, p=a / a.sum()); b = rng.dirichlet([1, 1, 1], 4000)
        samples[str(s)] = ((b[:, :, None] * Vn[F[fi]]).sum(1), F[fi][np.arange(4000), b.argmax(1)], Vn)
    out = {}
    for arm in SYN_ARMS:
        for st, sub, fn in STAGES:
            acc = {}
            for seed in (0, 1, 2):
                p = os.path.join(SYN_RUNS, f"{arm}_s{seed}{sub}", f"{fn}.npz")
                sk = sss.score_fit(p)                          # joint error, reload-verified
                zz = np.load(p, allow_pickle=True)
                labs = [MJ.clean_label(x) for x in zz["labels"]]
                for k, sid in enumerate(labs):
                    P, tv, Vn = samples[sid]
                    _, m = cKDTree(zz["verts"][k]).query(P)
                    leg = legc[tv] != "other"
                    r = acc.setdefault(sid, {"I": [], "geo": [], "G": [], "S": []})
                    r["I"].append(float((legc[m][leg] == legc[tv][leg]).mean()))
                    r["geo"].append(float(np.median(gf.normalised_error(D, sa, tv, m))))
                    r["G"].append(chamfer(zz["verts"][k], Vn))
                    r["S"].append(sk[sid]["med"])
            out[(arm, st)] = acc
            print(f"[syn] {arm:<12} {st}: I {np.mean([np.mean(v['I']) for v in acc.values()]):.3f}  "
                  f"S {100*np.median([np.mean(v['S']) for v in acc.values()]):.2f}%L  "
                  f"G {np.median([np.mean(v['G']) for v in acc.values()]):.5f}", flush=True)
    return out


def real():
    dd = ap.load_dd()
    names, parent = ap.joint_tree(dd)
    vlab = ap.template_vertex_labels(dd)
    legc = ap.coarse_vec(vlab, "leg")
    gt = json.load(open(os.path.join(JAB, "gt_joints_fitframe.json")))
    import pandas as pd
    S = pd.read_csv(os.path.join(JAB, "scores_specimen.csv"))
    prox = {}
    for sid in gt:
        J = {k: np.asarray(v["fit"], float) for k, v in gt[sid]["joints"].items()}
        P = scan_samples(sid)
        lab, _ = ap.label_points(P, J, parent, exclude=EXCL)
        keep = np.isin([seg_of(n) for n in lab], ("tr", "fe", "ti", "ta"))
        prox[sid] = (P, ap.coarse_vec(lab, "leg"), keep)
    out = {}
    for arm in REAL_ARMS:
        for st, sub, fn in STAGES:
            acc = {}
            for seed in (0, 1, 2):
                p = os.path.join(REAL_RUNS, f"{arm}_s{seed}{sub}", f"{fn}.npz")
                zz = np.load(p, allow_pickle=True)
                labs = [MJ.clean_label(x) for x in zz["labels"]]
                MJ.load_fit(p, set(labs))                      # reproduction guard
                for k, sid in enumerate(labs):
                    P, pl, keep = prox[sid]
                    _, m = cKDTree(zz["verts"][k]).query(P[keep])
                    r = acc.setdefault(sid, {"I": [], "G": [], "S": []})
                    r["I"].append(float((legc[m] == pl[keep]).mean()))
                    r["G"].append(chamfer(zz["verts"][k], P))
                    q = S[(S.arm == arm) & (S.seed == seed) & (S.stage == fn) & (S.sid == sid)]
                    r["S"].append(float(q.med_fk.iloc[0]))
            out[(arm, st)] = acc
            print(f"[real] {arm:<12} {st}: I {np.mean([np.mean(v['I']) for v in acc.values()]):.3f}  "
                  f"S {np.median([np.mean(v['S']) for v in acc.values()]):.1f}%WL  "
                  f"G {np.median([np.mean(v['G']) for v in acc.values()]):.5f}", flush=True)
    return out


def tests(res, arms, tag):
    rep = {}
    for arm in arms:
        for (a, b) in (("H0", "H1"), ("H1", "H2"), ("H2", "S2"), ("S2", "S3"), ("H2", "S3")):
            A, B = res[(arm, a)], res[(arm, b)]
            sids = sorted(A)
            dI = np.array([np.mean(B[s]["I"]) - np.mean(A[s]["I"]) for s in sids])
            dG = np.array([np.mean(B[s]["G"]) - np.mean(A[s]["G"]) for s in sids])
            dS = np.array([np.mean(B[s]["S"]) - np.mean(A[s]["S"]) for s in sids])
            sdI = np.array([np.std(B[s]["I"]) for s in sids])
            ev = int(((dI < -sdI) & (dG < 0)).sum())
            pI = stats.binomtest(int((dI < 0).sum()), len(dI)).pvalue
            pG = stats.binomtest(int((dG < 0).sum()), len(dG)).pvalue
            rep[f"{arm}|{a}->{b}"] = dict(dI_mean=float(dI.mean()), I_down=int((dI < 0).sum()), pI=float(pI),
                                         G_down=int((dG < 0).sum()), pG=float(pG), dS_mean=float(dS.mean()),
                                         erosion_events=ev, n=len(sids))
            print(f"  [{tag}] {arm:<12} {a}->{b}: dI {dI.mean():+.3f} (down {(dI < 0).sum()}/{len(dI)}, p {pI:.2g})  "
                  f"chamfer down {(dG < 0).sum()}/{len(dG)} (p {pG:.2g})  dS {dS.mean():+.4f}  erosion events {ev}", flush=True)
    return rep


def ser(res):
    return {f"{a}|{s}": v for (a, s), v in res.items()}


def main():
    syn = synthetic()
    rs = tests(syn, SYN_ARMS, "syn")
    json.dump(dict(trajectories=ser(syn), tests=rs), open(os.path.join(HERE, "out", "Q3c_synthetic.json"), "w"), indent=1)
    rl = real()
    rr = tests(rl, REAL_ARMS, "real")
    a3 = {s: np.mean(v["I"]) for s, v in rl[("A_prod", "S3")].items()}
    j3 = {s: np.mean(v["I"]) for s, v in rl[("J_nooffnorm", "S3")].items()}
    d = np.array([j3[s] - a3[s] for s in sorted(a3)])
    pj = stats.binomtest(int((d < 0).sum()), len(d)).pvalue
    print(f"\nArm-J prediction: J identity lower than A_prod at S3 on {(d < 0).sum()}/{len(d)} (sign p {pj:.3g}), "
          f"mean diff {d.mean():+.3f}")
    json.dump(dict(trajectories=ser(rl), tests=rr, armJ=dict(lower=int((d < 0).sum()), n=len(d), p=float(pj), mean=float(d.mean()))),
              open(os.path.join(HERE, "out", "Q3c_real.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
