"""Score every benchmark run against the fit-independent expert joints (PREREGISTRATION.md §2, §7).

Outputs (data/):
  scores_joints.csv    one row per arm x seed x stage x specimen x evaluated joint
  scores_specimen.csv  one row per arm x seed x stage x specimen (PA error, bones, side confusion,
                       chamfer, deform RMS, reproduction error)
Usage: score.py [--runs /hpcwork/nao48500/jab_runs] [--arms A_prod,...] [--legacy]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(BENCH, "..", ".."))
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
STAGES = [("hier", "H0_body"), ("hier", "H1_legs"), ("hier", "H2_joint"),
          ("moon", "Stage_2_deform_coarse"), ("moon", "Stage_3_deform_fine")]
EXCL = {"w_1_r", "w_2_r", "w_1_l", "w_2_l", "b_h"}
MODEL_OF = {"E_anteriorlimits": "3D_model_prep/OmniAnt_25PCs_anterior_limited.pkl"}
DEFAULT_MODEL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"


def region(n):
    if n.startswith("b_"):
        return "body_axis"
    if n.startswith("ma_"):
        return "mandible"
    if n.startswith("an_"):
        return "antenna"
    seg = n.split("_")[2]
    return {"co": "coxa", "tr": "leg_proximal", "fe": "leg_proximal",
            "ti": "leg_distal", "ta": "leg_distal", "pt": "leg_distal"}[seg]


def contralateral(n):
    return n[:-1] + ("l" if n.endswith("_r") else "r") if n.endswith(("_r", "_l")) else None


def local_frame(P):
    """Specimen-local anatomical frame from the GT: anterior, dorsal, right (orthonormal)."""
    g = lambda ks: np.mean([P[k] for k in ks if k in P], 0)
    head = [k for k in ("ma_r", "ma_l", "an_1_r", "an_1_l") if k in P]
    ant = (g(head) if head else P["b_t"]) - g(["b_a_3", "b_a_4", "b_a_5"])
    ant /= np.linalg.norm(ant)
    body = ["b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]
    dors = g(body) - g([k for k in P if k.startswith("l_") and k[4:6] in ("ta", "pt")])
    dors -= ant * dors.dot(ant)
    dors /= np.linalg.norm(dors)
    return np.stack([ant, dors, np.cross(ant, dors)])


def sim_align(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    U, S, Vt = np.linalg.svd((dst - cd).T @ (src - cs))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = np.trace(np.diag(S) @ E) / ((src - cs) ** 2).sum()
    return (src - cs) @ R.T * s + cd


def read_obj(p):
    V, F = [], []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
        elif L.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in L.split()[1:4]])
    return np.asarray(V), np.asarray(F)


def sample_surface(V, F, n, rng):
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    f = rng.choice(len(F), n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n)
    m = u + v > 1
    u[m], v[m] = 1 - u[m], 1 - v[m]
    return V[F[f, 0]] * (1 - u - v)[:, None] + V[F[f, 1]] * u[:, None] + V[F[f, 2]] * v[:, None]


_TARGETS = {}


def target_samples(sid, n=20000):
    if sid not in _TARGETS:
        V, F = read_obj(os.path.join(MESHDIR, f"{sid}_processed.obj"))
        c = V.mean(0)
        V = (V - c) / np.abs(V - c).max()
        _TARGETS[sid] = sample_surface(V, F, n, np.random.default_rng(0))
    return _TARGETS[sid]


def chamfer(verts, faces, sid, n=20000):
    P = sample_surface(verts, faces, n, np.random.default_rng(1))
    T = target_samples(sid, n)
    d1, _ = cKDTree(T).query(P)
    d2, _ = cKDTree(P).query(T)
    return float((d1 ** 2).mean() + (d2 ** 2).mean())


def score_npz(npz, arm, seed, stage, G, names, parents, joint_rows, spec_rows):
    sys.path.insert(0, HERE)
    import model_joints as MJ
    fits = MJ.load_fit(npz, wanted=set(G))
    faces = np.load(npz, allow_pickle=True)["faces"][0]
    for sid, f in fits.items():
        g = G[sid]
        wl = g["WL_fit"]
        P = {n: np.asarray(r["fit"]) for n, r in g["joints"].items()}
        use = [n for n in names if n in P and n not in EXCL]
        Fr = local_frame(P)
        gt = np.array([P[n] for n in use])
        for jdef in ("FK", "REG", "SKIN"):
            f[jdef + "_use"] = np.array([f[jdef][names.index(n)] for n in use])
        err = {j: np.linalg.norm(f[j + "_use"] - gt, axis=1) / wl * 100 for j in ("FK", "REG", "SKIN")}
        signed = (f["FK_use"] - gt) @ Fr.T / wl * 100
        for k, n in enumerate(use):
            joint_rows.append(dict(arm=arm, seed=seed, stage=stage, sid=sid, joint=n, region=region(n),
                                   err_fk=err["FK"][k], err_reg=err["REG"][k], err_skin=err["SKIN"][k],
                                   d_ant=signed[k, 0], d_dors=signed[k, 1], d_right=signed[k, 2]))
        pa = np.linalg.norm(sim_align(f["FK_use"], gt) - gt, axis=1) / wl * 100
        # bones: parent->child segments with both ends placed and evaluated
        bone_err = []
        for n in use:
            p = parents[names.index(n)]
            if p >= 0 and names[p] in use:
                lg = np.linalg.norm(P[n] - P[names[p]])
                lf = np.linalg.norm(f["FK"][names.index(n)] - f["FK"][p])
                if lg > 1e-9:
                    bone_err.append(abs(lf - lg) / lg * 100)
        # side confusion: model joint closer to contralateral GT than ipsilateral GT
        conf, nb = 0, 0
        for n in use:
            c = contralateral(n)
            if c and c in P:
                x = f["FK"][names.index(n)]
                conf += np.linalg.norm(x - P[c]) < np.linalg.norm(x - P[n])
                nb += 1
        body = [k for k, n in enumerate(use) if n.startswith("b_")]
        csize = np.sqrt(((gt[body] - gt[body].mean(0)) ** 2).sum(1).mean()) if len(body) > 2 else np.nan
        spec_rows.append(dict(arm=arm, seed=seed, stage=stage, sid=sid, tier=g["tier"], n_joints=len(use),
                              med_fk=float(np.median(err["FK"])), mean_fk=float(np.mean(err["FK"])),
                              med_reg=float(np.median(err["REG"])), med_skin=float(np.median(err["SKIN"])), med_pa=float(np.median(pa)),
                              med_fk_csize=float(np.median(err["FK"] * wl / 100 / csize * 100)),
                              bone_med=float(np.median(bone_err)) if bone_err else np.nan,
                              side_confusion=conf / max(nb, 1),
                              chamfer=chamfer(f["verts"], faces, sid),
                              deform_rms=f["deform_rms"], repro_err=f["repro_err"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/hpcwork/nao48500/jab_runs")
    ap.add_argument("--arms", default="")
    ap.add_argument("--legacy", action="store_true", help="also score the pre-existing Z8 production fit")
    ap.add_argument("--tag", default="")
    ap.add_argument("--_worker", default="")
    a = ap.parse_args()
    G = json.load(open(os.path.join(BENCH, "data/gt_joints_fitframe.json")))

    if a._worker:        # one model file per process (SMILIFY_SMAL_FILE is read at import time)
        import pickle
        sys.path.insert(0, HERE)
        import model_joints as MJ
        names = MJ.joint_names()
        with open(os.path.join(REPO, os.environ["SMILIFY_SMAL_FILE"]), "rb") as fh:
            u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
        parents = np.asarray(dd["kintree_table"])[0].astype(int)
        jobs = json.loads(a._worker)
        jr, sr = [], []
        for arm, seed, stage, npz in jobs:
            print(f"  scoring {arm} s{seed} {stage}", flush=True)
            score_npz(npz, arm, seed, stage, G, names, parents, jr, sr)
        pd.DataFrame(jr).to_csv(a.tag + "_joints.csv", index=False)
        pd.DataFrame(sr).to_csv(a.tag + "_spec.csv", index=False)
        return

    jobs = {}
    for d in sorted(glob.glob(os.path.join(a.runs, "*_s[0-9]"))):
        arm, seed = os.path.basename(d).rsplit("_s", 1)
        if a.arms and arm not in a.arms.split(","):
            continue
        for kind, st in STAGES:
            p = os.path.join(d + ("_hier" if kind == "hier" else ""), st + ".npz")
            if os.path.exists(p):
                jobs.setdefault(MODEL_OF.get(arm, DEFAULT_MODEL), []).append((arm, int(seed), st, p))
            elif st == "Stage_3_deform_fine":
                print(f"!! MISSING final stage for {arm} seed {seed} -- reported as failed run")
    if a.legacy:
        for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/Stage_3_deform_fine.npz"))):
            jobs.setdefault(DEFAULT_MODEL, []).append(("Z8_legacy_prod", 0, "Stage_3_deform_fine", p))
    J, S = [], []
    for model, jl in jobs.items():
        tag = os.path.join(BENCH, "data", f"_tmp_{os.path.basename(model)}")
        env = dict(os.environ, SMILIFY_SMAL_FILE=model, SMILIFY_COUPLE_JOINT_BLENDSHAPES="0")
        subprocess.run([sys.executable, __file__, "--_worker", json.dumps(jl), "--tag", tag],
                       check=True, env=env, cwd=REPO)
        J.append(pd.read_csv(tag + "_joints.csv")); S.append(pd.read_csv(tag + "_spec.csv"))
        os.remove(tag + "_joints.csv"); os.remove(tag + "_spec.csv")
    J, S = pd.concat(J), pd.concat(S)
    # legacy Z8 spans several chunk files: keep each specimen once
    S = S.drop_duplicates(["arm", "seed", "stage", "sid"])
    J = J.drop_duplicates(["arm", "seed", "stage", "sid", "joint"])
    J.to_csv(os.path.join(BENCH, "data/scores_joints.csv"), index=False)
    S.to_csv(os.path.join(BENCH, "data/scores_specimen.csv"), index=False)
    print(S[S.stage == "Stage_3_deform_fine"].groupby("arm")[["med_fk", "med_pa", "chamfer"]].median())


if __name__ == "__main__":
    main()
