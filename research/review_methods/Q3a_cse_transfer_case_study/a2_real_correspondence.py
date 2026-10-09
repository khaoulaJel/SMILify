"""Q3a / A2 on real scans: how anatomically right is (a) the CSE target the fitter received, and
(b) the correspondence implied by each fit, judged against expert-skeleton proxy labels?

Resolutions and segments follow the synthetic gate (out/A2_gate_synthetic.json): leg identity
(which leg, which side) and leg segment on tr / fe / ti / ta, with pt merged into ta (the proxy
always reads pretarsus as tarsus) and coxa EXCLUDED (proxy accuracy 55% there).

(a) CSE targets: for each template vertex v kept by the cycle filter, label the target position
    with the proxy and compare to v's template label.
(b) Fit-implied correspondence: for each fitted vertex v, take the nearest scan sample (the point
    the fit has effectively put v on) and compare its proxy label with v's template label. Vertices
    further than TAU from the scan are skipped and counted (they are not on the scan at all).

Frames: scan normalised as (V - mean) / max|V - mean| (JAB score.py, fitter load_meshes); expert
joints use JAB's `fit` coordinates in that frame. Checked numerically: CSE targets must lie on the
normalised scan (median nearest-sample distance reported and asserted small).
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import anatomy_proxy as ap  # noqa: E402

JAB = os.path.join(REPO, "diagnostics", "joint_alignment_benchmark", "data")
RUNS = "/hpcwork/nao48500/jab_runs"
MESHDIR = "/hpcwork/nao48500/jab_stage"
EXCL = {"w_1_r", "w_2_r", "w_1_l", "w_2_l", "b_h"}
SEGS = ("tr", "fe", "ti", "ta")
TAU = 0.02                      # normalised units; the scan's half-extent is 1
N_SCAN = 60000


def read_obj(p):
    V, F = [], []
    for L in open(p):
        if L.startswith("v "):
            V.append([float(x) for x in L.split()[1:4]])
        elif L.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in L.split()[1:4]])
    return np.asarray(V), np.asarray(F)


def scan_samples(sid):
    V, F = read_obj(os.path.join(MESHDIR, f"{sid}_processed.obj"))
    c = V.mean(0)
    V = (V - c) / np.abs(V - c).max()
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    rng = np.random.default_rng(0)
    fi = rng.choice(len(F), N_SCAN, p=a / a.sum())
    b = rng.dirichlet([1, 1, 1], N_SCAN)
    return (b[:, :, None] * V[F[fi]]).sum(1)


def seg_of(name):
    s = str(name).split("_")[2] if str(name).startswith("l_") else None
    return "ta" if s == "pt" else s


def score(template_names, proxy_names):
    """Agreement on leg vertices of segments SEGS. Returns dict of rates and error types."""
    t_seg = np.array([seg_of(n) for n in template_names], dtype=object)
    keep = np.isin(t_seg, SEGS)
    if keep.sum() == 0:
        return None
    t = template_names[keep]
    p = proxy_names[keep]
    t_leg, p_leg = ap.coarse_vec(t, "leg"), ap.coarse_vec(p, "leg")
    p_seg = np.array([seg_of(n) for n in p], dtype=object)
    same_leg = t_leg == p_leg
    side_t = np.array([x[-1] for x in t_leg])
    num_t = np.array([x[1] for x in t_leg])
    side_p = np.array([x[-1] if x != "other" else "-" for x in p_leg])
    num_p = np.array([x[1] if x != "other" else "-" for x in p_leg])
    out = dict(n=int(keep.sum()),
               leg_acc=float(same_leg.mean()),
               seg_acc=float((same_leg & (t_seg[keep] == p_seg)).mean()),
               err_wrong_side=float(((p_leg != "other") & (side_p != side_t) & (num_p == num_t)).mean()),
               err_wrong_legnum=float(((p_leg != "other") & (num_p != num_t)).mean()),
               err_nonleg=float((p_leg == "other").mean()))
    for s in SEGS:
        m = t_seg[keep] == s
        out[f"leg_acc_{s}"] = float(same_leg[m].mean()) if m.any() else None
    return out


def main():
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    dd = ap.load_dd()
    names, parent = ap.joint_tree(dd)
    vlab = ap.template_vertex_labels(dd)
    gt = json.load(open(os.path.join(JAB, "gt_joints_fitframe.json")))
    cse = np.load(os.path.join(RUNS, "inputs", "cse_all.npz"), allow_pickle=True)
    cse_names = [ap_clean(n) for n in cse["names"]]

    rows = {}
    for sid in sorted(gt):
        joints = {k: np.asarray(v["fit"], float) for k, v in gt[sid]["joints"].items()}
        S = scan_samples(sid)
        tree = cKDTree(S)
        prox_scan, _ = ap.label_points(S, joints, parent, exclude=EXCL)
        r = {"n_expert_joints": len(joints)}

        # (a) CSE targets
        i = cse_names.index(sid)
        m = cse["mask"][i].astype(bool)
        T = cse["verts"][i][m]
        d_on, _ = tree.query(T)
        r["cse_target_to_scan_median"] = float(np.median(d_on))
        assert np.median(d_on) < 0.01, f"{sid}: CSE targets are not on the normalised scan " \
                                        f"(median {np.median(d_on):.3f}) -- frame mismatch"
        pT, _ = ap.label_points(T, joints, parent, exclude=EXCL)
        r["cse"] = score(vlab[m], pT)
        r["cse_coverage"] = float(m.mean())

        # (b) fit-implied correspondence, per arm / stage / seed
        for arm in ("A_prod", "I_cse"):
            for stage, sub, fn in (("H2", "_hier", "H2_joint.npz"), ("S3", "", "Stage_3_deform_fine.npz")):
                accs = []
                for seed in (0, 1, 2):
                    z = np.load(os.path.join(RUNS, f"{arm}_s{seed}{sub}", fn), allow_pickle=True)
                    labs = [ap_clean(x) for x in z["labels"]]
                    Vf = z["verts"][labs.index(sid)]
                    d, k = tree.query(Vf)
                    on = d < TAU
                    s = score(vlab[on], prox_scan[k[on]])
                    s["frac_off_scan"] = float(1 - on.mean())
                    accs.append(s)
                r[f"{arm}_{stage}"] = {key: float(np.mean([a[key] for a in accs if a[key] is not None]))
                                       for key in accs[0] if accs[0][key] is not None}
        rows[sid] = r
        c = r["cse"]
        print(f"{sid[:14]:<14} CSE targets leg {c['leg_acc']:.2f} seg {c['seg_acc']:.2f} "
              f"(side {c['err_wrong_side']:.2f} num {c['err_wrong_legnum']:.2f} nonleg {c['err_nonleg']:.2f}) | "
              f"fit leg_acc H2 prod {r['A_prod_H2']['leg_acc']:.2f} cse {r['I_cse_H2']['leg_acc']:.2f} | "
              f"S3 prod {r['A_prod_S3']['leg_acc']:.2f} cse {r['I_cse_S3']['leg_acc']:.2f}", flush=True)

    json.dump(rows, open(os.path.join(HERE, "out", "A2_real_correspondence.json"), "w"), indent=1)


def ap_clean(lab):
    s = str(lab)
    for suf in ("_processed.obj", ".obj", "_processed"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


if __name__ == "__main__":
    main()
