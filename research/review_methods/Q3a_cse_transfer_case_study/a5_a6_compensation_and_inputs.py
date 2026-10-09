"""Q3a / A5 + A6.

A5 (H-pivot): between H2 and S3, how much does each arm change the free surface channels versus the
skeleton? Per specimen, seed-mean, for A_prod and I_cse:
  - deform_rms           free-form per-vertex offsets (S3 only; H2 has none)
  - |betas_trans|        per-joint translation offsets (move pivots relative to the bone)
  - |log_beta_scales|    per-joint scale residuals
  - pose change          mean geodesic angle between H2 and S3 joint rotations
  - pivot drift          mean FK-pivot displacement H2 -> S3 (% WL), from the stored params
Read from the saved npz; FK pivots regenerated through the model with the JAB loader, which refuses
any fit that does not reproduce its saved vertices.

A6 (descriptive only, not a predictor): scan point count, connected components, bbox aspect ratios,
A_prod-fitted betas Mahalanobis norm (shape_cov), fitted pose magnitude.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "joint_alignment_benchmark", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import anatomy_proxy as ap  # noqa: E402
import model_joints as mj  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

RUNS = "/hpcwork/nao48500/jab_runs"
MESHDIR = "/hpcwork/nao48500/jab_stage"
JAB = os.path.join(REPO, "diagnostics", "joint_alignment_benchmark", "data")
EXCL = {"w_1_r", "w_2_r", "w_1_l", "w_2_l", "b_h"}


def angle(a, b):
    return np.linalg.norm((Rot.from_rotvec(a.reshape(-1, 3)).inv() * Rot.from_rotvec(b.reshape(-1, 3))).as_rotvec(), axis=1)


def main():
    gt = json.load(open(os.path.join(JAB, "gt_joints_fitframe.json")))
    sids = sorted(gt)
    names = mj.joint_names()
    use = [k for k, n in enumerate(names) if n not in EXCL]
    dd = ap.load_dd()
    cov_inv = np.linalg.inv(np.asarray(dd["shape_cov"]))
    rows = {}
    for sid in sids:
        WL = gt[sid]["WL_fit"]
        r = {}
        for arm in ("A_prod", "I_cse"):
            acc = {k: [] for k in ("deform_rms", "betas_trans", "log_scales_H2", "log_scales_S3",
                                   "pose_change_deg", "pivot_drift_pctWL", "betas_maha")}
            for seed in (0, 1, 2):
                h2 = os.path.join(RUNS, f"{arm}_s{seed}_hier", "H2_joint.npz")
                s3 = os.path.join(RUNS, f"{arm}_s{seed}", "Stage_3_deform_fine.npz")
                fh, fs = mj.load_fit(h2, {sid})[sid], mj.load_fit(s3, {sid})[sid]
                zh, zs = np.load(h2, allow_pickle=True), np.load(s3, allow_pickle=True)
                ih = [mj.clean_label(x) for x in zh["labels"]].index(sid)
                i3 = [mj.clean_label(x) for x in zs["labels"]].index(sid)
                acc["deform_rms"].append(fs["deform_rms"])
                acc["betas_trans"].append(float(np.linalg.norm(zs["betas_trans"][i3], axis=-1).mean()))
                acc["log_scales_H2"].append(float(np.abs(zh["log_beta_scales"][ih]).mean()))
                acc["log_scales_S3"].append(float(np.abs(zs["log_beta_scales"][i3]).mean()))
                acc["pose_change_deg"].append(float(np.degrees(angle(zh["joint_rot"][ih], zs["joint_rot"][i3]).mean())))
                acc["pivot_drift_pctWL"].append(float(100 * np.linalg.norm(fs["FK"][use] - fh["FK"][use], axis=1).mean() / WL))
                b = zs["betas"][i3]
                acc["betas_maha"].append(float(np.sqrt(b @ cov_inv[:len(b), :len(b)] @ b)))
            r[arm] = {k: float(np.mean(v)) for k, v in acc.items()}

        V, F = [], []
        for L in open(os.path.join(MESHDIR, f"{sid}_processed.obj")):
            if L.startswith("v "):
                V.append([float(x) for x in L.split()[1:4]])
            elif L.startswith("f "):
                F.append([int(x.split("/")[0]) - 1 for x in L.split()[1:4]])
        V, F = np.asarray(V), np.asarray(F)
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        ncomp, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))))
        sizes = np.bincount(lab)
        ext = np.sort(V.max(0) - V.min(0))[::-1]
        r["input"] = dict(n_verts=len(V), n_components=int(ncomp),
                          largest_component_frac=float(sizes.max() / len(V)),
                          aspect_mid_long=float(ext[1] / ext[0]), aspect_short_long=float(ext[2] / ext[0]),
                          n_expert_joints=len(gt[sid]["joints"]), tier=gt[sid]["tier"])
        rows[sid] = r
        a, c = r["A_prod"], r["I_cse"]
        print(f"{sid[:14]:<14} drift prod {a['pivot_drift_pctWL']:5.1f} cse {c['pivot_drift_pctWL']:5.1f} | "
              f"pose chg {a['pose_change_deg']:5.1f}/{c['pose_change_deg']:5.1f} | "
              f"deform {a['deform_rms']:.4f}/{c['deform_rms']:.4f} | btrans {a['betas_trans']:.4f}/{c['betas_trans']:.4f} | "
              f"lscale H2->S3 prod {a['log_scales_H2']:.3f}->{a['log_scales_S3']:.3f} cse {c['log_scales_H2']:.3f}->{c['log_scales_S3']:.3f} | "
              f"maha {a['betas_maha']:.1f} | comps {r['input']['n_components']} ({r['input']['largest_component_frac']:.2f})", flush=True)
    json.dump(rows, open(os.path.join(HERE, "out", "A5_A6.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
