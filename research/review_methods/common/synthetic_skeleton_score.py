"""Family-S (skeleton) and free-channel scoring of synthetic P48 fits against ground truth.

Joint error = |FK pivot (fit) - FK pivot (GT)| / L, L = GT body-axis length (sum of b_t -> b_a_1 ->
... -> b_a_5 bone lengths; DEVIATIONS D1.6), per joint, summarised per specimen by median (verdict)
and mean (MPJPE comparability, D1.1), by JAB region. Excluded joints as JAB.

Fits are loaded with JAB's `model_joints.load_fit` (refuses a fit whose regenerated vertices differ
from the saved ones). GT joints are rebuilt through the model with a vertex round-trip assert
(`Q3a.../a2_gate_synthetic.gt_joints_p48`). Fit and GT are compared in the SAME frame: fits live in
the fitter's normalised target frame, so GT joints are mapped with the same per-specimen centre and
scale (vertex mean, max abs coordinate of the GT mesh).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "joint_alignment_benchmark", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "Q3a_cse_transfer_case_study"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import model_joints as mj  # noqa: E402
from score import region  # noqa: E402  (JAB's region definition)

EXCL = {"w_1_r", "w_2_r", "w_1_l", "w_2_l", "b_h"}
AXIS = ["b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]
_GT = None


def gt_p48():
    """{stem: dict(J=(55,3) in normalised target frame, L=body-axis length in that frame)}"""
    global _GT
    if _GT is None:
        from a2_gate_synthetic import gt_joints_p48
        z, J, _ = gt_joints_p48()
        names = mj.joint_names()
        ax = [names.index(n) for n in AXIS]
        _GT = {}
        for i, s in enumerate(z["names"]):
            V = z["verts"][i].astype(np.float64)
            c = V.mean(0)
            sc = np.abs(V - c).max()
            Jn = (J[i] - c) / sc
            L = float(sum(np.linalg.norm(Jn[ax[k + 1]] - Jn[ax[k]]) for k in range(len(ax) - 1)))
            _GT[str(s)] = dict(J=Jn, L=L)
    return _GT


def score_fit(npz_path):
    """Per specimen: median/mean joint error (fraction of L) overall and by region, + free channels."""
    gt = gt_p48()
    names = mj.joint_names()
    use = [k for k, n in enumerate(names) if n not in EXCL]
    regs = np.array([region(names[k]) for k in use])
    fits = mj.load_fit(npz_path, set(gt))
    z = np.load(npz_path, allow_pickle=True)
    labs = [mj.clean_label(x) for x in z["labels"]]
    out = {}
    for sid, f in fits.items():
        e = np.linalg.norm(f["FK"][use] - gt[sid]["J"][use], axis=1) / gt[sid]["L"]
        i = labs.index(sid)
        r = dict(med=float(np.median(e)), mean=float(np.mean(e)),
                 deform_rms=f["deform_rms"],
                 betas_trans=float(np.linalg.norm(z["betas_trans"][i], axis=-1).mean()),
                 log_scales=float(np.abs(z["log_beta_scales"][i]).mean()))
        for rg in np.unique(regs):
            r[f"med_{rg}"] = float(np.median(e[regs == rg]))
        out[sid] = r
    return out


if __name__ == "__main__":
    p = sys.argv[1]
    s = score_fit(p)
    m = np.array([v["med"] for v in s.values()])
    print(f"{p}: n={len(s)}  median joint error {100*np.median(m):.2f}% L  (mean of medians {100*m.mean():.2f}%)")
    for k in sorted(next(iter(s.values()))):
        if k.startswith("med_"):
            print(f"  {k:<18} {100*np.median([v[k] for v in s.values()]):6.2f}% L")
