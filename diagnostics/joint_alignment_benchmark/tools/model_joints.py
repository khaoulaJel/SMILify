"""Model joint positions from a saved fit, in the fitter's normalised target frame.

Two definitions, both pre-registered (PREREGISTRATION.md section 5.1):
  FK   forward-kinematics rotation pivots, `J_transformed + trans` -- where the rig articulates,
       including per-joint scale/translation residuals; excludes free-form `deform_verts`.
       PRIMARY: this is the skeleton the model actually has (and what an exported rig carries).
  REG  `J_regressor` applied to the final deformed vertices -- the definition used by G1-G10 and
       V5-V13. SECONDARY: continuity with earlier numbers.

Loading is verified, not trusted (CLAUDE.md "verify that a checkpoint actually loaded"): every
parameter tensor must match the npz shape exactly, and the vertices re-generated from the loaded
parameters must reproduce the saved `verts` array. A fit that does not reproduce is refused.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import torch  # noqa: E402

PARAMS = ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales", "betas_trans", "deform_verts"]
REPRO_TOL = 1e-4          # in normalised units (unit box); float32 GPU/CPU differences are ~1e-6


def clean_label(lab):
    s = str(lab)
    for suf in ("_processed.obj", ".obj", "_processed"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def load_fit(npz_path, wanted=None, device="cpu"):
    """Return {sid: dict(FK=(55,3), REG=(55,3), verts=(V,3), params)} for specimens in `wanted`."""
    from fitter_3d.trainer import SMAL3DFitter

    z = np.load(npz_path, allow_pickle=True)
    labels = [clean_label(l) for l in z["labels"]]
    idx = [i for i, s in enumerate(labels) if wanted is None or s in wanted]
    if not idx:
        return {}
    smal = SMAL3DFitter(batch_size=len(idx), device=device)
    names = [str(x) for x in np.asarray(smal.smal_model.J_names if hasattr(smal.smal_model, "J_names")
                                         else [])]
    with torch.no_grad():
        for k in PARAMS:
            src = torch.as_tensor(np.asarray(z[k], np.float32)[idx], device=device)
            dst = getattr(smal, k)
            assert tuple(dst.shape) == tuple(src.shape), (npz_path, k, dst.shape, src.shape)
            dst.copy_(src)
        verts, skin = smal(return_joints=True)
        # smal(return_joints=True) is NOT the FK pivot for this model: static_joint_locs is absent,
        # so smal_torch regresses joints from the skinned PRE-deform vertices. The pivots the rig
        # rotates about are J_transformed (validated: == J_regressor @ v_template at rest, 9e-8).
        fk = smal.smal_model.J_transformed + smal.trans.unsqueeze(1)
        verts = verts.double()
        Jr = smal.smal_model.J_regressor.double()
        reg = (torch.einsum("jv,nvc->njc", Jr, verts) if Jr.shape[1] == verts.shape[1]
               else torch.einsum("vj,nvc->njc", Jr, verts))
    saved = np.asarray(z["verts"], np.float64)[idx]
    err = np.abs(verts.cpu().numpy() - saved).max()
    assert err < REPRO_TOL, f"{npz_path}: regenerated verts differ from saved by {err:.2e}"
    out = {}
    for j, i in enumerate(idx):
        out[labels[i]] = dict(FK=fk[j].double().cpu().numpy(), REG=reg[j].cpu().numpy(),
                              SKIN=skin[j].double().cpu().numpy(),
                              verts=verts[j].cpu().numpy(), repro_err=float(err),
                              deform_rms=float(np.sqrt((np.asarray(z["deform_verts"][i]) ** 2).sum(-1).mean())),
                              source=npz_path)
    return out


def joint_names():
    import pickle
    with open(os.path.join(REPO, os.environ["SMILIFY_SMAL_FILE"]), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    return [str(x) for x in np.asarray(dd["J_names"]).ravel()]
