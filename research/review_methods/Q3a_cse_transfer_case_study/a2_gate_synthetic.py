"""Q3a / A2 validation gate: is the expert-skeleton proxy label trustworthy, and at which resolution?

On synthetic P48 the true per-point label is known (dominant skinning joint of the sampled face, the
labels.py definition). Build the proxy from the GROUND-TRUTH FK joints exactly as it will be built
from expert joints on real scans (same excluded joints as JAB: w_*, b_h), and measure agreement.

Ground-truth joints are rebuilt through the model from ground_truth.npz; the regenerated vertices
must reproduce the stored GT vertices (asserted) before any joint is trusted.

Pre-registered gate (PREREGISTRATION.md): usable at 'leg' if accuracy >= 0.85, at 'leg_seg' if
>= 0.75. 'appendage' is reported for information.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import anatomy_proxy as ap  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402

EXCL = {"w_1_r", "w_2_r", "w_1_l", "w_2_l", "b_h"}   # JAB's exclusions
GATE = {"leg": 0.85, "leg_seg": 0.75}
N_SAMPLE = 8000


def gt_joints_p48():
    z = np.load(os.path.join(REPO, "diagnostics/moonshot/synth_power48/ground_truth.npz"))
    n = len(z["verts"])
    f = SMAL3DFitter(batch_size=n, device="cpu")
    with torch.no_grad():
        for k in ("betas", "joint_rot", "log_beta_scales"):
            src = torch.as_tensor(z[k], dtype=torch.float32)
            assert tuple(getattr(f, k).shape) == tuple(src.shape), k
            getattr(f, k).copy_(src)
        for k in ("global_rot", "trans", "betas_trans", "deform_verts"):
            getattr(f, k).zero_()
        v = f()
        J = (f.smal_model.J_transformed + f.trans.unsqueeze(1)).numpy()
    err = float(np.abs(v.numpy() - z["verts"]).max())
    assert err < 1e-4, f"GT round-trip failed: regenerated verts differ by {err:.2e}"
    return z, J, err


def sample_faces(V, F, n, rng):
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    fi = rng.choice(len(F), n, p=a / a.sum())
    b = rng.dirichlet([1, 1, 1], n)
    P = (b[:, :, None] * V[F[fi]]).sum(1)
    corner = F[fi][np.arange(n), b.argmax(1)]
    return P, corner


def main():
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    dd = ap.load_dd()
    names, parent = ap.joint_tree(dd)
    vlab = ap.template_vertex_labels(dd)
    F = np.asarray(dd["f"]).astype(np.int64)
    z, J, rt = gt_joints_p48()
    print(f"GT round-trip max |dv| = {rt:.2e} (ok)")

    rng = np.random.default_rng(0)
    res = {lv: [] for lv in ("leg", "leg_seg", "appendage")}
    conf = {}
    for i in range(len(J)):
        P, corner = sample_faces(z["verts"][i].astype(float), F, N_SAMPLE, rng)
        joints = {n: J[i, k] for k, n in enumerate(names)}
        prox, _ = ap.label_points(P, joints, parent, exclude=EXCL)
        true = vlab[corner]
        for lv in res:
            t, p = ap.coarse_vec(true, lv), ap.coarse_vec(prox, lv)
            m = t != "other" if lv != "appendage" else np.ones(len(t), bool)
            res[lv].append(float((t[m] == p[m]).mean()))
            if lv == "leg_seg":
                for a, b in zip(t[m], p[m]):
                    conf[(a.split("_")[-1], b.split("_")[-1] if b != "other" else "other")] = \
                        conf.get((a.split("_")[-1], b.split("_")[-1] if b != "other" else "other"), 0) + 1

    out = {"n_specimens": len(J), "gt_roundtrip_max_abs": rt, "excluded_joints": sorted(EXCL)}
    print(f"\nproxy accuracy vs true labels, n={len(J)} P48 specimens, {N_SAMPLE} points each")
    for lv, a in res.items():
        a = np.asarray(a)
        verdict = ("" if lv not in GATE else
                   ("PASS" if np.mean(a) >= GATE[lv] else "FAIL") + f" (gate {GATE[lv]})")
        print(f"  {lv:<10} mean {a.mean():.3f}  min {a.min():.3f}  p10 {np.percentile(a, 10):.3f}  {verdict}")
        out[lv] = dict(mean=float(a.mean()), min=float(a.min()), per_specimen=a.tolist(), verdict=verdict)

    segs = ["co", "tr", "fe", "ti", "ta", "pt"]
    print("\nleg-segment confusion (rows true, cols proxy; 'other' = proxy called it non-leg), row %")
    print("      " + "".join(f"{s:>7}" for s in segs + ["other"]))
    for a in segs:
        tot = sum(v for (x, _), v in conf.items() if x == a)
        print(f"{a:>5} " + "".join(f"{100 * conf.get((a, b), 0) / max(tot, 1):7.1f}" for b in segs + ["other"]))
    out["seg_confusion"] = {f"{a}->{b}": v for (a, b), v in conf.items()}
    json.dump(out, open(os.path.join(HERE, "out", "A2_gate_synthetic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
