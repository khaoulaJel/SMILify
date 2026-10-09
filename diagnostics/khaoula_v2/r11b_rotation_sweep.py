"""R11b -- remove the geometric baseline's inherited canonical frame, and see which method survives.

R11 found DINO recovers laterality well (82.6%) but loses to nearest-template-vertex proximity
(86.1%). That baseline was unusually advantaged: `synth_power48` specimens share the template's
topology AND sit in a roughly common orientation, so proximity preserves side almost for free.

Real scans do not offer that. R7 could not construct a reliable bilateral reference frame; R8 found
the body rotationally degenerate about its long axis. So the question R11 could not answer is:

    when the canonical frame is removed, which degrades -- geometry, or semantics?

Method: apply a rotation of magnitude theta about a fixed random axis to each test specimen, before
rendering AND before the geometric query. Both methods face the identical handicap. Sweep theta and
report the degradation curve. A crossing point is the result.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out_R11")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "moonshot"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("TORCH_HOME", "/hpcwork/nao48500/torch_hub")

import r11_lateralized_identity as R11  # noqa: E402  reuse everything

DEV = R11.DEV
ANGLES = [0, 30, 60, 90, 180]
N_TEST = 4


def rot_matrix(axis, theta_deg):
    """Rodrigues rotation about a unit axis."""
    t = np.deg2rad(theta_deg)
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def main():
    os.makedirs(OUT, exist_ok=True)
    v_t, f_t, _ = R11.T10.load_template()
    import pickle as _pk
    with open(os.path.join(REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        _u = _pk._Unpickler(fh); _u.encoding = "latin1"; dd = _u.load()
    jn = [str(x) for x in dd["J_names"]]
    dom = np.asarray(dd["weights"]).argmax(1)
    labels_all = np.array([R11.lateral_group_of(jn[d]) for d in dom])

    Ft, cov_t = R11.per_vertex_features(v_t, f_t)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ft[cov_t])
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(Ft[cov_t]), labels_all[cov_t])
    print(f"[r11b] probe trained on template ({cov_t.mean()*100:.1f}% coverage)", flush=True)

    gt = np.load(R11.SYNTH, allow_pickle=True)
    V = gt["verts"].astype(np.float64)
    names = [str(x) for x in gt["names"]]
    v_t_np = np.asarray(v_t.cpu())
    from scipy.spatial import cKDTree
    tree = cKDTree(v_t_np)
    lat_keys = list(R11.MIRROR)

    rs = np.random.RandomState(0)
    axis = rs.randn(3)          # ONE fixed axis, shared by all specimens/angles, seeded
    rows = []
    for theta in ANGLES:
        Rm = rot_matrix(axis, theta)
        das, gas = [], []
        for i in range(min(N_TEST, len(names))):
            vi = V[i] - V[i].mean(0)
            vi = vi / np.abs(vi).max()
            vi = vi @ Rm.T                      # SAME handicap applied to both methods
            vt_ = torch.tensor(vi, dtype=torch.float32, device=DEV)
            Fs, cov = R11.per_vertex_features(vt_, f_t)
            true = labels_all[cov]
            lat = np.isin(true, lat_keys)
            side_true = np.array([t[-1] for t in true[lat]])

            pred = np.array(clf.predict(sc.transform(Fs[cov])), dtype=object)
            sd = np.array([p[-1] if p in R11.MIRROR else "?" for p in pred[lat]])
            das.append(float((side_true == sd).mean()))

            gpred = labels_all[tree.query(vi[cov])[1]]
            sg = np.array([p[-1] if p in R11.MIRROR else "?" for p in gpred[lat]])
            gas.append(float((side_true == sg).mean()))
        rows.append(dict(theta_deg=theta, dino=float(np.mean(das)),
                         geometric=float(np.mean(gas)),
                         dino_per_specimen=das, geometric_per_specimen=gas))
        print(f"  theta={theta:3d}deg   DINO {np.mean(das)*100:5.1f}%   "
              f"geometric {np.mean(gas)*100:5.1f}%", flush=True)

    cross = [r["theta_deg"] for r in rows if r["dino"] > r["geometric"]]
    res = dict(angles=ANGLES, n_test=N_TEST, axis=[float(x) for x in axis],
               rows=rows, dino_beats_geometry_at=cross)
    with open(os.path.join(OUT, "r11b_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    print("\n" + "=" * 74)
    print("R11b -- lateral accuracy vs global rotation (canonical frame removed)")
    print("=" * 74)
    print(f"{'theta':>8s}{'DINO':>10s}{'geometric':>12s}{'winner':>12s}")
    for r in rows:
        w = "DINO" if r["dino"] > r["geometric"] else "geometry"
        print(f"{r['theta_deg']:>7d}d{r['dino']*100:>9.1f}%{r['geometric']*100:>11.1f}%{w:>12s}")
    print(f"\nDINO beats geometry at theta = {cross if cross else 'never'}")
    print("\nReading: geometry inherits a canonical frame at theta=0 and should collapse as theta")
    print("grows; DINO's view-based features do not depend on that frame. If DINO holds while")
    print("geometry falls, the semantic route is vindicated for the real-scan regime (R7/R8).")


if __name__ == "__main__":
    main()
