"""Execution plan §4: secondary (integrity) + third decision-rule condition (anterior joint-
scale pathology), on the FULL corpus, arm A (D1_PROD) vs arm B (D1_PROD_SCALECAP). Reuses
diagnostics/moonshot/metrics.py's deformation_metrics() unchanged (same suite used for the
"shipped" integrity dashboard elsewhere) and the model's own weights.argmax(1)/J_names to group
joints into head/mandible/antenna, same grouping convention as measure.py/score_synth_roundtrip.py.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "moonshot"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from metrics import deformation_metrics  # noqa: E402

DEV = "cpu"

with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
    u = pickle._Unpickler(fh)
    u.encoding = "latin1"
    dd = u.load()
jn = [str(x) for x in dd["J_names"]]
v_tpl = torch.tensor(np.asarray(dd["v_template"], dtype=np.float32), device=DEV)
faces = torch.tensor(np.asarray(dd["f"]).astype(np.int64), device=DEV)

ANTERIOR = {
    "head": [i for i, n in enumerate(jn) if n == "b_h"],
    "mandible": [i for i, n in enumerate(jn) if n.startswith("ma")],
    "antenna": [i for i, n in enumerate(jn) if n.startswith("an_")],
}


def anterior_ratio(log_beta_scales):
    """exp(log_beta_scales) range (max/min across the 3 scale axes) per anterior group, per
    specimen -- same statistic T0.4's bench50_clean check used (49-83% compression claim)."""
    s = np.exp(log_beta_scales)  # (n_joints, 3)
    out = {}
    for part, idx in ANTERIOR.items():
        if not idx:
            continue
        vals = s[idx].reshape(-1)
        out[part] = float(vals.max() / max(vals.min(), 1e-9))
    return out


def run_arm(tag, run_dir_glob):
    runs = sorted(glob.glob(os.path.join(REPO, "diagnostics", "moonshot", "runs", run_dir_glob)))
    integ = {k: [] for k in ("edge_logratio_absmean", "deform_mag_mean", "deform_mag_p95", "folded_face_frac", "tri_quality_mean")}
    ant = {k: [] for k in ANTERIOR}
    n = 0
    for rd in runs:
        p = os.path.join(rd, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            continue
        d = np.load(p)
        verts = torch.tensor(d["verts"], dtype=torch.float32, device=DEV)
        dverts = torch.tensor(d["deform_verts"], dtype=torch.float32, device=DEV) if "deform_verts" in d else None
        lbs = d["log_beta_scales"]
        for i in range(verts.shape[0]):
            m = deformation_metrics(verts[i], v_tpl, faces, dverts[i] if dverts is not None else None)
            for k in integ:
                if k in m:
                    integ[k].append(m[k])
            a = anterior_ratio(lbs[i])
            for k, v in a.items():
                ant[k].append(v)
            n += 1
    return n, integ, ant


def summarize(name, n, integ, ant):
    print(f"\n=== {name}: {n} specimens ===")
    print("  integrity:")
    for k, v in integ.items():
        if v:
            print(f"    {k:<22} mean {np.mean(v):.5f}  p95 {np.percentile(v, 95):.5f}")
    print("  anterior joint-scale range (max/min), population:")
    for k, v in ant.items():
        if v:
            print(f"    {k:<10} mean {np.mean(v):.2f}x  median {np.median(v):.2f}x  p90 {np.percentile(v, 90):.2f}x  max {np.max(v):.2f}x")
    return {"n": n, "integrity": {k: (float(np.mean(v)), float(np.percentile(v, 95))) for k, v in integ.items() if v},
            "anterior": {k: (float(np.mean(v)), float(np.median(v)), float(np.percentile(v, 90)), float(np.max(v))) for k, v in ant.items() if v}}


if __name__ == "__main__":
    nA, iA, aA = run_arm("A", "AB_A_[WC]*")
    nB, iB, aB = run_arm("B", "AB_B_[WC]*")
    sA = summarize("Arm A (baseline, D1_PROD)", nA, iA, aA)
    sB = summarize("Arm B (scale_cap, D1_PROD_SCALECAP)", nB, iB, aB)

    print("\n=== DELTA (B - A), integrity ===")
    for k in iA:
        if iA[k] and iB[k]:
            dA, dB = np.mean(iA[k]), np.mean(iB[k])
            print(f"  {k:<22} A {dA:.5f}  B {dB:.5f}  delta {dB - dA:+.5f}  ({100 * (dB - dA) / max(abs(dA), 1e-9):+.1f}%)")

    print("\n=== DELTA (B - A), anterior joint-scale pathology (compression = B should be SMALLER) ===")
    for k in aA:
        if aA[k] and aB[k]:
            mA, mB = np.mean(aA[k]), np.mean(aB[k])
            pct = 100 * (mB - mA) / mA
            print(f"  {k:<10} A mean {mA:.2f}x  B mean {mB:.2f}x  {pct:+.1f}%  ({'compressed' if pct < 0 else 'WORSE'})")

    out = {"arm_A": sA, "arm_B": sB}
    outp = os.path.join(os.path.dirname(__file__), "integrity_anterior_full_corpus.json")
    with open(outp, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {outp}")
