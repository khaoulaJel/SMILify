"""Q3a / A7b: noise control for A7. Could expert annotation noise alone produce the real bend angles?

Adds isotropic Gaussian noise to TRAINING ground-truth joints with per-coordinate sigma chosen so the
3D error magnitude matches a given fraction of Weber's length, then recomputes A7's statistic.
Weber's length on synthetic specimens is not available from landmarks; it is approximated by the
real scans' WL_fit relative to their scan extent (median), applied to the training joints' frame by
matching the body-axis length (b_t -> b_a_5 chain) between real expert skeletons and training GT.
Noise levels: 2.4% WL (measured surface-landmark repeatability, V11, frame-checked in RULES_AUDIT R3)
and 2x / 4x that, since interior joint centres are probably less repeatable than surface landmarks.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from a7_articulation_vs_training import bends, rel, names  # noqa: E402

gt = json.load(open(os.path.join(REPO, "diagnostics/joint_alignment_benchmark/data/gt_joints_fitframe.json")))
AX = ["b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]


def axis_len(get):
    pts = [get(n) for n in AX]
    pts = [p for p in pts if p is not None]
    return sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)) if len(pts) > 1 else np.nan

ratio = []
for sid, v in gt.items():
    jj = {k: np.asarray(x["fit"]) for k, x in v["joints"].items()}
    L = axis_len(lambda n: jj.get(n))
    if np.isfinite(L) and L > 0:
        ratio.append(v["WL_fit"] / L)
wl_per_axis = float(np.median(ratio))
print(f"real WL / body-axis length: median {wl_per_axis:.3f} (n={len(ratio)})")

c = np.load("/hpcwork/nao48500/review_methods/corpus_b2_with_joints.npz")["joints"]
rng = np.random.default_rng(0)
out = {"wl_per_axis": wl_per_axis}
for frac in (0.0, 0.024, 0.048, 0.096):
    meds, allj = [], []
    for k in range(0, len(c), 10):
        Jk = c[k]
        WL = wl_per_axis * axis_len(lambda n: Jk[names.index(n)])
        sig = frac * WL / np.sqrt(3)          # 3D error magnitude ~ frac * WL
        Jn = Jk + rng.normal(0, sig, Jk.shape) if frac > 0 else Jk
        r = rel(bends(lambda n: Jn[names.index(n)]))
        meds.append(np.median(r)); allj.append(r)
    allj = np.concatenate(allj)
    print(f"noise {100*frac:4.1f}% WL: per-specimen median bend pct5/50/95/max = "
          f"{np.percentile(meds, [5, 50, 95, 100]).round(1)}   all-joint p50/p99 = {np.percentile(allj, [50, 99]).round(1)}")
    out[f"{frac}"] = dict(per_spec=np.percentile(meds, [5, 50, 95, 100]).tolist(), all_p50_p99=np.percentile(allj, [50, 99]).tolist())
json.dump(out, open(os.path.join(HERE, "out", "A7b_noise_control.json"), "w"), indent=1)
