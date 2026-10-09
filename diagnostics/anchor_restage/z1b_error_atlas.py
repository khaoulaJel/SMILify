"""Z1b -- whole-body error atlas on OPEN-shape-space fits.

Every prior per-part diagnosis in this project was made on fits whose shape space was frozen
(REPORT.md 6.7: betas sd 0.0004 vs prior sd 0.20-1.70). Z1 showed the headline anchor built on
those fits was stale by 0.98 -> 0.55. This re-measures the SYMPTOM list on the same 50 specimens
with the shape space open, so we act on failures that are still real.

Per anatomical group (PART_GROUPS_FINE, vertex indices from skinning weights):
  deform_ratio   mean |deform_verts| relative to thorax -- REPORT 6.10's head-carried ratio,
                 the metric X1 used as its MECHANISM endpoint. 1.0 = group deforms like the
                 thorax; <1 = the parametric model is under-serving it; >1 = free-form is
                 carrying the group.
  deform_abs     mean |deform_verts|, absolute (so a ratio cannot hide a dead denominator)
  scale_range    per-group max/min of exp(log_beta_scales) -- X1's PROXY quantity
  share          group's share of total free-form variance vs its share of vertices.
                 >1 means free-form work is CONCENTRATED there. This is the "where is the
                 model failing" question stated as a ratio that is comparable across groups.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from part_groups import PART_GROUPS_FINE, get_part_vertex_indices  # noqa: E402

RUNS = ["bench50_G1_learned", "bench50_G3_zero", "bench50_G1b_learned_lowpose"]
ORDER = ["thorax", "head", "mandible", "antenna_r", "antenna_l", "waist", "gaster",
         "leg1_r", "leg2_r", "leg3_r", "leg1_l", "leg2_l", "leg3_l"]


def main():
    parts = get_part_vertex_indices(PART_GROUPS_FINE)
    for run in RUNS:
        p = os.path.join(REPO, "diagnostics/moonshot/runs", run, "Stage_3_deform_fine.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        dv = np.asarray(d["deform_verts"], dtype=np.float64)          # (n, V, 3)
        lbs = np.asarray(d["log_beta_scales"], dtype=np.float64)      # (n, 55, 3)
        n, V, _ = dv.shape
        mag = np.linalg.norm(dv, axis=2)                              # (n, V)
        var = (dv - dv.mean(0, keepdims=True)) ** 2                   # (n, V, 3)
        tot_var = var.sum()

        thorax = float(mag[:, parts["thorax"]].mean())
        print("\n" + "=" * 92)
        print(f"{run}   n={n}   thorax |deform| = {thorax:.5f}   (6.10 range 0.019-0.043)")
        print("=" * 92)
        print(f"{'group':<11}{'nverts':>7}{'|deform|':>11}{'ratio/thx':>11}"
              f"{'var share':>11}{'vert share':>11}{'concentr.':>11}{'scale rng':>11}")
        for g in ORDER:
            idx = parts[g]
            if len(idx) == 0:
                continue
            m = float(mag[:, idx].mean())
            vs = float(var[:, idx].sum() / tot_var)
            ps = len(idx) / V
            joints = PART_GROUPS_FINE[g]
            sc = np.exp(lbs[:, joints, :])
            rng = float((sc.reshape(n, -1).max(1) / np.maximum(sc.reshape(n, -1).min(1), 1e-9)).mean())
            print(f"{g:<11}{len(idx):>7}{m:>11.5f}{m/thorax:>11.3f}"
                  f"{vs*100:>10.1f}%{ps*100:>10.1f}%{vs/ps:>11.2f}{rng:>11.2f}")


if __name__ == "__main__":
    main()
