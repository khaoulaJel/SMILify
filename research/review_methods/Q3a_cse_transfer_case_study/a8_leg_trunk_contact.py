"""Q3a / A8 (descriptive): are legs folded against the trunk on the specimens whose CSE targets fall
off the legs? Fit-independent: for each expert distal leg joint (ti, ta, pt), distance to the nearest
scan sample whose expert-skeleton proxy label is trunk/head/mandible/antenna (i.e. not a leg), as %
of Weber's length. Small = the leg touches non-leg surface. Same quantity on P48 ground truth
(GT joints, true labels) gives the in-distribution reference.
"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "common"))
import anatomy_proxy as ap
from a2_real_correspondence import scan_samples, EXCL, JAB
dd = ap.load_dd(); names, parent = ap.joint_tree(dd)
gt = json.load(open(os.path.join(JAB, "gt_joints_fitframe.json")))
DIST = lambda n: n.startswith("l_") and n.split("_")[2] in ("ti", "ta", "pt")
out = {}
for sid in sorted(gt):
    J = {k: np.asarray(v["fit"]) for k, v in gt[sid]["joints"].items()}
    S = scan_samples(sid)
    lab, _ = ap.label_points(S, J, parent, exclude=EXCL)
    nonleg = S[ap.coarse_vec(lab, "leg") == "other"]
    d, _ = cKDTree(nonleg).query(np.array([J[n] for n in J if DIST(n)]))
    d = 100 * d / gt[sid]["WL_fit"]
    out[sid] = dict(median=float(np.median(d)), frac_lt5=float((d < 5).mean()), n=len(d))
    print(f"REAL {sid[:14]:<14} distal-joint to non-leg surface: median {np.median(d):5.1f}% WL, "
          f"frac < 5% WL {(d < 5).mean():.2f} (n={len(d)})")
# synthetic reference: P48 GT, true labels; WL approximated as 0.380 x body-axis length (A7b ratio)
from a2_gate_synthetic import gt_joints_p48
z, Jg, _ = gt_joints_p48()
vl = ap.template_vertex_labels(dd); F = np.asarray(dd["f"]).astype(int)
AX = [names.index(n) for n in ["b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]]
ref = []
for i in range(0, len(Jg), 4):
    V = z["verts"][i]; nl = V[ap.coarse_vec(vl, "leg") == "other"]
    WL = 0.380 * sum(np.linalg.norm(Jg[i, AX[k + 1]] - Jg[i, AX[k]]) for k in range(5))
    d, _ = cKDTree(nl).query(np.array([Jg[i, k] for k, n in enumerate(names) if DIST(n)]))
    ref.append((np.median(100 * d / WL), ((100 * d / WL) < 5).mean()))
ref = np.array(ref)
print(f"SYN  P48 (12 specimens): median {np.median(ref[:, 0]):.1f}% WL (range {ref[:, 0].min():.1f}-{ref[:, 0].max():.1f}), "
      f"frac < 5% WL {ref[:, 1].mean():.2f}")
out["_synthetic_ref"] = dict(median_range=[float(ref[:, 0].min()), float(ref[:, 0].max())], frac_lt5_mean=float(ref[:, 1].mean()))
json.dump(out, open(os.path.join(HERE, "out", "A8_leg_trunk_contact.json"), "w"), indent=1)
