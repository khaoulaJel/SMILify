"""Q3a / A7: is real-scan articulation outside the training distribution? FIT-INDEPENDENT check.

Bend angle at a leg joint = angle between the incoming bone (parent -> joint) and the outgoing bone
(joint -> child), minus the same angle in the template rest pose. Computed from EXPERT joints on the
real scans (no fit involved) and from ground-truth FK joints on the training corpus and P48, so the
three are directly comparable. Joints: tr, fe, ti, ta of all six legs (both neighbours must be
placed by the expert). Also the coxa-to-body splay: angle of the coxa->trochanter bone vs rest.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import anatomy_proxy as ap

dd = ap.load_dd()
names, parent = ap.joint_tree(dd)
child = {}
for n, p in parent.items():
    if p is not None and n.startswith("l_"):
        child[p] = n
J = [n for n in names if n.startswith("l_") and n.split("_")[2] in ("tr", "fe", "ti", "ta")]


def ang(u, v):
    c = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def bends(get):
    """get(name) -> (3,) or None. Returns {joint: bend angle (deg)}"""
    out = {}
    for j in J:
        p, c = parent[j], child.get(j)
        if c is None:
            continue
        a, b, d = get(p), get(j), get(c)
        if a is None or b is None or d is None:
            continue
        out[j] = float(ang(b - a, d - b))
    return out

rest_J = np.asarray(dd["J"])
rest = bends(lambda n: rest_J[names.index(n)])


def rel(b):
    return np.array([abs(b[j] - rest[j]) for j in b])

res = {}
gt = json.load(open(os.path.join(REPO, "diagnostics/joint_alignment_benchmark/data/gt_joints_fitframe.json")))
real = {}
for sid, v in gt.items():
    jj = {k: np.asarray(x["fit"]) for k, x in v["joints"].items()}
    r = rel(bends(lambda n: jj.get(n)))
    real[sid] = r
c = np.load("/hpcwork/nao48500/review_methods/corpus_b2_with_joints.npz")["joints"]
train = [rel(bends(lambda n, k=k: c[k, names.index(n)])) for k in range(0, len(c), 10)]
tr_med = np.array([np.median(x) for x in train])
print(f"template rest bends computed for {len(rest)} joints")
print(f"TRAIN (every 10th of 4000) per-specimen median |bend - rest|: pct5/50/95/max = "
      f"{np.percentile(tr_med, [5, 50, 95, 100]).round(1)}")
allj_train = np.concatenate(train)
print(f"TRAIN all joints |bend-rest| pct50/95/99 = {np.percentile(allj_train, [50, 95, 99]).round(1)}")
for sid, r in sorted(real.items()):
    frac_out = float((r > np.percentile(allj_train, 99)).mean())
    print(f"REAL {sid[:14]:<14} n={len(r):2d} median {np.median(r):5.1f}  p90 {np.percentile(r, 90):5.1f}  "
          f"frac joints beyond train p99: {frac_out:.2f}")
    res[sid] = dict(n=len(r), median=float(np.median(r)), p90=float(np.percentile(r, 90)), frac_beyond_train_p99=frac_out)
res["_train"] = dict(per_specimen_median_pct=np.percentile(tr_med, [5, 50, 95, 100]).tolist(),
                     all_joints_p50_p95_p99=np.percentile(allj_train, [50, 95, 99]).tolist())
json.dump(res, open(os.path.join(HERE, "out", "A7_articulation.json"), "w"), indent=1)
