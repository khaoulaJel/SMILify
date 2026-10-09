"""Score both fitting arms against Fabian's reference joint-distance CSV.

This does NOT need Blender or the addon: joint positions are J_regressor @ verts, which is
exactly what the addon's recalculate_joint_positions() computes. It therefore gives an early
read on ARM A (stock master fitter) vs ARM B (investigation recipe) using Fabian's numbers as
an independent target that neither arm was tuned on.

CAVEAT, stated up front. Fabian's 2025 export recomputed the J_regressor from the armature by
inverse-distance-to-nearest-10-vertices, whereas this script uses the pkl's trained J_regressor.
So the absolute agreement here is NOT the replication figure -- the Blender export is. But the
regressor choice is applied identically to both arms, so the A/B ranking is valid.

Head width (b_h_l/b_h_r) is excluded: those bones do not exist in the pkl, only in Blender.
"""
import csv, pickle, sys
from collections import defaultdict
import numpy as np

REPO = "/rwthfs/rz/cluster/home/nao48500/SMILify"
REF = f"{REPO}/diagnostics/atta_reference/measurements"
ARMS = {
    "ARM_A_master": f"{REPO}/diagnostics/atta_reference/runs/ARM_A_master/Stage_3_deform_fine.npz",
    "ARM_B_moonshot": f"{REPO}/diagnostics/moonshot/runs/ATTA20/Stage_3_deform_fine.npz",
}

d = pickle.load(open(f"{REPO}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl", "rb"), encoding="latin1")
J_reg = np.asarray(d["J_regressor"], dtype=np.float64)
names = [str(x) for x in d["J_names"]]
idx = {n: i for i, n in enumerate(names)}
print(f"model: {J_reg.shape[0]} joints x {J_reg.shape[1]} verts")

ref_len = {}
for r in list(csv.reader(open(f"{REF}/ant_body_lengths.csv")))[1:]:
    ref_len[r[0]] = float(r[1])

# Fabian's scaled distances, shapes 01..20, pairs among the pkl's 55 joints only
fab = defaultdict(dict)
with open(f"{REF}/SMPL_Object_joint_distances.csv") as fh:
    for row in csv.reader(fh):
        if len(row) < 6 or not (len(row[0]) == 2 and row[0].isdigit()):
            continue
        j1, j2, scaled = row[1], row[2], row[5]
        if j1 not in idx or j2 not in idx or scaled in ("N/A", ""):
            continue
        fab[row[0]][tuple(sorted((j1, j2)))] = float(scaled)
print(f"reference: {len(fab)} shapes, {len(next(iter(fab.values())))} comparable pairs each")

results = {}
for arm, path in ARMS.items():
    z = np.load(path, allow_pickle=True)
    verts, labels = z["verts"].astype(np.float64), [str(x).replace(".obj", "") for x in z["labels"]]
    joints = np.einsum("jv,nvc->njc", J_reg, verts)          # (20, 55, 3)
    errs, per_shape = [], {}
    for n, shp in enumerate(labels):
        P = joints[n]
        body = np.linalg.norm(P[idx["b_t"]] - P[idx["b_a_5"]])
        scale = ref_len[shp] / body                            # mm per model unit
        se = []
        for (j1, j2), ref_mm in fab[shp].items():
            got = np.linalg.norm(P[idx[j1]] - P[idx[j2]]) * scale
            if ref_mm > 1e-9:
                se.append(abs(got - ref_mm) / ref_mm)
        per_shape[shp] = float(np.median(se) * 100)
        errs += se
    results[arm] = (float(np.median(errs) * 100), per_shape)
    print(f"\n{arm}: median |rel err| vs reference = {np.median(errs)*100:.2f}%  (n={len(errs)} pair-measurements)")

a, b = results["ARM_A_master"][0], results["ARM_B_moonshot"][0]
print("\n" + "=" * 62)
print(f"{'shape':7}{'ARM_A %':>10}{'ARM_B %':>10}{'winner':>10}")
for shp in sorted(results["ARM_A_master"][1]):
    va, vb = results["ARM_A_master"][1][shp], results["ARM_B_moonshot"][1][shp]
    print(f"{shp:7}{va:10.2f}{vb:10.2f}{('ARM_B' if vb < va else 'ARM_A'):>10}")
wins_b = sum(results["ARM_B_moonshot"][1][s] < results["ARM_A_master"][1][s] for s in results["ARM_A_master"][1])
print("=" * 62)
print(f"OVERALL median: ARM_A {a:.2f}%   ARM_B {b:.2f}%")
print(f"ARM_B wins {wins_b}/20 specimens")
print(f"VERDICT: {'ARM_B (investigation recipe)' if b < a else 'ARM_A (stock master)'} is closer to the reference")
