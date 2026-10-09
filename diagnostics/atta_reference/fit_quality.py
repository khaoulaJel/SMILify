"""Independent fit-quality check: how well does each arm actually fit the SPECIMEN meshes?

compare_arms.py measures agreement with Fabian's 2025 CSV, but that CSV is itself fitter
output, not ground truth -- so an arm using the same fitter as Fabian would score well there
by reproducing him, regardless of registration quality. This script instead measures each
arm's fitted mesh against the actual scanned mesh (symmetric chamfer), which no arm was
tuned against and which Fabian's numbers cannot bias.
"""
import glob, os, pickle
import numpy as np
from scipy.spatial import cKDTree

REPO = "/rwthfs/rz/cluster/home/nao48500/SMILify"
MESH = "/hpcwork/nao48500/atta20"
ARMS = {
    "ARM_A_master": f"{REPO}/diagnostics/atta_reference/runs/ARM_A_master/Stage_3_deform_fine.npz",
    "ARM_B_moonshot": f"{REPO}/diagnostics/moonshot/runs/ATTA20/Stage_3_deform_fine.npz",
}

def load_obj(p):
    """Load and apply the SAME normalisation the fitter applies at load time.

    fitter_3d/utils.py load_meshes() lines 341-344: centre on the vertex mean, then divide by
    max(abs(coord)). The fitted verts in the npz live in that normalised frame, so the targets
    must be put there too. Comparing raw scan coordinates against fitted verts produced chamfer
    values of ~400% of body diagonal that scaled with specimen size -- a frame mismatch, not a
    fit quality signal.
    """
    v = np.array([[float(x) for x in L.split()[1:4]] for L in open(p) if L.startswith("v ")])
    v = v - v.mean(0)
    return v / np.abs(v).max(0).max()

targets = {os.path.basename(f)[:2]: load_obj(f) for f in sorted(glob.glob(f"{MESH}/*.obj"))}

res = {}
for arm, path in ARMS.items():
    z = np.load(path, allow_pickle=True)
    verts = z["verts"].astype(np.float64)
    labels = [str(x).replace(".obj", "") for x in z["labels"]]
    per = {}
    for n, shp in enumerate(labels):
        F, T = verts[n], targets[shp]
        # symmetric chamfer, normalised by the target's own extent so sizes are comparable
        d1, _ = cKDTree(T).query(F)
        d2, _ = cKDTree(F).query(T)
        scale = np.linalg.norm(T.max(0) - T.min(0))  # both already normalised; ~O(1)
        per[shp] = float((d1.mean() + d2.mean()) / 2 / scale * 100)
    res[arm] = per
    print(f"{arm}: median chamfer = {np.median(list(per.values())):.4f}% of body diagonal")

print("\n" + "=" * 60)
print(f"{'shape':7}{'ARM_A %':>11}{'ARM_B %':>11}{'better':>12}")
for shp in sorted(res["ARM_A_master"]):
    va, vb = res["ARM_A_master"][shp], res["ARM_B_moonshot"][shp]
    print(f"{shp:7}{va:11.4f}{vb:11.4f}{('ARM_B' if vb < va else 'ARM_A'):>12}")
wb = sum(res["ARM_B_moonshot"][s] < res["ARM_A_master"][s] for s in res["ARM_A_master"])
ma = np.median(list(res["ARM_A_master"].values())); mb = np.median(list(res["ARM_B_moonshot"].values()))
print("=" * 60)
print(f"ARM_A {ma:.4f}%   ARM_B {mb:.4f}%   ARM_B better on {wb}/20")
print(f"VERDICT (fit quality): {'ARM_B (investigation)' if mb < ma else 'ARM_A (stock master)'} fits the real meshes better")
