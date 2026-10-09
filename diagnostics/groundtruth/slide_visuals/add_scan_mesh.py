"""Add the real scan mesh (vertices + faces, transformed into the same fitter frame as the
fitted mesh) to the saved case, so the GIF can show scan (grey) vs fitted template overlay."""
import os
import numpy as np

OUT = "/w0/tmp/nao48500/login23-1_353989/claude-54434/-rwthfs-rz-cluster-home-nao48500-SMILify/9f42922d-7ac6-494a-aafb-70b3f65325c7/scratchpad/mesh_case"
d = dict(np.load(os.path.join(OUT, "case.npz"), allow_pickle=True))

sid = "Aenictus_ceylonicus_CASENT0878084"
obj_path = f"/hpcwork/nao48500/worker_alt_data/{sid}_processed.obj"

V, Ff = [], []
for L in open(obj_path):
    if L.startswith("v "):
        V.append([float(x) for x in L.split()[1:4]])
    elif L.startswith("f "):
        Ff.append([int(tok.split("/")[0]) - 1 for tok in L.split()[1:4]])
V = np.array(V)
Ff = np.array(Ff, dtype=np.int64)

c, sc = d["center"], float(d["scale"])
target_verts = (V - c) / sc

d["target_verts"] = target_verts
d["target_faces"] = Ff
np.savez(os.path.join(OUT, "case.npz"), **d)
print("scan mesh:", target_verts.shape, Ff.shape)
