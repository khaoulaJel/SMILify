"""Dump, from an expert annotation .blend, every mesh object's world-space vertices and every
joint Empty's world location. Run with the bpy venv (headless Blender). Output: one .npz per .blend.
This is the fit-INDEPENDENT link between the annotation frame and the scan: the annotator placed the
joints in the same Blender scene as the mesh, so mesh-vs-.obj registration gives GT joints in scan
coordinates without consulting any fitted skeleton."""
import sys, os, json
import numpy as np
import bpy

src, out = sys.argv[1], sys.argv[2]
bpy.ops.wm.open_mainfile(filepath=src)
meshes, empties, info = {}, {}, {}
dg = bpy.context.evaluated_depsgraph_get()
for o in bpy.data.objects:
    mw = np.array(o.matrix_world)
    if o.type == "MESH":
        me = o.data
        V = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
        Vw = V @ mw[:3, :3].T + mw[:3, 3]
        meshes[o.name] = Vw
        info[o.name] = dict(type="MESH", n=len(V), matrix_world=mw.tolist(),
                            n_faces=len(me.polygons), parent=o.parent.name if o.parent else None)
    elif o.type == "EMPTY":
        empties[o.name] = mw[:3, 3].copy()
        info[o.name] = dict(type="EMPTY", matrix_world=mw.tolist(),
                            props={k: str(o[k]) for k in o.keys()},
                            parent=o.parent.name if o.parent else None)
np.savez(out, **{"mesh__" + k: v for k, v in meshes.items()},
         **{"empty__" + k: v for k, v in empties.items()}, info=json.dumps(info))
print(json.dumps({k: (v["type"], v.get("n")) for k, v in info.items()})[:3000])
