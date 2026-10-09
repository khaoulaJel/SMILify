"""Is force_static_joint_locs=True a valid workaround for Bug 1?

Same real upstream/master export_joint_distances, but now the stub mesh carries TWO shape keys
with genuinely different geometry. If the static path were a usable substitute, the two shape
keys would export different joint distances.
"""
import ast, types, csv as _csv
import numpy as np

src = open("/tmp/upstream_master_addon.py").read()
tree = ast.parse(src)
fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "export_joint_distances")
fn_src = ast.get_source_segment(src, fn_node)

NV, NJ = 12, 3
class Bone:
    def __init__(s, n, p): s.name = n; s.head_local = p
class Armature:
    def __init__(s):
        s.data = types.SimpleNamespace(bones=[Bone(f"j{i}", np.array([float(i),0,0])) for i in range(NJ)])
class Vert:
    def __init__(s, co): s.co = co
class KeyBlock:
    def __init__(s, name, scale): s.name = name; s.value = 0.0; s._scale = scale
class ShapeKeys:
    def __init__(s, kbs): s.key_blocks = kbs
class MeshData:
    def __init__(s):
        s._base = np.array([[float(i), 0.0, 0.0] for i in range(NV)])
        s.shape_keys = ShapeKeys([KeyBlock("Basis",1.0), KeyBlock("small",1.0), KeyBlock("big",3.0)])
        s.vertices = [Vert(c) for c in s._base]
    def update(s): pass
class MeshObj:
    def __init__(s): s.type="MESH"; s.data = MeshData(); s._props = {"static_joint_locs": True}
    def get(s,k,d=None): return s._props.get(k,d)
    def find_armature(s): return Armature()
    def evaluated_get(s, dg):
        # honour the active shape key: scale the geometry, as a real evaluated mesh would
        active = next((k for k in s.data.shape_keys.key_blocks[1:] if k.value > 0), None)
        scale = active._scale if active else 1.0
        ev = types.SimpleNamespace()
        ev.data = types.SimpleNamespace(vertices=[Vert(c*scale) for c in s.data._base])
        return ev
class Context:
    def __init__(s,o): s.active_object=o; s.scene=types.SimpleNamespace(
        smpl_tool=types.SimpleNamespace(j_regressor_method="inverse_distance",
                                        has_reference_data=False, reference_joint_pair=""))
    def evaluated_depsgraph_get(s): return types.SimpleNamespace(update=lambda: None)
    view_layer = types.SimpleNamespace(update=lambda: None)

ns = {"np":np, "csv":_csv,
      "export_J_regressor_to_npy": lambda *a,**k: np.eye(NJ, NV),
      "recalculate_joint_positions": lambda v,J: np.asarray(J) @ np.asarray(v),
      "get_reference_measurements": lambda c: {}}
exec(fn_src, ns)

out = "/tmp/_repro_static.csv"
ok, msg = ns["export_joint_distances"](Context(MeshObj()), out)
print("export returned:", ok, msg, "\n")
rows = list(_csv.reader(open(out)))
print("exported rows (shape, joint1, joint2, distance):")
for r in rows:
    print("  ", r)
vals = {}
for r in rows[1:]:
    if r[0] in ("small","big"): vals.setdefault(r[0], []).append(float(r[3]))
print("\n'small' shape-key distances:", vals.get("small"))
print("'big'   shape-key distances:", vals.get("big"), " (geometry is 3x larger)")
print("\nIDENTICAL despite 3x different geometry?", vals.get("small") == vals.get("big"))
