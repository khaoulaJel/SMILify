"""MINIMAL REPRODUCIBLE CASE — Bug 1: J_regressor UnboundLocalError.

Executes the REAL `export_joint_distances` source, extracted verbatim from
upstream/master's 3D_model_prep/SMIL_processing_addon.py, with bpy and the addon's
helpers stubbed. Nothing about the function body is modified.

Two runs, differing ONLY in the value of the object's `static_joint_locs` custom property:
  A) static_joint_locs = True   (as when force_static_joint_locs is ticked)
  B) static_joint_locs absent   (as in OmniAnt_25PCs_joint_limited.pkl -> .get() returns False)

Run with:  python repro_bug1_j_regressor.py
"""
import ast, sys, types, traceback
import numpy as np

SRC = "/tmp/upstream_master_addon.py"   # git show upstream/master:3D_model_prep/SMIL_processing_addon.py
src = open(SRC).read()
tree = ast.parse(src)
fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "export_joint_distances")
fn_src = ast.get_source_segment(src, fn_node)

# ---------------- stubs: the minimum needed to reach the guard ----------------
NV, NJ = 12, 4

class Bone:
    def __init__(s, n): s.name = n; s.head_local = np.zeros(3)
class ArmData:
    def __init__(s): s.bones = [Bone(f"j{i}") for i in range(NJ)]
class Armature:
    def __init__(s): s.data = ArmData()
class Vert:
    def __init__(s): s.co = np.zeros(3)
class MeshData:
    def __init__(s): s.vertices = [Vert() for _ in range(NV)]; s.shape_keys = None
    def update(s): pass
class MeshObj:
    """Stands in for the imported SMIL mesh object."""
    def __init__(s, props): s.type = "MESH"; s._props = props; s.data = MeshData()
    def get(s, k, default=None): return s._props.get(k, default)
    def find_armature(s): return Armature()
    def evaluated_get(s, depsgraph): return s
class Tool:
    j_regressor_method = "inverse_distance"; has_reference_data = False; reference_joint_pair = ""
class Scene:
    smpl_tool = Tool()
class Context:
    def __init__(s, obj): s.active_object = obj; s.scene = Scene()
    def evaluated_depsgraph_get(s): return object()
    view_layer = types.SimpleNamespace(update=lambda: None)

ns = {
    "np": np,
    "csv": __import__("csv"),
    "export_J_regressor_to_npy": lambda *a, **k: np.zeros((NJ, NV)),
    "recalculate_joint_positions": lambda verts, Jreg: np.zeros((NJ, 3)),
    "get_reference_measurements": lambda ctx: {},
}
exec(fn_src, ns)
export_joint_distances = ns["export_joint_distances"]

def run(label, props):
    print(f"\n--- {label} : obj props = {props} ---")
    obj = MeshObj(props)
    try:
        ok, msg = export_joint_distances(Context(obj), "/tmp/_repro_out.csv")
        print(f"  RESULT: returned ok={ok}  msg={msg!r}")
    except Exception as e:
        print(f"  RESULT: RAISED {type(e).__name__}: {e}")
        tb = traceback.format_exc().strip().splitlines()
        print(f"          at -> {tb[-2].strip()}")

print("=" * 78)
print("Bug 1 repro — real export_joint_distances from upstream/master, bpy stubbed")
print("=" * 78)
run("A. static_joint_locs = True", {"static_joint_locs": True})
run("B. static_joint_locs absent (the production model's actual state)", {})
print()
