"""Offline test of build_J_regressor_with_reporters, executed against the REAL source text.

Blender is not importable here, so the function is extracted from measurements.py and exec'd
with stubs. This verifies the property that matters: trained rows survive byte-exact and only
reporter bones get freshly computed rows -- including when Blender orders bones differently
from the model's J_names.
"""
import ast, sys
import numpy as np

SRC = "3D_model_prep/smil_importer/measurements.py"
tree = ast.parse(open(SRC).read())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_J_regressor_with_reporters")
code = ast.get_source_segment(open(SRC).read(), fn)

NV, NJ = 40, 6
rng = np.random.default_rng(0)
trained = rng.random((NJ, NV))
names = [f"j{i}" for i in range(NJ)]
computed_marker = np.full((NJ + 2, NV), -1.0)   # sentinel: any computed row is all -1

ns = {"np": np,
      "export_J_regressor_to_npy": lambda *a, **k: computed_marker,
      "get_smpl_data": lambda ctx: {"J_regressor": trained, "J_names": names}}
exec(code, ns)
f = ns["build_J_regressor_with_reporters"]

class B:  # minimal bone/armature/mesh stubs
    def __init__(s, n): s.name = n
class Arm:
    def __init__(s, bs): s.data = type("D", (), {"bones": [B(n) for n in bs]})()
class Tool:
    j_regressor_method = "inverse_distance"
class Mesh:
    def __init__(s, nv): s.data = type("D", (), {"vertices": [None] * nv})()

def check(label, bone_order, expect_reporters):
    R, msg = f(None, Mesh(NV), Arm(bone_order), Tool())
    assert R.shape == (len(bone_order), NV), f"{label}: bad shape {R.shape}"
    bad = []
    for i, bn in enumerate(bone_order):
        if bn in names:
            if not np.array_equal(R[i], trained[names.index(bn)]):
                bad.append(f"{bn}: trained row NOT preserved")
        else:
            if not np.all(R[i] == -1.0):
                bad.append(f"{bn}: reporter row not freshly computed")
    assert not bad, f"{label}: " + "; ".join(bad)
    print(f"  PASS  {label}\n        -> {msg}")

print("build_J_regressor_with_reporters:")
check("2 reporters appended at end", names + ["b_h_l", "b_h_r"], 2)
check("reporters interleaved mid-list", names[:3] + ["b_h_l"] + names[3:] + ["b_h_r"], 2)
check("bones reordered vs J_names", list(reversed(names)) + ["b_h_l", "b_h_r"], 2)
check("no reporters (unchanged rig)", names, 0)

# degraded paths must fall back rather than produce a wrong-shaped or mismatched matrix
ns["get_smpl_data"] = lambda ctx: None
R, msg = f(None, Mesh(NV), Arm(names + ["b_h_l"]), Tool())
assert np.all(R == -1.0) and "recomputed" in msg
print(f"  PASS  no stored model data -> falls back\n        -> {msg}")

ns["get_smpl_data"] = lambda ctx: {"J_regressor": np.zeros((NJ, NV + 5)), "J_names": names}
R, msg = f(None, Mesh(NV), Arm(names + ["b_h_l"]), Tool())
assert np.all(R == -1.0) and "does not match mesh" in msg
print(f"  PASS  stale regressor (wrong vert count) -> falls back\n        -> {msg}")
print("\nALL PASS")
