"""Real Blender renders of the head-width joints (b_h_l / b_h_r) on the template and on every
registered specimen. Written for Blender 4.2 + the SMIL Model Importer.

HOW TO RUN -- see blender/HOW_TO_RUN.md. Two ways:
  * GUI:   Scripting tab -> Open this file -> Run Script      (save the .blend first)
  * CLI:   blender -b <file.blend> -P blender_screenshots.py -- <output_folder>

NOT YET TESTED IN BLENDER (none on the cluster where it was written). The joint regression mirrors the
importer's exporter (10 nearest Basis vertices, weights 1/d) and was verified in Python against the
exported CSV to 0.05%; the bpy calls themselves are unrun. Look at the first image before trusting
the batch -- the console prints each step.

Output
  00_template_{anterior,dorsal,lateral}.png   template head with the two joints (spheres)
  00_whole_model_dorsal.png                   whole template, dorsal
  <specimen>_head_{anterior,dorsal}.png       every registered specimen, REGRESSED joints
"""
import os
import sys

import bpy
from mathutils import Matrix, Vector, kdtree

REPORTERS = ("b_h_l", "b_h_r")


# --------------------------------------------------------------------------- output folder
def output_dir():
    if "--" in sys.argv and len(sys.argv) > sys.argv.index("--") + 1:
        d = sys.argv[sys.argv.index("--") + 1]
    elif bpy.data.filepath:                          # next to the saved .blend
        d = bpy.path.abspath("//blender_screens")
    else:                                            # unsaved session: never the Blender install dir
        d = os.path.join(os.path.expanduser("~"), "blender_screens")
    os.makedirs(d, exist_ok=True)
    return d


OUT = output_dir()
print(f"[hw] writing to {OUT}")

# --------------------------------------------------------------------------- find the objects
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"
        and all(n in o.data.bones for n in REPORTERS)]
if not arms:
    raise RuntimeError("No armature contains bones b_h_l and b_h_r. Add them (or Append the armature "
                       "from bone_placement.blend) before running this script.")
arm = arms[0]

meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.data.shape_keys
          and len(o.data.shape_keys.key_blocks) > 1]
if not meshes:
    raise RuntimeError("No mesh with specimen shape keys found. Import the fit with the SMIL Model Importer first.")
act = bpy.context.view_layer.objects.active
mesh = act if act in meshes else max(meshes, key=lambda o: len(o.data.shape_keys.key_blocks))
keys = mesh.data.shape_keys.key_blocks
specimens = list(keys[1:])
print(f"[hw] armature '{arm.name}', mesh '{mesh.name}', {len(specimens)} specimen shape keys")


def bone_world(name):
    return arm.matrix_world @ arm.data.bones[name].head_local


# --------------------------------------------------------------------------- regressor (as exporter)
basis_world = [mesh.matrix_world @ v.co for v in keys[0].data]
kd = kdtree.KDTree(len(basis_world))
for i, co in enumerate(basis_world):
    kd.insert(co, i)
kd.balance()
REG = {}
for name in REPORTERS:
    nn = kd.find_n(bone_world(name), 10)
    inv = [(idx, 1.0 / max(dist, 1e-12)) for _co, idx, dist in nn]
    s = sum(w for _, w in inv)
    REG[name] = [(idx, w / s) for idx, w in inv]
    print(f"[hw] {name}: max regressor weight {max(w for _, w in REG[name]):.3f}")


def group_members(prefix_test):
    """Vertex indices whose dominant-ish weight (>0.5) is in a group matching prefix_test(name)."""
    gid = {g.index for g in mesh.vertex_groups if prefix_test(g.name)}
    if not gid:
        return []
    return [v.index for v in mesh.data.vertices if any(g.group in gid and g.weight > 0.5 for g in v.groups)]


HEAD_V = group_members(lambda n: n == "b_h")
THORAX_V = group_members(lambda n: n == "b_t")
LEG_V = group_members(lambda n: n.startswith("l_") and "_co_" in n)
print(f"[hw] vertex groups: head {len(HEAD_V)}, thorax {len(THORAX_V)}, coxae {len(LEG_V)}")


# --------------------------------------------------------------------------- helpers
def set_only(key):
    for k in keys:
        k.value = 0.0                                # keys are ADDITIVE: always zero the rest
    if key is not None:
        key.value = 1.0
    bpy.context.view_layer.update()


def evaluated_world():
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    m = ev.to_mesh()
    pts = [ev.matrix_world @ v.co for v in m.vertices]
    ev.to_mesh_clear()
    return pts


def centroid(pts, idx):
    return sum((pts[i] for i in idx), Vector((0.0, 0.0, 0.0))) / max(len(idx), 1)


def regressed(pts):
    jl = sum((pts[i] * w for i, w in REG["b_h_l"]), Vector((0.0, 0.0, 0.0)))
    jr = sum((pts[i] * w for i, w in REG["b_h_r"]), Vector((0.0, 0.0, 0.0)))
    return jl, jr


def head_axes(pts, jl, jr):
    """This shot's own anatomical frame: lateral from the two joints, antero-posterior from the
    thorax->head centroids, dorsal pointing away from the coxae. Falls back to bones if the vertex
    groups are missing."""
    lat = (jl - jr).normalized()
    if HEAD_V and THORAX_V:
        ap = centroid(pts, HEAD_V) - centroid(pts, THORAX_V)
    else:
        ap = bone_world("b_h") - bone_world("b_a_5")
    ap = (ap - lat * ap.dot(lat)).normalized()
    up = lat.cross(ap).normalized()
    legs = centroid(pts, LEG_V) if LEG_V else bone_world("b_t") - up
    ref = centroid(pts, THORAX_V) if THORAX_V else bone_world("b_t")
    if (legs - ref).dot(up) > 0:                     # legs are ventral
        up = -up
    return lat, ap, up


def make_marker(name, rgba):
    o = bpy.data.objects.get(name)
    if o is None:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16)
        o = bpy.context.active_object
        o.name = name
        mat = bpy.data.materials.new(name + "_mat")
        mat.diffuse_color = rgba
        o.data.materials.append(mat)
    return o


ML = make_marker("HW_joint_b_h_l", (0.83, 0.33, 0.00, 1.0))
MR = make_marker("HW_joint_b_h_r", (0.12, 0.37, 0.66, 1.0))

# head-width bar between the two joints (seen in the X-ray shots)
BAR = bpy.data.objects.get("HW_bar")
if BAR is None:
    bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, vertices=32)
    BAR = bpy.context.active_object
    BAR.name = "HW_bar"
    _m = bpy.data.materials.new("HW_bar_mat")
    _m.diffuse_color = (0.05, 0.18, 0.32, 1.0)
    BAR.data.materials.append(_m)

# Head-only view: a Mask modifier on the `b_h` vertex group, enabled for RENDER only. It stays off in
# the viewport, so the regression (which reads the evaluated mesh through the viewport depsgraph)
# always sees every vertex and keeps the exporter's vertex indices.
HEAD_MASK = mesh.modifiers.get("HW_head_only")
if HEAD_MASK is None and "b_h" in mesh.vertex_groups:
    HEAD_MASK = mesh.modifiers.new("HW_head_only", "MASK")
    HEAD_MASK.vertex_group = "b_h"
    HEAD_MASK.threshold = 0.5
if HEAD_MASK is not None:
    HEAD_MASK.show_viewport = False
    HEAD_MASK.show_render = False
else:
    print("[hw] no 'b_h' vertex group -- head-only shots will show the whole model")

cam_data = bpy.data.cameras.get("HWcam") or bpy.data.cameras.new("HWcam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.get("HWcam") or bpy.data.objects.new("HWcam", cam_data)
if cam.name not in bpy.context.scene.collection.objects:
    bpy.context.scene.collection.objects.link(cam)
sc = bpy.context.scene
sc.camera = cam
sc.render.engine = "BLENDER_WORKBENCH"
sc.render.resolution_x = sc.render.resolution_y = 1400
sc.render.image_settings.file_format = "PNG"
sc.display.shading.light = "STUDIO"
sc.display.shading.color_type = "MATERIAL"
cam_data.clip_end = 1000.0


def shoot(path, target, toward_camera, up, size, head_only=False, xray=False):
    if HEAD_MASK is not None:
        HEAD_MASK.show_render = head_only
    sc.display.shading.show_xray = xray
    sc.display.shading.xray_alpha = 0.45
    BAR.hide_render = not xray
    toward_camera = toward_camera.normalized()
    cam.location = target + toward_camera * size * 4.0
    back = toward_camera                              # camera looks along its local -Z
    right = up.cross(back).normalized()
    upv = back.cross(right).normalized()
    cam.rotation_euler = Matrix((right, upv, back)).transposed().to_euler()
    cam_data.ortho_scale = size
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"[hw] wrote {path}")


def place_markers(jl, jr):
    d = jl - jr
    r = d.length * 0.055
    for m, p in ((ML, jl), (MR, jr)):
        m.location = p
        m.scale = (r, r, r)
    rb = d.length * 0.012
    BAR.location = (jl + jr) / 2
    BAR.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    BAR.scale = (rb, rb, d.length / 2)
    bpy.context.view_layer.update()


# --------------------------------------------------------------------------- template
set_only(None)
pts = evaluated_world()
jl, jr = regressed(pts)
lat, ap, up = head_axes(pts, jl, jr)
place_markers(jl, jr)
mid, hw = (jl + jr) / 2, (jl - jr).length
shoot(f"{OUT}/00_template_anterior.png", mid, ap, up, hw * 1.6, head_only=True)
shoot(f"{OUT}/00_template_anterior_xray.png", mid, ap, up, hw * 1.6, head_only=True, xray=True)
shoot(f"{OUT}/00_template_anterior_context.png", mid, ap, up, hw * 1.8)
shoot(f"{OUT}/00_template_dorsal.png", mid, up, ap, hw * 1.6, head_only=True)
shoot(f"{OUT}/00_template_lateral.png", mid, lat, up, hw * 1.6, head_only=True)
body_c = centroid(pts, range(len(pts)))
span = max((p - body_c).length for p in pts)
shoot(f"{OUT}/00_whole_model_dorsal.png", body_c, up, ap, span * 2.1)

# --------------------------------------------------------------------------- every specimen
for k in specimens:
    set_only(k)
    pts = evaluated_world()
    jl, jr = regressed(pts)
    lat, ap, up = head_axes(pts, jl, jr)
    place_markers(jl, jr)
    mid, hw = (jl + jr) / 2, (jl - jr).length
    tag = k.name.split(".")[0]
    shoot(f"{OUT}/{tag}_head_anterior.png", mid, ap, up, hw * 1.6, head_only=True)
    shoot(f"{OUT}/{tag}_head_anterior_xray.png", mid, ap, up, hw * 1.6, head_only=True, xray=True)
    shoot(f"{OUT}/{tag}_head_dorsal.png", mid, up, ap, hw * 1.6, head_only=True)
    print(f"[hw] {k.name}: regressed joint separation {hw:.5f} (Blender units)")

set_only(None)
print(f"[hw] done -> {OUT}")
