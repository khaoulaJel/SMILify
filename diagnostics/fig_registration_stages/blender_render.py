"""Task 2 final renders (style B): fit in one colour over the scan's silhouette, Blender Cycles.

Renders, in the fitter frame (scan centred, max |coord| = 1; head +x, dorsal +z), with ONE orthographic
camera and ONE light rig for everything:
  scan.png              the scan alone (light grey)
  fit_<panel>.png       the fitted template alone (blue), for init / pose / Stage_2 / Stage_3
All RGBA on a transparent film, so `compose_figure.py` can lay the fit over the scan silhouette.

Camera = the pre-registered dorsolateral view (render_stages.py: look_at dist 2.6, elev 22, azim 140 in
the renderer's (x, z, -y) frame, orthographic half-width 1/1.15), translated into Blender's Z-up frame.

Run with the bpy venv (Blender 5.0.1 as a Python module):
    B=/hpcwork/nao48500/bpy_venv/lib/python3.11/site-packages/bpy
    LD_LIBRARY_PATH=$B/lib /hpcwork/nao48500/bpy_venv/bin/python -I \
        diagnostics/fig_registration_stages/blender_render.py --npz run_cpu1/probes/focus_meshes.npz --out run_cpu1/blender
"""

import argparse
import math
import os

import bpy
import numpy as np

PANELS = [("init", "init"), ("H2_joint", "pose"), ("Stage_2_deform_coarse", "Stage_2"),
          ("Stage_3_deform_fine", "Stage_3")]
FIT_HEX, SCAN_HEX = "#2a78d6", "#b4b3ae"
ELEV, AZIM, DIST, HALF_W = 22.0, 140.0, 2.6, 1.0 / 1.15
WORLD, KEY, FILL = 0.30, 260.0, 80.0  # ambient strength, area-light energies (W); tuned on low-res probes


def srgb_to_linear(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c) + (1.0,)


def camera_location():
    """Renderer frame (X, Y, Z) = (x, z, -y): pytorch3d eye = dist*(cos e sin a, sin e, cos e cos a)."""
    e, a = math.radians(ELEV), math.radians(AZIM)
    X, Y, Z = DIST * math.cos(e) * math.sin(a), DIST * math.sin(e), DIST * math.cos(e) * math.cos(a)
    return (X, -Z, Y)  # back to the fitter frame: x = X, y = -Z, z = Y


def material(name, hexcol, roughness):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = srgb_to_linear(hexcol)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
    return m


def add_mesh(name, verts, faces, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts, float).tolist(), [], np.asarray(faces, int).tolist())
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def setup_scene(size, samples):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "16"
    sc.view_settings.view_transform = "Standard"  # keep the palette's sRGB colours, no filmic tone shift

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = WORLD  # soft ambient + occlusion
    sc.world = world

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2 * HALF_W
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    cam.location = camera_location()
    direction = -np.array(cam.location)
    rot = _look_rotation(direction, up=(0, 0, 1))
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = rot
    sc.camera = cam

    # key: large soft area light above / camera-left; fill: weaker, opposite side
    for name, loc, energy, size_l in (("key", (1.5, 2.5, 3.5), KEY, 3.0), ("fill", (-2.5, 1.0, 1.5), FILL, 4.0)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size = energy, size_l
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_mode = "QUATERNION"
        lo.rotation_quaternion = _look_rotation(-np.array(loc), up=(0, 0, 1))
        sc.collection.objects.link(lo)


def _look_rotation(direction, up):
    """Quaternion rotating Blender's -Z (camera/light forward) onto `direction`, +Y onto `up`."""
    from mathutils import Vector

    return Vector(direction).to_track_quat("-Z", "Y")


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1800)
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--panels", default="all", help="comma list of panel labels, or 'all'")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)

    setup_scene(args.size, args.samples)
    fit_mat, scan_mat = material("fit", FIT_HEX, 0.5), material("scan", SCAN_HEX, 0.65)
    scan = add_mesh("scan", z["target_verts"], z["target_faces"], scan_mat)
    fits = {lab: add_mesh(f"fit_{lab}", z[st], z["faces"], fit_mat) for st, lab in PANELS}
    want = [lab for _, lab in PANELS] if args.panels == "all" else args.panels.split(",")

    def only(ob):
        for o in [scan] + list(fits.values()):
            o.hide_render = o is not ob

    only(scan)
    render_to(os.path.join(args.out, "scan.png"))
    for lab in want:
        only(fits[lab])
        render_to(os.path.join(args.out, f"fit_{lab}.png"))


if __name__ == "__main__":
    main()
