"""Task 1 visual judgement: one contact sheet per specimen (run inside Blender, background).

Columns: raw scan | V0 | V1 | V2 | 2024 worker mesh. Rows: lateral, dorsal (from the raw scan's own
principal axes). Every output is placed in the raw frame with the rigid transform t1_measure.py
recovered (out/measure_<s>.json), so all panels share one orthographic camera and are comparable.
Outputs are split into loose parts and each part gets its own colour (largest = neutral grey-blue),
so dropped anatomy, kept fragments and debris are visible. Workbench engine, flat studio light.

Usage: blender --background --python t1_render.py -- <specimen> [<specimen> ...]
Writes /hpcwork/nao48500/holotype_task1/renders/<specimen>_<column>_<view>.png
"""
import colorsys
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
O = "/hpcwork/nao48500/holotype_task1"
SRC = {"raw": os.environ.get("RAW_ROOT", "/hpcwork/nao48500/antscan_data") + "/{s}/{s}.stl",
       "V0": O + "/V0/{s}/{s}_processed.obj", "V1": O + "/V1/{s}/{s}_processed.obj",
       "V2": O + "/V2/{s}/{s}_processed.obj", "W2024": "/hpcwork/nao48500/worker_ALT/{s}_processed.obj"}


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "OBJECT"
    sc.render.resolution_x = sc.render.resolution_y = 900
    sc.render.film_transparent = False
    sc.world = bpy.data.worlds.new("w"); sc.world.color = (1, 1, 1)


def load(path):
    before = set(bpy.data.objects)
    if path.endswith(".stl"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    return [o for o in bpy.data.objects if o not in before][0]


def colour_parts(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.separate(type="LOOSE"); bpy.ops.object.mode_set(mode="OBJECT")
    parts = sorted([o for o in bpy.context.selected_objects], key=lambda o: -len(o.data.polygons))
    for i, p in enumerate(parts):
        p.color = (0.55, 0.62, 0.72, 1) if i == 0 else (*colorsys.hsv_to_rgb((i * 0.618) % 1, 0.85, 0.9), 1)
    return parts


def camera(centre, axis_view, axis_up, extent):
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"; cam.data.ortho_scale = extent * 1.15
    z = Vector(axis_view).normalized(); y = Vector(axis_up).normalized(); x = y.cross(z)
    cam.matrix_world = Matrix.Translation(Vector(centre) + z * extent * 3) @ Matrix(
        ((x[0], y[0], z[0], 0), (x[1], y[1], z[1], 0), (x[2], y[2], z[2], 0), (0, 0, 0, 1)))
    cam.data.clip_end = extent * 10
    bpy.context.scene.camera = cam
    return cam


def main():
    specimens = sys.argv[sys.argv.index("--") + 1:]
    os.makedirs(O + "/renders", exist_ok=True)
    for s in specimens:
        meas = json.load(open(os.path.join(HERE, "out", f"measure_{s}.json")))
        clear()
        raw = load(SRC["raw"].format(s=s))
        co = np.empty(len(raw.data.vertices) * 3); raw.data.vertices.foreach_get("co", co)
        V = co.reshape(-1, 3)[:: max(1, len(raw.data.vertices) // 200000)]   # bpy collections do not support step slicing
        c = V.mean(0); w, ev = np.linalg.eigh(np.cov((V - c).T)); ax = ev[:, ::-1]
        extent = float(np.ptp((V - c) @ ax[:, 0]))
        views = {"lateral": (ax[:, 2], ax[:, 1]), "dorsal": (ax[:, 1], ax[:, 2])}
        for col in ("raw", "V0", "V1", "V2", "W2024"):
            for o in list(bpy.data.objects):
                if o.type == "MESH":
                    o.hide_render = True
            if col == "raw":
                raw.hide_render = False; raw.color = (0.7, 0.7, 0.7, 1); shown = [raw]
            else:
                if meas.get(col, {}).get("missing", True) and "transform" not in meas.get(col, {}):
                    continue
                obj = load(SRC[col].format(s=s))
                obj.matrix_world = Matrix(meas[col]["transform"]) @ obj.matrix_world
                bpy.ops.object.select_all(action="DESELECT")
                shown = colour_parts(obj)
            for name, (vdir, up) in views.items():
                cam = camera(c, vdir, up, extent)
                bpy.context.scene.render.filepath = f"{O}/renders/{s}_{col}_{name}.png"
                bpy.ops.render.render(write_still=True)
                bpy.data.objects.remove(cam)
            if col != "raw":
                for p in shown:
                    bpy.data.objects.remove(p)
        print(f"[render] {s} done", flush=True)


if __name__ == "__main__":
    main()
