"""Task 1 / P5: does V1's 20% face-loss abort fire on a bad weld? Run inside Blender (background).

Constructed cases, each passed to V1's own apply_modifiers (imported unchanged from versions/):
  A  two parallel grid sheets (1 x 1 spacing) 0.1 apart, explicit weld distance 0.3
     -> cross-surface fusion: every vertex of sheet A merges with its partner on sheet B
  B  one grid sheet with 0.1 spacing, explicit weld distance 0.3
     -> within-surface collapse: neighbouring vertices of the SAME sheet merge, triangles degenerate
  C  control: two sheets 2.0 apart, weld 0.3 -> nothing should merge
Reports faces before/after the weld, the code's own face-loss %, whether RuntimeError (the abort) was
raised, and whether the two sheets became one connected piece.
"""
import os
import sys

import bmesh
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "versions"))
import V1_fixbranch as V1  # noqa: E402


def grid(bm, n, spacing, z):
    vs = [[bm.verts.new((i * spacing, j * spacing, z)) for j in range(n)] for i in range(n)]
    for i in range(n - 1):
        for j in range(n - 1):
            bm.faces.new((vs[i][j], vs[i + 1][j], vs[i + 1][j + 1]))
            bm.faces.new((vs[i][j], vs[i + 1][j + 1], vs[i][j + 1]))


def make(name, sheets, n, spacing):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    for z in sheets:
        grid(bm, n, spacing, z)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob; ob.select_set(True)
    return ob


def pieces(ob):
    bm = bmesh.new(); bm.from_mesh(ob.data); bm.verts.ensure_lookup_table()
    seen, n = set(), 0
    for v in bm.verts:
        if v.index in seen:
            continue
        n += 1; stack = [v]
        while stack:
            x = stack.pop()
            if x.index in seen:
                continue
            seen.add(x.index); stack.extend(e.other_vert(x) for e in x.link_edges)
    bm.free(); return n


cases = {"A_two_sheets_gap0.1": ([0.0, 0.1], 30, 1.0), "B_one_sheet_spacing0.1": ([0.0], 30, 0.1),
         "C_control_gap2.0": ([0.0, 2.0], 30, 1.0)}
for name, (sheets, n, spacing) in cases.items():
    ob = make(name, sheets, n, spacing)
    f0, p0 = len(ob.data.polygons), pieces(ob)
    try:
        V1.apply_modifiers(ob, weld_merge_threshold=0.3, fill_holes_sides=0)
        aborted = False
    except RuntimeError as e:
        aborted = True
        print(f"[abort] {name}: RuntimeError: {str(e)[:120]}")
    f1, p1 = len(ob.data.polygons), pieces(ob)
    print(f"[P5] {name}: faces {f0} -> {f1} ({100 * (f0 - f1) / f0:.1f}% loss), pieces {p0} -> {p1}, "
          f"abort raised: {aborted}", flush=True)
