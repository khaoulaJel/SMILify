"""04_blender_trait_annotator.py -- place GLAD trait landmarks on the 12 expert-annotated specimens.

WHAT THIS IS FOR
G13 showed joint-derived measurements score R ~ 0 against expert truth. T4 showed surface-derived
traits carry 2.8x more genus signal on identical fits. The missing piece is TRUTH values for the
surface traits on real specimens -- that is what you are producing here.

WHY IT AUTO-ROTATES EVERY SPECIMEN
The 12 scans are in twelve DIFFERENT orientations (verified: head->gaster runs along +Y on six of
them, +X on two, -X on the rest). "The anteriormost point" would otherwise mean a different
direction on every specimen, and no validation would be possible. On load, this tool builds each
specimen's own anatomical frame from Fabian's joints -- lateral from bilateral joint pairs,
antero-posterior from head to gaster -- and rotates mesh and joints into ONE canonical pose:

    +X anterior      right side is -Y      +Z dorsal

So Numpad 1 is always profile, Numpad 3 is always full-face, Numpad 7 is always dorsal, on every
specimen. Placed points are exported in the ORIGINAL scan coordinates (the rotation is undone on
export) so they stay directly comparable to Fabian's joint file.

HOW TO RUN
1. Put the specimen .obj files and the gt_expert *_joints.json files in one folder.
2. Blender > Scripting > open this file > Run Script.
3. Press N in the viewport > "SMIL Traits" tab. Set the folder, pick a specimen, hit Load.

WORKFLOW PER SPECIMEN
The panel walks you through 7 landmarks in a fixed order. For each one it shows what to place, the
view to use, and it displays ONLY the expert joints that are useful for that landmark (the other
~48 are hidden, which is most of what makes this fast). Shift+Right-click on the surface to put the
3D cursor there, then Place. It validates immediately and advances.
"""

import bpy
import json
import os
import math
from mathutils import Vector, Matrix

# ---------------------------------------------------------------------------
# The 7 landmarks. Two traits need no placement at all: FL comes from Fabian's
# l_3_fe -> l_3_ti (12/12 specimens) and SL from an_1 -> an_2 (10/12).
#   (key, label, trait, view, [anchor joints to show], instruction)
# ---------------------------------------------------------------------------
# CONVENTION O1, RESOLVED 2026-09-04: HW INCLUDES THE EYES.
# The published GLAD definition is "maximum head width across the eyes", it needs no judgement
# about where an eye ends, and -- decisively -- the FITTED side has no choice: no eye geometry is
# separable from the head part (0 vertices), so trait_extract.py computes HW as the maximum lateral
# extent of the whole head including any eye bulge. Excluding eyes in the truth while the fit
# includes them would make the comparison measure convention mismatch instead of accuracy, which is
# the b_h failure from RESULTS_G1. Mandibles, antennae and scapes are excluded from HW throughout.
LANDMARKS = [
    ("clypeal_ant_mid", "Anterior clypeal margin", "HL, ML", "RIGHT", ["b_h", "ma_r", "ma_l"],
     "Front of the head, ON THE MIDLINE, just above the gap between the mandibles. "
     "NOT a mandible (they stick out further) and not the little flap behind them."),
    ("cephalic_post_mid", "Posterior cephalic margin", "HL", "FRONT", ["b_h", "b_t"],
     "Back edge of the head capsule, on the midline. Just IN FRONT of the b_h neck marker. "
     "Not inside the neck hole."),
    ("head_width_r", "Max head width, RIGHT", "HW", "RIGHT", ["b_h"],
     "Widest point of the HEAD CAPSULE, right side. INCLUDE the eyes, EXCLUDE mandibles, "
     "antennae and scapes. Convention O1 resolved as 'across the eyes' -- see below."),
    ("head_width_l", "Max head width, LEFT", "HW", "RIGHT", ["b_h"],
     "Same on the left, eyes included. Should be roughly mirror-opposite the right one."),
    ("mandibular_apex_r", "Mandibular apex, RIGHT", "ML", "RIGHT", ["ma_r"],
     "Tip of the right jaw. Start at the ma_r marker (its articulation) and follow the "
     "mandible outward to its point. It is the frontmost thing on the animal."),
    ("wl_anterior_r", "Weber anterior, RIGHT", "WL", "FRONT", ["l_1_co_r", "b_t"],
     "PROFILE view. The mesosoma is the box carrying the legs; l_1_co_r marks its FRONT leg. "
     "Place at the front edge of that box, above and just ahead of the marker, on the right side."),
    ("wl_posterior_r", "Weber posterior, RIGHT", "WL", "FRONT", ["l_3_co_r", "b_a_1"],
     "PROFILE view. l_3_co_r marks the HIND leg. Place just BEHIND and BELOW it, on the side "
     "of the body (not the underside centre line). Same left-right depth as Weber anterior."),
]

# BLENDER'S VIEW NAMES ARE CAMERA-RELATIVE, NOT OBJECT-RELATIVE, and with the canonical frame
# (+X anterior) they come out counter-intuitive. Verified, not assumed:
#   view_axis 'RIGHT' = Numpad 3 = camera at +X looking back = FULL-FACE (the ant faces you;
#                       its right side appears on the LEFT of your screen)
#   view_axis 'FRONT' = Numpad 1 = camera at -Y = the ant's RIGHT SIDE = PROFILE (head to the
#                       right of screen, gaster to the left)
# An earlier version had these swapped, which would have shown profile for the head landmarks.
PAIRS = [("l_1_co_r", "l_1_co_l"), ("l_2_co_r", "l_2_co_l"), ("l_3_co_r", "l_3_co_l"),
         ("ma_r", "ma_l"), ("an_1_r", "an_1_l")]
COLL_J, COLL_L = "SMIL_ExpertJoints", "SMIL_TraitLandmarks"
_S = {"sid": "", "R": None, "origin": None, "joints": {}, "placed": {}, "scale": 1.0}


# ---------------------------------------------------------------------------
def canonical_frame(P):
    """Rotation taking this specimen into +X anterior / -Y right / +Z dorsal.

    Derived from the expert joints, never assumed from coordinate order -- the 12 scans are in 12
    different orientations. Returns None if the anchors are missing.
    """
    lat = Vector((0, 0, 0))
    n = 0
    for r, l in PAIRS:
        if r in P and l in P:
            lat += Vector(P[r]) - Vector(P[l])
            n += 1
    ant = P.get("b_h") or P.get("b_t")          # Eciton has no b_h; b_t is the fallback
    post = P.get("b_a_3") or P.get("b_a_2")
    if n < 2 or ant is None or post is None:
        return None
    lat.normalize()
    ap = (Vector(ant) - Vector(post))
    ap -= ap.dot(lat) * lat
    if ap.length < 1e-9:
        return None
    ap.normalize()
    dv = lat.cross(ap)
    if dv.length < 1e-9:
        return None
    dv.normalize()
    # rows: ap -> +X, -lat -> +Y (so the RIGHT side lands on -Y), dv -> +Z
    R = Matrix((ap, -lat, dv))
    if R.determinant() < 0:
        R = Matrix((ap, -lat, -dv))
    return R


def _clear(name):
    if name in bpy.data.collections:
        c = bpy.data.collections[name]
        for o in list(c.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(c)


def _coll(name):
    _clear(name)
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def _find(folder, sid, exts):
    for f in sorted(os.listdir(folder)):
        stem = os.path.splitext(f)[0]
        if f.lower().endswith(exts) and sid in stem:
            return os.path.join(folder, f)
    return None


def specimen_list(folder):
    out = []
    if folder and os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.endswith("_joints.json"):
                try:
                    sid = json.load(open(os.path.join(folder, f)))["specimen_id"]
                except Exception:
                    continue
                sid = sid.replace("_edited", "")
                half = len(sid) // 2                      # one export doubled its own stem
                if sid[:half] == sid[half:]:
                    sid = sid[:half]
                out.append(sid)
    return sorted(set(out))


# ---------------------------------------------------------------------------
class TraitState(bpy.types.PropertyGroup):
    folder: bpy.props.StringProperty(name="Folder", subtype="DIR_PATH", default="")
    specimen: bpy.props.EnumProperty(
        name="Specimen",
        items=lambda s, c: [(x, x[:34], "") for x in specimen_list(bpy.path.abspath(s.folder))]
        or [("none", "-- set folder --", "")])
    step: bpy.props.IntProperty(default=0, min=0, max=len(LANDMARKS) - 1)


class TRAIT_OT_load(bpy.types.Operator):
    bl_idname = "trait.load"
    bl_label = "Load specimen"

    def execute(self, ctx):
        st = ctx.scene.trait_st
        folder = bpy.path.abspath(st.folder)
        sid = st.specimen
        if not os.path.isdir(folder) or sid in ("", "none"):
            self.report({"ERROR"}, "Set the folder and pick a specimen")
            return {"CANCELLED"}
        jf = _find(folder, sid, (".json",))
        mf = _find(folder, sid, (".obj",))
        if not jf or not mf:
            self.report({"ERROR"}, f"Need both a .obj and a _joints.json for {sid}")
            return {"CANCELLED"}
        P = {j["joint_name"]: j["position"]
             for j in json.load(open(jf))["joints"] if j.get("position") is not None}
        R = canonical_frame(P)
        if R is None:
            self.report({"ERROR"}, "Cannot build an anatomical frame: expert anchors missing")
            return {"CANCELLED"}

        for o in list(bpy.data.objects):
            if o.type == "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        _clear(COLL_J)
        _clear(COLL_L)
        try:
            bpy.ops.wm.obj_import(filepath=mf)
        except Exception:
            bpy.ops.import_scene.obj(filepath=mf)
        mesh = next(o for o in bpy.data.objects if o.type == "MESH")

        origin = Vector(P.get("b_t") or list(P.values())[0])
        # normalise scale too, so the viewport and marker sizes behave the same on every specimen
        pts = [R @ (Vector(v) - origin) for v in P.values()]
        span = max(max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)) or 1.0
        s = 1.0 / span
        mesh.matrix_world = (Matrix.Scale(s, 4) @ R.to_4x4()
                             @ Matrix.Translation(-origin) @ mesh.matrix_world)

        _S.update(sid=sid, R=R, origin=origin, scale=s, joints={}, placed={})
        c = _coll(COLL_J)
        for name, pos in P.items():
            e = bpy.data.objects.new(name, None)
            e.empty_display_type = "SPHERE"
            e.empty_display_size = 0.02
            e.location = (R @ (Vector(pos) - origin)) * s
            e.show_name = True
            c.objects.link(e)
            _S["joints"][name] = e
        _coll(COLL_L)
        st.step = 0
        bpy.ops.trait.focus()
        self.report({"INFO"}, f"{sid} loaded and rotated canonical (+X anterior, right = -Y)")
        return {"FINISHED"}


class TRAIT_OT_focus(bpy.types.Operator):
    """Show only the anchors that help with the current landmark, and set the view."""
    bl_idname = "trait.focus"
    bl_label = "Set view + anchors"

    def execute(self, ctx):
        st = ctx.scene.trait_st
        key, lbl, tr, view, anchors, _ = LANDMARKS[st.step]
        for name, e in _S["joints"].items():
            e.hide_viewport = name not in anchors
        for a in ctx.screen.areas:
            if a.type == "VIEW_3D":
                with ctx.temp_override(area=a, region=next(r for r in a.regions
                                                           if r.type == "WINDOW")):
                    try:
                        bpy.ops.view3d.view_axis(type=view)
                        bpy.ops.view3d.view_all()
                    except Exception:
                        pass
                break
        return {"FINISHED"}


class TRAIT_OT_place(bpy.types.Operator):
    bl_idname = "trait.place"
    bl_label = "Place at cursor"

    def execute(self, ctx):
        st = ctx.scene.trait_st
        if not _S["sid"]:
            self.report({"ERROR"}, "Load a specimen first")
            return {"CANCELLED"}
        key, lbl, tr, view, anchors, _ = LANDMARKS[st.step]
        loc = ctx.scene.cursor.location.copy()
        _S["placed"][key] = loc

        c = bpy.data.collections.get(COLL_L) or _coll(COLL_L)
        if key in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[key], do_unlink=True)
        e = bpy.data.objects.new(key, None)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = 0.025
        e.location = loc
        e.show_name = True
        c.objects.link(e)

        msg = validate(key, _S["placed"])
        if msg:
            self.report({"WARNING"}, msg)
        else:
            self.report({"INFO"}, f"{lbl} placed")
        if st.step < len(LANDMARKS) - 1:
            st.step += 1
            bpy.ops.trait.focus()
        return {"FINISHED"}


def validate(key, placed):
    """Immediate checks, in the CANONICAL frame (+X anterior, right = -Y, +Z dorsal).
    Catches at specimen 1 what would otherwise be found at specimen 12."""
    p = placed[key]
    if key.endswith("_mid") and abs(p.y) > 0.05:
        return f"{key}: y={p.y:+.3f} -- this should be on the MIDLINE (near y=0)"
    if key.endswith("_r") and p.y > 0:
        return f"{key}: y={p.y:+.3f} is on the LEFT. Right side is negative y."
    if key == "head_width_l" and p.y < 0:
        return "head_width_l: this is on the right side. Left is positive y."
    if key == "mandibular_apex_r":
        cl = placed.get("clypeal_ant_mid")
        if cl and p.x < cl.x:
            return "mandibular apex should stick out FURTHER FORWARD than the clypeal margin"
    if key == "cephalic_post_mid":
        cl = placed.get("clypeal_ant_mid")
        if cl and p.x > cl.x:
            return "posterior margin is in FRONT of the clypeal margin -- they look swapped"
    if key == "wl_posterior_r":
        a = placed.get("wl_anterior_r")
        if a:
            if p.x > a.x:
                return "Weber posterior is in FRONT of Weber anterior -- swapped?"
            if abs(p.y - a.y) > 0.06:
                return (f"the two Weber points are {abs(p.y-a.y):.3f} apart sideways -- they must "
                        "sit at the same depth or WL absorbs mesosoma width")
    return ""


class TRAIT_OT_step(bpy.types.Operator):
    bl_idname = "trait.step"
    bl_label = "step"
    delta: bpy.props.IntProperty(default=1)

    def execute(self, ctx):
        st = ctx.scene.trait_st
        st.step = max(0, min(len(LANDMARKS) - 1, st.step + self.delta))
        bpy.ops.trait.focus()
        return {"FINISHED"}


class TRAIT_OT_export(bpy.types.Operator):
    bl_idname = "trait.export"
    bl_label = "Export"

    def execute(self, ctx):
        st = ctx.scene.trait_st
        if not _S["sid"]:
            self.report({"ERROR"}, "Nothing loaded")
            return {"CANCELLED"}
        missing = [k for k, *_ in LANDMARKS if k not in _S["placed"]]
        Rinv = _S["R"].transposed()          # rotation: inverse == transpose
        out = {}
        for k, v in _S["placed"].items():
            orig = Rinv @ (Vector(v) / _S["scale"]) + _S["origin"]
            out[k] = dict(canonical=list(v), original=list(orig))
        p = os.path.join(bpy.path.abspath(st.folder), f"{_S['sid']}_traits.json")
        json.dump(dict(specimen_id=_S["sid"], frame="canonical: +X anterior, right=-Y, +Z dorsal",
                       note="`original` is in the source scan's coordinates, directly comparable "
                            "to the expert joints file. `canonical` is after the rotation applied "
                            "on load.",
                       missing=missing, landmarks=out), open(p, "w"), indent=1)
        if missing:
            self.report({"WARNING"}, f"Wrote {p} -- MISSING: {', '.join(missing)}")
        else:
            self.report({"INFO"}, f"Wrote {p} -- all 7 placed")
        return {"FINISHED"}


class TRAIT_PT_panel(bpy.types.Panel):
    bl_label = "SMIL Traits"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SMIL Traits"

    def draw(self, ctx):
        L = self.layout
        st = ctx.scene.trait_st
        L.prop(st, "folder")
        L.prop(st, "specimen")
        L.operator("trait.load", icon="IMPORT")
        if not _S["sid"]:
            return
        L.separator()
        key, lbl, tr, view, anchors, instr = LANDMARKS[st.step]
        box = L.box()
        box.label(text=f"{st.step+1} of {len(LANDMARKS)}   ->  {tr}", icon="INFO")
        box.label(text=lbl)
        human = ("FULL-FACE  (Numpad 3)" if view == "RIGHT"
                 else "PROFILE, right side  (Numpad 1)")
        box.label(text=human, icon="CAMERA_DATA")
        box.label(text=("ant is facing you" if view == "RIGHT"
                        else "head right, gaster left"))
        for line in _wrap(instr, 40):
            box.label(text=line)
        if anchors:
            box.label(text="showing: " + ", ".join(anchors))
        row = L.row(align=True)
        row.operator("trait.step", text="< prev").delta = -1
        row.operator("trait.step", text="next >").delta = 1
        L.operator("trait.focus", icon="HIDE_OFF")
        L.separator()
        L.label(text="Shift+Right-click the surface, then:")
        L.operator("trait.place", icon="VERTEXSEL",
                   text="Place" + ("  (replaces)" if key in _S["placed"] else ""))
        L.separator()
        done = sum(1 for k, *_ in LANDMARKS if k in _S["placed"])
        L.label(text=f"{done}/{len(LANDMARKS)} placed on {_S['sid'][:22]}")
        for k, l, *_ in LANDMARKS:
            L.label(text=("OK  " if k in _S["placed"] else "--  ") + l, )
        L.operator("trait.export", icon="EXPORT")


def _wrap(s, n):
    words, cur, out = s.split(), "", []
    for w in words:
        if len(cur) + len(w) + 1 > n:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


_CLS = (TraitState, TRAIT_OT_load, TRAIT_OT_focus, TRAIT_OT_place, TRAIT_OT_step,
        TRAIT_OT_export, TRAIT_PT_panel)


def register():
    for c in _CLS:
        try:
            bpy.utils.register_class(c)
        except Exception:
            try:
                bpy.utils.unregister_class(c)
            except Exception:
                pass
            bpy.utils.register_class(c)
    bpy.types.Scene.trait_st = bpy.props.PointerProperty(type=TraitState)


if __name__ == "__main__":
    register()
    print(f"SMIL Traits ready -- {len(LANDMARKS)} landmarks per specimen")
