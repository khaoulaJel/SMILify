"""04_blender_trait_annotator.py -- place GLAD trait landmarks on the 12 expert-annotated specimens.

EXTENDED 2026-09-07 from 7 to 12 landmarks. The original 7 were designed to MEASURE traits;
these are also FITTING targets, and V5 showed that supervising interior rig pivots buys
anatomy by distorting the surface. Coverage audit: 89% of corpus violations sat on a
surface-constrained trait, but the LEFT mandible was unconstrained (ML/HL is the largest
violation class, 44) and the antennae had NO surface landmark at all -- scape length came
from the an_1->an_2 interior pivots, which would reintroduce the very mechanism the
landmark arm exists to avoid. Added: mandibular_apex_l, antennal_insertion_r/l,
scape_apex_r/l.

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
1. Put the specimen .obj files, the gt_expert *_joints.json files, and
   joint_to_obj_transform.json in one folder.
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
    ("mandibular_apex_l", "Mandibular apex, LEFT", "ML", "RIGHT", ["ma_l"],
     "Tip of the LEFT jaw, mirror of the right one. Start at the ma_l marker (its "
     "articulation) and follow the mandible out to its point."),
    # DEFINITION PINNED after V11. "The socket" is a wide annular depression, and several points
    # on it are equally defensible -- which is why one round landed ~40% of Weber's length from
    # another on Formica while every round agreed to ~2% on Discothyrea. The rule below picks ONE
    # point, so the landmark is repeatable regardless of how large the socket is.
    ("antennal_insertion_r", "Antennal insertion, RIGHT", "SL", "RIGHT", ["an_1_r", "b_h"],
     "Where the antenna TUBE ITSELF emerges from the head, on the right. NOT the rim of the "
     "socket -- the socket is a wide ring and its rim gives a different answer every time. "
     "Follow the antenna inward to the exact point where it leaves the head surface."),
    # V11 measured this landmark at 37.5% repeatability while the RIGHT insertion sat at 2.4%.
    # Tested and it is NOT a left/right swap. An asymmetry that large on a symmetric structure is
    # the VIEW: from full-face the left socket is occluded by the left mandible and scape, so
    # Shift+Right-click lands on whatever is in front of it. Viewed from the LEFT side nothing is
    # between the camera and the socket.
    ("antennal_insertion_l", "Antennal insertion, LEFT", "SL", "BACK", ["an_1_l", "b_h"],
     "Same rule on the LEFT: where the antenna TUBE emerges from the head, not the socket rim. "
     "Shown from the LEFT side (Ctrl+Numpad 1), head on the RIGHT of the screen."),
    ("scape_apex_r", "Scape apex, RIGHT", "SL", "RIGHT", ["an_1_r", "an_2_r"],
     "Far end of the SCAPE -- the first long antenna segment, running from the socket to "
     "the elbow. Place at the elbow end, near the an_2_r marker, ON the surface."),
    ("scape_apex_l", "Scape apex, LEFT", "SL", "RIGHT", ["an_1_l", "an_2_l"],
     "Same elbow end of the LEFT scape."),
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
        # The _joints.json and the .obj are in DIFFERENT frames -- same shape, but the mesh is
        # ~100x larger with permuted axes. Deriving the rotation from the joints and applying it
        # to the mesh (the original bug) left the anchors off the animal and the views wrong.
        # joint_to_obj_transform.json carries the exact similarity per specimen, solved through
        # the fitting pipeline: annotation -> fitter frame (7-DOF fit against the production
        # joints) -> .obj frame (inverse of load_meshes' (v-mean)/max|v|). Verified: joints land
        # 0.5-2.8% of the mesh diagonal inside the surface, which is where interior pivots belong.
        tf_path = os.path.join(folder, "joint_to_obj_transform.json")
        if not os.path.exists(tf_path):
            self.report({"ERROR"}, "joint_to_obj_transform.json missing from the folder")
            return {"CANCELLED"}
        TF = json.load(open(tf_path)).get(sid)
        if TF is None:
            self.report({"ERROR"}, f"no joint->obj transform recorded for {sid}")
            return {"CANCELLED"}
        Mrot = Matrix(TF["R"]).transposed()      # rows are the map's basis; Blender wants columns
        c_src, c_dst, k = Vector(TF["src_centroid"]), Vector(TF["dst_centroid"]), TF["scale"]
        P = {n: list(k * (Mrot @ (Vector(v) - c_src)) + c_dst) for n, v in P.items()}
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
        # Resolve by NAME, not by held reference. Blender's undo rebuilds objects and invalidates
        # any Python pointer kept across it ("StructRNA of type Object has been removed"), which
        # made a single Ctrl+Z break every later step. Names survive undo; stale entries are
        # dropped rather than raising.
        for name in list(_S["joints"]):
            e = bpy.data.objects.get(name)
            if e is None:
                _S["joints"].pop(name, None)
                continue
            try:
                e.hide_viewport = name not in anchors
            except ReferenceError:
                _S["joints"].pop(name, None)
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
            try:
                bpy.ops.trait.focus()
            except Exception:
                pass          # the placement is already stored; never lose it to a view error
        return {"FINISHED"}


def validate(key, placed):
    """Immediate checks, in the CANONICAL frame (+X anterior, right = -Y, +Z dorsal).
    Catches at specimen 1 what would otherwise be found at specimen 12."""
    p = placed[key]
    if key.endswith("_mid") and abs(p.y) > 0.05:
        return f"{key}: y={p.y:+.3f} -- this should be on the MIDLINE (near y=0)"
    # Mandibular APEXES are exempt from the side test. Many ants hold their mandibles crossed or
    # strongly curved, so the tip of the right jaw legitimately sits on the left. Only the
    # ARTICULATION (the ma_r/ma_l joint) cannot cross. Confirmed on Aenictus, whose mandibles are
    # genuinely twisted and crossed -- flagging that was the checker's error, not the annotator's.
    if not key.startswith("mandibular_apex"):
        if key.endswith("_r") and p.y > 0:
            return f"{key}: y={p.y:+.3f} is on the LEFT. Right side is negative y."
        if key.endswith("_l") and p.y < 0:
            return f"{key}: y={p.y:+.3f} is on the RIGHT. Left side is positive y."
    if key in ("mandibular_apex_r", "mandibular_apex_l"):
        cl = placed.get("clypeal_ant_mid")
        if cl and p.x < cl.x:
            return "mandibular apex should stick out FURTHER FORWARD than the clypeal margin"
    if key.startswith("scape_apex_"):
        ins = placed.get("antennal_insertion_" + key[-1])
        if ins and (p - ins).length < 0.02:
            return ("the scape apex is on top of its insertion -- it belongs at the far "
                    "end of the scape, near the elbow")
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
                       coordinate_frame="obj",
                       note="`original` is in the .obj MESH frame. The expert joints file is in a "
                            "different frame (~100x smaller, permuted axes); use "
                            "joint_to_obj_transform.json to bring the joints here, which is what "
                            "this tool now does on load. `canonical` is after the load rotation.",
                       missing=missing, landmarks=out), open(p, "w"), indent=1)
        if missing:
            self.report({"WARNING"}, f"Wrote {p} -- MISSING: {', '.join(missing)}")
        else:
            self.report({"INFO"}, f"Wrote {p} -- all {len(LANDMARKS)} placed")
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
        human = {"RIGHT": "FULL-FACE  (Numpad 3)",
                 "FRONT": "PROFILE, right side  (Numpad 1)",
                 "BACK":  "PROFILE, LEFT side  (Ctrl+Numpad 1)"}.get(view, view)
        box.label(text=human, icon="CAMERA_DATA")
        box.label(text={"RIGHT": "ant is facing you",
                        "FRONT": "head right, gaster left",
                        "BACK":  "head LEFT, gaster right (mirrored)"}.get(view, ""))
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
