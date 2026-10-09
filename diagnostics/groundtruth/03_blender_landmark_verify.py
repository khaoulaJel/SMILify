"""03_blender_landmark_verify.py -- verify / place the GLAD trait landmarks on the template.

SELF-CONTAINED. Copy this one file to any machine with Blender; it needs no other file except the
template mesh itself. The candidate table below is pasted from landmark_candidates.py's output,
the same convention 02_blender_joint_annotator.py uses for JOINT_SCHEMA.

HOW TO RUN
1. Blender > File > Import > Wavefront (.obj) > template_SMIL_OmniAnt.obj
   IMPORTANT: turn OFF every merge / "Split by ..." option in the import panel.
2. Scripting tab > open this file > Run Script.
3. Press N in the 3D viewport > "SMIL Landmarks" tab.

WORKFLOW
- "Check mesh" first. It asserts 10229 vertices. If Blender merged vertices on import the indices
  are meaningless and everything downstream is wrong. This is the guard for that, and it is not
  optional -- run it before you place anything.
- "Show candidates" drops a labelled sphere at each proposal so you can see all ten at once.
- To correct one: Edit Mode, select ONE vertex, back to Object Mode, pick the landmark in the
  dropdown, "Set From Selected". The index is read from the mesh, never typed.
- "Export" writes landmark_template_indices.json -- the file the extraction layer reads.

The two Weber's-length endpoints are computed by RULE rather than placed by hand: they are the
anterior and posteroventral extremes of the mesosoma silhouette in RIGHT PROFILE, both constrained
to the same lateral band so the diagonal cannot pick up mesosoma width. They are therefore Type III
constructed points, not Type I, and must be reported on the Type III axis. They are drawn here only
so the rule can be sanity-checked by eye -- confirm the anterior one is on the front of the pronotum
and not the neck, and that the posterior one is on the lateral metapleuron and NOT the midline keel.
An earlier midline-based proposal for the posterior point was rejected for exactly that reason.
See LANDMARK_PROTOCOL.md section 5 and Appendix A.
"""

import bpy
import json
import os

EXPECTED_VERTS = 10229
OUT = "landmark_template_indices.json"
COLL = "SMIL_Landmarks"

# ---------------------------------------------------------------------------
# Candidate table -- pasted from landmark_candidates.py. Fields:
#   (name, candidate_vertex, confidence, short definition / what to check)
# A candidate_vertex of -1 means NO usable proposal exists: place it yourself.
# DO NOT hand-edit indices here. Correct them in Blender and re-export.
# ---------------------------------------------------------------------------
CANDIDATES = [
    ("clypeal_margin_ant_mid", 773, "medium",
     "HL anterior endpoint. Anteriormost midline point of the head. Check it is on the median "
     "clypeal margin, NOT a mandible or the labrum."),
    ("cephalic_margin_post_mid", 511, "medium",
     "HL posterior endpoint. Midpoint of the occipital margin. Check it is not inside the neck "
     "or the occipital foramen."),
    ("mandibular_apex_r", 1662, "high",
     "Apical tooth of the right mandible. HIGHEST-RISK point: the template is a MEAN mandible, "
     "so correspondence may fail on trap-jaw (Odontomachus) and long-mandible (Aenictus) forms."),
    ("antennal_insertion_r", 585, "medium",
     "Centre of the right torulus, where the scape articulates with the head."),
    ("scape_apex_r", 1800, "medium",
     "Distal end of the right scape. SL EXCLUDES the basal condyle and neck, so the proximal "
     "endpoint is the articulation, not the basal bulb."),
    ("wl_anterior_r", 943, "RULE",
     "Weber's length anterior endpoint, computed by RULE not placed by hand: anteriormost mesosoma "
     "vertex in the right lateral band (y in -0.098..-0.055), excluding the neck. This is a Type III "
     "constructed point, not Type I. Shown so you can sanity-check the rule found the front of the "
     "pronotum in profile and not the neck."),
    ("wl_posterior_r", 2336, "RULE",
     "Weber's length posterior endpoint, computed by RULE not placed by hand: posteroventral-most "
     "mesosoma vertex in the same right lateral band, just behind and below the hind coxa. Type III, "
     "not Type I. Sanity-check it is on the metapleuron and NOT on the midline keel."),
    ("petiole_ant_mid", 2699, "medium",
     "Anterior margin of the petiole, on the VENTRAL midline outline. SAME-LINE RULE: both petiole "
     "points must sit on the same outline or the 'length' is a diagonal. The earlier candidate 5144 "
     "was on the DORSAL outline, making 60% of the measured length vertical rather than longitudinal."),
    ("petiole_post_mid", 5158, "medium",
     "Posterior margin of the petiole, on the same VENTRAL midline outline as petiole_ant_mid. The "
     "ventral row spans the full segment (-0.153 to -0.231); the dorsal row stops short at -0.188, "
     "so the two outlines genuinely disagree about where the petiole ends."),
    ("gaster_apex_mid", 5193, "medium",
     "Posteriormost midline point of the gaster (TBL endpoint). Taken over b_a_3 + b_a_4 because "
     "b_a_5 carries no vertices at all."),
]

_LM = {n: dict(vertex=v, confidence=c, note=t,
               source="candidate" if v >= 0 else "UNPLACED")
       for n, v, c, t in CANDIDATES}
_ORDER = [n for n, _, _, _ in CANDIDATES]


def _items(self=None, ctx=None):
    out = []
    for n in _ORDER:
        d = _LM[n]
        mark = "OK " if d["source"] == "placed" else ("-- " if d["vertex"] < 0 else "?? ")
        out.append((n, mark + n, d["note"]))
    return out


def _mesh():
    o = bpy.context.active_object
    if o and o.type == "MESH" and len(o.data.vertices) == EXPECTED_VERTS:
        return o
    for o in bpy.data.objects:
        if o.type == "MESH" and len(o.data.vertices) == EXPECTED_VERTS:
            return o
    o = bpy.context.active_object
    return o if (o and o.type == "MESH") else None


def _default_dir():
    if bpy.data.filepath:
        return os.path.dirname(bpy.data.filepath)
    for o in bpy.data.objects:
        if o.type == "MESH":
            return os.path.expanduser("~")
    return os.path.expanduser("~")


class SMIL_State(bpy.types.PropertyGroup):
    current: bpy.props.EnumProperty(name="Landmark", items=_items)
    outdir: bpy.props.StringProperty(name="Out dir", default=_default_dir(), subtype="DIR_PATH")


def _draw_empties():
    if COLL in bpy.data.collections:
        c = bpy.data.collections[COLL]
        for o in list(c.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(c)
    m = _mesh()
    if not m:
        return
    c = bpy.data.collections.new(COLL)
    bpy.context.scene.collection.children.link(c)
    for n in _ORDER:
        v = _LM[n]["vertex"]
        if v < 0 or v >= len(m.data.vertices):
            continue
        e = bpy.data.objects.new(n, None)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = 0.012
        e.show_name = True
        c.objects.link(e)
        # VERTEX-parent the marker to the template, not a baked world position. An empty placed
        # at `matrix_world @ co` is an independent object: move or rotate the mesh and the
        # markers stay behind, silently pointing at nothing. Vertex parenting also makes the
        # marker follow the vertex under mesh edits, which is a live check that the index is
        # the one you think it is -- nudge a vertex and the right sphere must move with it.
        try:
            e.parent = m
            e.parent_type = "VERTEX"
            e.parent_vertices = (v, v, v)   # only [0] is used for parent_type VERTEX
            e.matrix_parent_inverse.identity()
            e.location = (0.0, 0.0, 0.0)
        except Exception:
            # Fallback for Blender builds that reject vertex parenting on an Empty: object
            # parenting still travels with the mesh, it just does not track deformation.
            e.parent = m
            e.parent_type = "OBJECT"
            e.matrix_parent_inverse = m.matrix_world.inverted()
            e.location = m.matrix_world @ m.data.vertices[v].co


class SMIL_OT_check(bpy.types.Operator):
    bl_idname = "smil.check_mesh"
    bl_label = "Check mesh"

    def execute(self, ctx):
        m = _mesh()
        if not m:
            self.report({"ERROR"}, "No mesh in the scene -- import the template OBJ first")
            return {"CANCELLED"}
        n = len(m.data.vertices)
        if n != EXPECTED_VERTS:
            self.report({"ERROR"}, "%d verts, expected %d -- INDICES INVALID. Re-import the OBJ "
                                   "with every merge/split option OFF." % (n, EXPECTED_VERTS))
            return {"CANCELLED"}
        self.report({"INFO"}, "OK: %d vertices, indices valid" % n)
        return {"FINISHED"}


class SMIL_OT_show(bpy.types.Operator):
    bl_idname = "smil.show"
    bl_label = "Show candidates"

    def execute(self, ctx):
        _draw_empties()
        self.report({"INFO"}, "Candidates drawn")
        return {"FINISHED"}


class SMIL_OT_set(bpy.types.Operator):
    bl_idname = "smil.set_from_selected"
    bl_label = "Set From Selected"

    def execute(self, ctx):
        m = _mesh()
        if not m:
            self.report({"ERROR"}, "No mesh")
            return {"CANCELLED"}
        if ctx.mode == "EDIT_MESH":
            self.report({"ERROR"}, "Switch to Object Mode first -- selection only syncs on mode change")
            return {"CANCELLED"}
        sel = [v.index for v in m.data.vertices if v.select]
        if len(sel) != 1:
            self.report({"ERROR"}, "Select exactly ONE vertex (%d selected)" % len(sel))
            return {"CANCELLED"}
        name = ctx.scene.smil_lm.current
        _LM[name]["vertex"] = sel[0]
        _LM[name]["source"] = "placed"
        _draw_empties()
        self.report({"INFO"}, "%s = vertex %d" % (name, sel[0]))
        return {"FINISHED"}


class SMIL_OT_export(bpy.types.Operator):
    bl_idname = "smil.export"
    bl_label = "Export"

    def execute(self, ctx):
        d = bpy.path.abspath(ctx.scene.smil_lm.outdir) or _default_dir()
        p = os.path.join(d, OUT)
        unplaced = [n for n in _ORDER if _LM[n]["vertex"] < 0]
        payload = dict(model="3D_model_prep/SMIL_OmniAnt.pkl",
                       n_vertices=EXPECTED_VERTS,
                       protocol="LANDMARK_PROTOCOL.md v0.1",
                       unplaced=unplaced,
                       landmarks={n: _LM[n] for n in _ORDER})
        with open(p, "w") as fh:
            json.dump(payload, fh, indent=1)
        if unplaced:
            self.report({"WARNING"}, "Wrote %s -- STILL UNPLACED: %s" % (p, ", ".join(unplaced)))
        else:
            self.report({"INFO"}, "Wrote %s -- all %d placed" % (p, len(_ORDER)))
        return {"FINISHED"}


class SMIL_PT_panel(bpy.types.Panel):
    bl_label = "SMIL Landmarks"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SMIL Landmarks"

    def draw(self, ctx):
        L = self.layout
        st = ctx.scene.smil_lm
        L.operator("smil.check_mesh", icon="CHECKMARK")
        L.operator("smil.show", icon="HIDE_OFF")
        L.separator()
        L.prop(st, "current")
        cur = _LM.get(st.current)
        if cur:
            box = L.box()
            box.label(text="index: %s" % (cur["vertex"] if cur["vertex"] >= 0 else "UNPLACED"))
            box.label(text="source: %s" % cur["source"])
            box.label(text="confidence: %s" % cur["confidence"])
            for line in cur["note"].split(". "):
                if line.strip():
                    box.label(text=line.strip()[:58])
        L.operator("smil.set_from_selected", icon="VERTEXSEL")
        L.separator()
        L.prop(st, "outdir")
        L.operator("smil.export", icon="EXPORT")
        done = sum(1 for n in _ORDER if _LM[n]["vertex"] >= 0)
        L.label(text="%d/%d placed" % (done, len(_ORDER)))


_CLS = (SMIL_State, SMIL_OT_check, SMIL_OT_show, SMIL_OT_set, SMIL_OT_export, SMIL_PT_panel)


def register():
    for c in _CLS:
        try:
            bpy.utils.register_class(c)
        except Exception:
            bpy.utils.unregister_class(c)
            bpy.utils.register_class(c)
    bpy.types.Scene.smil_lm = bpy.props.PointerProperty(type=SMIL_State)


if __name__ == "__main__":
    register()
    _draw_empties()
    print("SMIL Landmarks ready: %d landmarks, %d unplaced"
          % (len(_ORDER), sum(1 for n in _ORDER if _LM[n]["vertex"] < 0)))
