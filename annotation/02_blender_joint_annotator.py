"""
02_blender_joint_annotator.py

HOW TO RUN:
1. Open Blender.
2. Import the processed .obj for one specimen (File > Import > Wavefront (.obj)).
3. Go to the "Scripting" tab (top of Blender).
4. Open this file (or paste its contents into a new text block).
5. Click "Run Script" (the play button).
6. A new tab called "SMIL Annotate" appears in the 3D viewport sidebar
   (press N in the viewport if the sidebar isn't showing).

WORKFLOW PER SPECIMEN:
1. In the panel, set the Specimen ID (e.g. "specimen_07").
2. The joint list shows every joint from the model schema, in kinematic order,
   grouped by leg/body/antenna. Each entry shows a checkmark once placed.
3. Click a joint in the list to select it as "current".
4. In the 3D viewport, orbit/zoom to the anatomical location, then
   Shift + Right-click on the mesh surface to snap Blender's 3D cursor there.
   (Turn on "Snap cursor to surface": the default Shift+Right-click already
   projects onto the nearest face under the cursor.)
5. In the panel, pick a visibility tag: clear / ambiguous / occluded_estimated / not_present.
6. Click "Place At Cursor" (or, if the structure is missing/unclickable, click
   "Mark Without Placing" after setting the tag to not_present / occluded_estimated).
7. A small labeled Empty appears at that point in the viewport — this is your
   visual record; you can see and review everything placed so far at any time.
8. Repeat for all joints (or as many as are visible/present on this specimen).
9. Click "Export JSON" — writes <specimen_id>_joints.json next to the .blend file
   (or to the path set in the panel).

This tool never asks you to type a joint name. Every placed point is tagged
with the exact integer joint index pulled from the model file in Phase 1 —
comparison against the fitter's output later is a direct index lookup, not a
fuzzy string match.
"""

import bpy
import json
import os

# ---------------------------------------------------------------------------
# Joint schema — pasted directly from the Phase 1 extraction output.
# DO NOT hand-edit these names; if the model file changes, re-run
# 01_extract_joint_schema.py and update this list to match exactly.
# ---------------------------------------------------------------------------
JOINT_SCHEMA = [
    (0, "b_t"), (1, "b_a_1"), (2, "b_a_2"), (3, "b_a_3"), (4, "b_a_4"), (5, "b_a_5"),
    (6, "l_1_co_r"), (7, "l_1_tr_r"), (8, "l_1_fe_r"), (9, "l_1_ti_r"), (10, "l_1_ta_r"), (11, "l_1_pt_r"),
    (12, "l_2_co_r"), (13, "l_2_tr_r"), (14, "l_2_fe_r"), (15, "l_2_ti_r"), (16, "l_2_ta_r"), (17, "l_2_pt_r"),
    (18, "l_3_co_r"), (19, "l_3_tr_r"), (20, "l_3_fe_r"), (21, "l_3_ti_r"), (22, "l_3_ta_r"), (23, "l_3_pt_r"),
    (24, "w_1_r"), (25, "w_2_r"),
    (26, "l_1_co_l"), (27, "l_1_tr_l"), (28, "l_1_fe_l"), (29, "l_1_ti_l"), (30, "l_1_ta_l"), (31, "l_1_pt_l"),
    (32, "l_2_co_l"), (33, "l_2_tr_l"), (34, "l_2_fe_l"), (35, "l_2_ti_l"), (36, "l_2_ta_l"), (37, "l_2_pt_l"),
    (38, "l_3_co_l"), (39, "l_3_tr_l"), (40, "l_3_fe_l"), (41, "l_3_ti_l"), (42, "l_3_ta_l"), (43, "l_3_pt_l"),
    (44, "w_1_l"), (45, "w_2_l"),
    (46, "b_h"),
    (47, "ma_r"), (48, "an_1_r"), (49, "an_2_r"), (50, "an_3_r"),
    (51, "ma_l"), (52, "an_1_l"), (53, "an_2_l"), (54, "an_3_l"),
]

# The four points that build the body-relative reference frame (Phase 2).
# No extra clicks needed — these are already in JOINT_SCHEMA above.
FRAME_JOINTS = {"origin": "b_t", "forward": "b_h", "lateral_r": "l_2_co_r", "lateral_l": "l_2_co_l"}

VISIBILITY_TAGS = [
    ("clear", "Clear", "Landmark is unambiguous and directly visible"),
    ("ambiguous", "Ambiguous", "Landmark is visible but position is uncertain"),
    ("occluded_estimated", "Occluded (estimated)", "Structure is present and captured; position is a best estimate due to self-occlusion/pose"),
    ("corrupted_geometry", "Corrupted geometry", "Geometry exists here but is distorted/wrong — a preprocessing artifact (hole-fill, decimation, etc.), not real anatomy"),
    ("not_present", "Not present", "No reliable point can be placed: structure is absent or was never captured in the raw scan"),
]

NOT_PRESENT_REASONS = [
    ("biological", "Biological (real damage/absence)", "The physical specimen itself lacks this structure (injury, wear, autotomy)"),
    ("scan_capture_gap", "Scan capture gap", "Structure was physically present but the raw scan never captured it (occlusion, thin geometry, reflectivity)"),
    ("uncertain", "Uncertain", "Can't tell whether this is biological absence or a scan capture gap"),
    ("n_a", "N/A", "Not applicable — use for tags other than not_present"),
]

ANNOTATION_COLLECTION_NAME = "SMIL_Joint_Annotations"


# ---------------------------------------------------------------------------
# Scene-level state (stored as custom properties so it survives file save)
# ---------------------------------------------------------------------------
def get_collection():
    coll = bpy.data.collections.get(ANNOTATION_COLLECTION_NAME)
    if coll is None:
        coll = bpy.data.collections.new(ANNOTATION_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(coll)
    return coll


def marker_name(joint_idx, joint_name):
    return f"JT_{joint_idx:02d}_{joint_name}"


class SMIL_JointItem(bpy.types.PropertyGroup):
    idx: bpy.props.IntProperty()
    name: bpy.props.StringProperty()


class SMIL_AnnotatorProps(bpy.types.PropertyGroup):
    specimen_id: bpy.props.StringProperty(name="Specimen ID", default="specimen_01")
    annotator_id: bpy.props.StringProperty(name="Annotator", default="")
    export_dir: bpy.props.StringProperty(name="Export Dir", default="//", subtype='DIR_PATH')
    known_mesh_defects: bpy.props.StringProperty(
        name="Known Mesh Defects",
        default="",
        description="Plain-language note on any preprocessing artifacts/corruption on this specimen's scan (e.g. 'right leg1 distal corrupted by aborted step; right mandible cut by hole-fill'). Leave blank if none."
    )
    current_joint_index: bpy.props.IntProperty(default=0)
    visibility_tag: bpy.props.EnumProperty(name="Visibility", items=VISIBILITY_TAGS, default="clear")
    not_present_reason: bpy.props.EnumProperty(
        name="Reason (if not_present)",
        items=NOT_PRESENT_REASONS,
        default="n_a",
        description="Only relevant when Visibility is 'Not present' — why no reliable point exists"
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class SMIL_OT_place_at_cursor(bpy.types.Operator):
    bl_idname = "smil.place_at_cursor"
    bl_label = "Place At Cursor"
    bl_description = "Drop a joint marker at the current 3D cursor position, tagged with the selected joint and visibility"

    def execute(self, context):
        props = context.scene.smil_annotator_props
        idx = props.current_joint_index
        jname = JOINT_SCHEMA[idx][1]
        coll = get_collection()

        name = marker_name(idx, jname)
        existing = bpy.data.objects.get(name)
        if existing:
            bpy.data.objects.remove(existing, do_unlink=True)

        empty = bpy.data.objects.new(name, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.01
        empty.location = context.scene.cursor.location
        empty["joint_index"] = idx
        empty["joint_name"] = jname
        empty["visibility"] = props.visibility_tag
        empty["not_present_reason"] = props.not_present_reason
        coll.objects.link(empty)

        self.report({'INFO'}, f"Placed {jname} (idx {idx}) — {props.visibility_tag}")
        _advance_to_next_unplaced(props)
        return {'FINISHED'}


class SMIL_OT_mark_without_placing(bpy.types.Operator):
    bl_idname = "smil.mark_without_placing"
    bl_label = "Mark Without Placing"
    bl_description = "Record this joint as not_present / occluded_estimated without a 3D position"

    def execute(self, context):
        props = context.scene.smil_annotator_props
        idx = props.current_joint_index
        jname = JOINT_SCHEMA[idx][1]
        coll = get_collection()

        name = marker_name(idx, jname)
        existing = bpy.data.objects.get(name)
        if existing:
            bpy.data.objects.remove(existing, do_unlink=True)

        empty = bpy.data.objects.new(name, None)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 0.005
        empty.location = (0.0, 0.0, 0.0)
        empty["joint_index"] = idx
        empty["joint_name"] = jname
        empty["visibility"] = props.visibility_tag
        empty["not_present_reason"] = props.not_present_reason
        empty["no_position"] = True
        coll.objects.link(empty)

        self.report({'INFO'}, f"Marked {jname} (idx {idx}) as {props.visibility_tag}, no position")
        _advance_to_next_unplaced(props)
        return {'FINISHED'}


class SMIL_OT_select_joint(bpy.types.Operator):
    bl_idname = "smil.select_joint"
    bl_label = "Select Joint"
    joint_index: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.smil_annotator_props.current_joint_index = self.joint_index
        return {'FINISHED'}


class SMIL_OT_export_json(bpy.types.Operator):
    bl_idname = "smil.export_json"
    bl_label = "Export JSON"
    bl_description = "Write all placed/marked joints for this specimen to a JSON file"

    def execute(self, context):
        props = context.scene.smil_annotator_props
        coll = get_collection()

        records = []
        for idx, jname in JOINT_SCHEMA:
            name = marker_name(idx, jname)
            obj = bpy.data.objects.get(name)
            if obj is None:
                records.append({
                    "joint_index": idx,
                    "joint_name": jname,
                    "position": None,
                    "visibility": "unannotated",
                })
                continue
            rec = {
                "joint_index": idx,
                "joint_name": jname,
                "visibility": obj.get("visibility", "unknown"),
                "not_present_reason": obj.get("not_present_reason", "n_a"),
            }
            if obj.get("no_position", False):
                rec["position"] = None
            else:
                rec["position"] = [obj.location.x, obj.location.y, obj.location.z]
            records.append(rec)

        n_placed = sum(1 for r in records if r["visibility"] != "unannotated")
        n_total = len(records)

        out = {
            "specimen_id": props.specimen_id,
            "annotator_id": props.annotator_id,
            "n_joints_total": n_total,
            "n_joints_annotated": n_placed,
            "known_mesh_defects": props.known_mesh_defects.strip(),
            "frame_joints": FRAME_JOINTS,
            "joints": records,
        }

        export_dir = bpy.path.abspath(props.export_dir)
        os.makedirs(export_dir, exist_ok=True)
        out_path = os.path.join(export_dir, f"{props.specimen_id}_joints.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        self.report({'INFO'}, f"Exported {n_placed}/{n_total} joints to {out_path}")
        return {'FINISHED'}


def _advance_to_next_unplaced(props):
    coll = get_collection()
    n = len(JOINT_SCHEMA)
    start = props.current_joint_index
    for offset in range(1, n + 1):
        idx = (start + offset) % n
        jname = JOINT_SCHEMA[idx][1]
        if bpy.data.objects.get(marker_name(idx, jname)) is None:
            props.current_joint_index = idx
            return
    # everything placed; leave as-is


# ---------------------------------------------------------------------------
# UI Panel
# ---------------------------------------------------------------------------
class SMIL_PT_annotator_panel(bpy.types.Panel):
    bl_label = "SMIL Joint Annotator"
    bl_idname = "SMIL_PT_annotator_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SMIL Annotate"

    def draw(self, context):
        layout = self.layout
        props = context.scene.smil_annotator_props
        coll = get_collection()

        layout.prop(props, "specimen_id")
        layout.prop(props, "annotator_id")
        layout.prop(props, "export_dir")
        layout.prop(props, "known_mesh_defects")
        layout.separator()

        idx, jname = JOINT_SCHEMA[props.current_joint_index]
        box = layout.box()
        box.label(text=f"Current joint: {jname}  (index {idx})", icon='EMPTY_AXIS')
        box.prop(props, "visibility_tag")
        if props.visibility_tag == "not_present":
            box.prop(props, "not_present_reason")
        row = box.row(align=True)
        row.operator("smil.place_at_cursor", icon='CURSOR')
        row.operator("smil.mark_without_placing", icon='X')

        layout.separator()
        layout.operator("smil.export_json", icon='EXPORT')
        layout.separator()

        placed = sum(1 for i, n in JOINT_SCHEMA if bpy.data.objects.get(marker_name(i, n)) is not None)
        layout.label(text=f"Progress: {placed} / {len(JOINT_SCHEMA)}")

        col = layout.column(align=True)
        for i, n in JOINT_SCHEMA:
            done = bpy.data.objects.get(marker_name(i, n)) is not None
            icon = 'CHECKMARK' if done else 'BLANK1'
            row = col.row(align=True)
            op = row.operator("smil.select_joint", text=f"{i:02d}  {n}", icon=icon,
                               depress=(i == props.current_joint_index))
            op.joint_index = i


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
classes = [
    SMIL_JointItem,
    SMIL_AnnotatorProps,
    SMIL_OT_place_at_cursor,
    SMIL_OT_mark_without_placing,
    SMIL_OT_select_joint,
    SMIL_OT_export_json,
    SMIL_PT_annotator_panel,
]


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.smil_annotator_props = bpy.props.PointerProperty(type=SMIL_AnnotatorProps)


def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.smil_annotator_props


register()