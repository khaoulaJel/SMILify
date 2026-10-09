"""
08_fabian_review_tool.py

Minimal review tool. Your only job with this:

1. Open the .blend file.
2. Object Mode (default). Click any small sphere marker you think is in
   the wrong place.
3. Press G, move the mouse to the correct spot, click the LEFT mouse
   button to confirm. (Snap-to-Volume is already configured in the file —
   the marker will snap toward the center of the mesh under your cursor
   as you drag.)
4. Repeat for any joints you want to correct. Nothing else needs touching.
5. Scripting tab, paste this file, Run Script. A "SMIL Review" tab appears
   in the 3D viewport sidebar (press N if you don't see the sidebar).
6. Click "Export JSON". Done — only the JSON is written, no extra files.

You do NOT need to: create anything, touch the armature/bones, enter Edit
Mode, or change any dropdown. Just drag spheres and export.
"""

import bpy
import json
import os

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

FRAME_JOINTS = {"origin": "b_t", "forward": "b_h", "lateral_r": "l_2_co_r", "lateral_l": "l_2_co_l"}


def _marker_name(idx, jname):
    return f"MARKER_{idx:02d}_{jname}"


def _find_target_mesh():
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    return meshes[0] if len(meshes) == 1 else None


class SMIL_OT_export_json_review(bpy.types.Operator):
    bl_idname = "smil.export_json_review"
    bl_label = "Export JSON"
    bl_description = "Write the corrected joint positions to JSON only"

    def execute(self, context):
        props = context.scene.smil_annotator_props
        tags = context.scene.smil_joint_tags

        mesh_obj = _find_target_mesh()
        if mesh_obj is not None:
            loc, rot = mesh_obj.location, mesh_obj.rotation_euler
            if any(abs(v) > 1e-4 for v in loc) or any(abs(v) > 1e-4 for v in rot):
                self.report({'ERROR'},
                            f"Export ABORTED — the mesh isn't at its original identity transform "
                            f"anymore (loc={tuple(round(v,3) for v in loc)}, "
                            f"rot={tuple(round(v,3) for v in rot)}). Don't move/rotate the mesh "
                            f"itself — only drag the small sphere markers.")
                return {'CANCELLED'}

        records = []
        for idx, jname in JOINT_SCHEMA:
            tag = next((t for t in tags if t.joint_index == idx), None)
            if tag is None or not tag.confirmed:
                records.append({"joint_index": idx, "joint_name": jname,
                                 "position": None, "visibility": "unannotated"})
                continue
            rec = {"joint_index": idx, "joint_name": jname, "visibility": tag.visibility}
            if tag.visibility == "not_present":
                rec["not_present_reason"] = tag.not_present_reason
                rec["position"] = None
            else:
                marker = bpy.data.objects.get(_marker_name(idx, jname))
                if marker is not None:
                    loc = marker.location
                    rec["position"] = [loc.x, loc.y, loc.z]
                else:
                    rec["position"] = None  # shouldn't happen, but never fabricate a position
            records.append(rec)

        n_placed = sum(1 for r in records if r["visibility"] != "unannotated")
        n_total = len(records)

        out = {
            "specimen_id": props.specimen_id,
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


class SMIL_PT_review_panel(bpy.types.Panel):
    bl_label = "SMIL Review (Fabian)"
    bl_idname = "SMIL_PT_review_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SMIL Review"

    def draw(self, context):
        layout = self.layout
        props = context.scene.smil_annotator_props

        box = layout.box()
        box.label(text="To correct a joint:", icon='INFO')
        box.label(text="Click the small sphere.")
        box.label(text="Press G, move, click to confirm.")
        box.label(text="That's it — nothing else to touch.")
        layout.separator()

        layout.prop(props, "specimen_id")
        layout.prop(props, "export_dir")
        layout.separator()
        layout.operator("smil.export_json_review", icon='EXPORT')


classes = [SMIL_OT_export_json_review, SMIL_PT_review_panel]


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)


register()
