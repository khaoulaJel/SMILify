"""Recompute pre_gap/target_a/target_b + near-touch normal_dot for every specimen's saved
PRE_RECONSTRUCTION.obj, to test normal_dot (near-touch angle, not just distance) as a candidate
predictor of final_relative_alpha - the gap_ratio-based power-law fit only explains ~37% of
variance, this checks whether the ANGLE at the near-touch point explains more of the residual.

Usage: blender --background --python probe_normal_dot_all.py
"""
import sys
import os
import glob
import re
import numpy as np

sys.path.insert(0, "/home/nao48500/SMILify")
sys.path.insert(0, "/home/nao48500/SMILify/custom_processing")

import bpy
import prepare_antscan_data_for_mesh_fitting_manifold as base
import prepare_antscan_data_for_mesh_fitting_alphawrap as aw

RUN_DIR = "/home/nao48500/SMILify/diagnostics/phase1_corpus_20260817"
paths = sorted(glob.glob(f"{RUN_DIR}/tmp/task_*/alphawrap_prerecon_diagnostics/*_PRE_RECONSTRUCTION.obj"))

for path in paths:
    name = os.path.basename(path).replace("_PRE_RECONSTRUCTION.obj", "")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.wm.obj_import(filepath=path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=path)
    obj = bpy.context.selected_objects[0]

    verts, faces = base._triangulated_verts_faces(obj)
    pre_gap, target_a, target_b = base._min_self_approach_gap(verts, faces, return_pair=True)
    nt = aw._classify_near_touch(verts, faces, target_a, target_b)
    print(f"PROBE3_RESULT specimen={name} pre_gap={pre_gap:.6g} normal_dot={nt['normal_dot']:.4f} "
          f"interpretation={nt['interpretation']}")
