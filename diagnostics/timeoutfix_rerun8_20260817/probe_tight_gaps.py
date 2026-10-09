"""Headless Blender probe: for each of the 4 specimens that still time out at 900s, compute the
real pre_gap/target_a/target_b via the ACTUAL pipeline function (not a reimplementation - avoids
the OBJ-roundtrip precision trap documented elsewhere in this session), then report local vertex
density around each landmark. Goal: distinguish a genuine tight anatomical pinch point (real,
should stay preserved) from scan noise / an internal duplicate-surface artifact (spurious, should
arguably be cleaned rather than chased with more alpha_wrap compute budget) - the same ambiguity
flagged in prepare_antscan_data_for_mesh_fitting_alphawrap.py's own docstring for a prior specimen
(worker_ALT's Dorylus_sp._CASENT0744698).

Usage: blender --background --python probe_tight_gaps.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, "/home/nao48500/SMILify")
sys.path.insert(0, "/home/nao48500/SMILify/custom_processing")

import bpy
import prepare_antscan_data_for_mesh_fitting_manifold as base

SPECS = {
    "15_Cyphoidris": "diagnostics/moonshot/bench50_clean/Cyphoidris_afrc-tz01_CASENT0744810_processed.obj",
    "25_Lasius": "diagnostics/moonshot/bench50_clean/Lasius_nr._fuliginosus_CASENT0878037_processed.obj",
    "31_Nesomyrmex": "diagnostics/moonshot/bench50_clean/Nesomyrmex_angulatus_CASENT0744820_processed.obj",
    "46_Strumigenys_stenorhina": "diagnostics/moonshot/bench50_clean/Strumigenys_stenorhina_OKENT0105268_processed.obj",
}

for name, rel_path in SPECS.items():
    path = os.path.join("/home/nao48500/SMILify", rel_path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.wm.obj_import(filepath=path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=path)
    obj = bpy.context.selected_objects[0]

    verts, faces = base._triangulated_verts_faces(obj)
    pre_gap, target_a, target_b = base._min_self_approach_gap(verts, faces, return_pair=True)

    # local vertex density: how many OTHER vertices lie within 5x pre_gap of target_a/target_b -
    # a genuine thin-anatomy pinch point should have a modest, locally coherent vertex neighborhood
    # (part of a normal thin tube surface); duplicate/degenerate internal surface noise tends to
    # show an anomalously dense or degenerate local cluster.
    d_a = np.linalg.norm(verts - target_a, axis=1)
    d_b = np.linalg.norm(verts - target_b, axis=1)
    radius = pre_gap * 5.0
    n_near_a = int(np.sum(d_a < radius))
    n_near_b = int(np.sum(d_b < radius))

    print(f"PROBE_RESULT specimen={name} pre_gap={pre_gap:.6g} target_a={list(target_a)} "
          f"target_b={list(target_b)} n_verts_within_5xgap_of_a={n_near_a} "
          f"n_verts_within_5xgap_of_b={n_near_b} total_verts={len(verts)}")
