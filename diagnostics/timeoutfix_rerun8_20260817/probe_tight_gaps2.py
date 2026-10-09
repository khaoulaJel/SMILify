"""Follow-up probe: normal alignment at target_a/target_b. A genuine near-touch between two
distinct solid surfaces should have normals facing roughly OPPOSITE each other (dot product near
-1, they face each other across the gap). A duplicate/overlapping-surface artifact (same patch
counted twice, offset by a tiny amount) would show normals facing roughly the SAME direction (dot
product near +1). This is a much sharper discriminator than local vertex density alone.

Usage: blender --background --python probe_tight_gaps2.py
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

    d_a = np.linalg.norm(verts - target_a, axis=1)
    d_b = np.linalg.norm(verts - target_b, axis=1)
    ia = int(np.argmin(d_a))
    ib = int(np.argmin(d_b))

    # average normal of faces touching vertex ia / ib (area-weighted via cross product magnitude)
    def vertex_normal(vi):
        face_mask = np.any(faces == vi, axis=1)
        tris = verts[faces[face_mask]]
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        n = n / norms
        avg = n.mean(axis=0)
        avg_norm = np.linalg.norm(avg)
        return avg / avg_norm if avg_norm > 1e-9 else avg, face_mask.sum()

    na, nfa = vertex_normal(ia)
    nb, nfb = vertex_normal(ib)
    dot = float(np.dot(na, nb))
    print(f"PROBE2_RESULT specimen={name} pre_gap={pre_gap:.6g} n_faces_at_a={nfa} "
          f"n_faces_at_b={nfb} normal_dot={dot:.4f} "
          f"interpretation={'ARTIFACT-like(parallel)' if dot > 0.3 else ('GENUINE-near-touch(opposing)' if dot < -0.3 else 'AMBIGUOUS')}")
