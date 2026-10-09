"""Recompute pre_gap/target_a/target_b on Strumigenys_alberti's pre_reconstruction mesh, and check
which raw connected component each landmark point sits on (or nearest to, if the exact vertex
isn't in this mesh) - a proxy for whether the tracked near-touch is a real cross-part junction
(e.g. leg vs gaster, different scan fragments) vs a same-part measurement.

Usage: blender --background --python probe_landmark_location.py
"""
import sys
sys.path.insert(0, "/home/nao48500/SMILify")
sys.path.insert(0, "/home/nao48500/SMILify/custom_processing")

import bpy
import numpy as np
import trimesh
import prepare_antscan_data_for_mesh_fitting_manifold as base

PATH = "/home/nao48500/SMILify/diagnostics/eyeball_groundtruth_20260817/06_FIDFAIL_Strumigenys_alberti/pre_reconstruction.obj"

bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.wm.obj_import(filepath=PATH)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=PATH)
obj = bpy.context.selected_objects[0]

verts, faces = base._triangulated_verts_faces(obj)
pre_gap, target_a, target_b = base._min_self_approach_gap(verts, faces, return_pair=True)
print(f"PROBE_LANDMARK pre_gap={pre_gap:.6g} target_a={list(target_a)} target_b={list(target_b)}")

mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
edges = mesh.edges_unique
n = len(mesh.vertices)
adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
adj = adj + adj.T
n_comp, labels = connected_components(adj, directed=False)
print(f"PROBE_LANDMARK n_components={n_comp}")
for i in range(n_comp):
    comp_verts = mesh.vertices[labels == i]
    d_a = np.min(np.linalg.norm(comp_verts - target_a, axis=1))
    d_b = np.min(np.linalg.norm(comp_verts - target_b, axis=1))
    bmin, bmax = comp_verts.min(axis=0), comp_verts.max(axis=0)
    print(f"PROBE_LANDMARK component={i} n_verts={len(comp_verts)} "
          f"dist_to_target_a={d_a:.4g} dist_to_target_b={d_b:.4g} "
          f"bbox_extent={list(bmax-bmin)} bbox_center={list((bmin+bmax)/2)}")

# overall mesh bbox for context (helps interpret which end is head/gaster/legs)
overall_bbox = mesh.bounds
print(f"PROBE_LANDMARK overall_bbox_min={list(overall_bbox[0])} overall_bbox_max={list(overall_bbox[1])}")
