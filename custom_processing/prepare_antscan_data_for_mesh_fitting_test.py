import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
import math
import numpy as np
import os
import time
import sys
import random
import json
from collections import deque, Counter


def _median_edge_length(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    lengths = sorted(edge.calc_length() for edge in bm.edges)
    bpy.ops.object.mode_set(mode="OBJECT")

    if not lengths:
        return 0.0
    mid = len(lengths) // 2
    if len(lengths) % 2:
        return lengths[mid]
    return (lengths[mid - 1] + lengths[mid]) / 2


def apply_modifiers(
    obj,
    edge_split_angle=1.5708,
    weld_merge_threshold=None,
    dissolve_angle_limit=0.01,
    fill_holes_sides=0,
    min_island_faces=4,
    max_weld_face_loss_pct=20.0,
    verbose_diagnostics=False,
    run_edge_split=True,
    run_weld=True,
):
    def _checkpoint(label):
        if verbose_diagnostics:
            print(f"  [apply_modifiers] holes after {label}: {count_holes(obj)}")

    if weld_merge_threshold is None:
        bbox_size = obj.dimensions
        max_dimension = max(bbox_size)
        bbox_threshold = max_dimension * 0.002
        print(f"Bounding box size: {bbox_size}, bbox-based threshold: {bbox_threshold}")

        median_edge = _median_edge_length(obj)
        local_threshold = median_edge * 0.3
        print(f"Median edge length: {median_edge}, local-density-based threshold: {local_threshold}")

        if median_edge > 0:
            weld_merge_threshold = min(bbox_threshold, local_threshold)
        else:
            weld_merge_threshold = bbox_threshold

        if weld_merge_threshold <= 0:
            print("Warning: Calculated weld_merge_threshold is 0 or negative. Using default value.")
            weld_merge_threshold = 2

        print(f"Using weld_merge_threshold: {weld_merge_threshold}")

    _checkpoint("entry (before any sub-step)")

    if run_edge_split:
        # EdgeSplit exists to break shading continuity at hard edges for rendering (crisp corners
        # instead of smooth-shaded ones) - a viewport/render concern, irrelevant to a mesh headed
        # into registration. On a voxel-remeshed organic surface (naturally full of >90-degree
        # face-angle transitions - it's curvy, faceted geometry, not flat CAD panels) it fires
        # constantly and duplicates vertices along thousands of "hard" edges, most of which are
        # nothing more than normal surface curvature. Confirmed via verbose_diagnostics on this
        # exact pipeline: on an already-watertight (0-hole) voxel-remeshed mesh, EdgeSplit alone
        # introduced ~2116 holes, most (not all) later re-closed by Weld. Skip it entirely
        # (run_edge_split=False) when the input is already watertight and shading fidelity doesn't
        # matter for the downstream use case - which is exactly the winding_number branch's case.
        edge_split = obj.modifiers.new(name="EdgeSplit", type="EDGE_SPLIT")
        edge_split.split_angle = edge_split_angle
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")
    _checkpoint("EdgeSplit" if run_edge_split else "EdgeSplit (skipped)")

    removed_islands = filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) before Weld")
    _checkpoint("filter_small_components (pre-Weld)")

    face_count_before_weld = len(obj.data.polygons)
    if run_weld:
        weld = obj.modifiers.new(name="Weld", type="WELD")
        weld.merge_threshold = weld_merge_threshold

        bpy.ops.object.modifier_apply(modifier="Weld")
    else:
        print("Skipping Weld step")

    face_count_after_weld = len(obj.data.polygons)
    _checkpoint("Weld")

    if face_count_before_weld > 0:
        face_loss_pct = 100 * (face_count_before_weld - face_count_after_weld) / face_count_before_weld
        print(f"Weld: {face_count_before_weld} -> {face_count_after_weld} faces ({face_loss_pct:.1f}% loss)")
        if face_loss_pct > max_weld_face_loss_pct:
            raise RuntimeError(
                f"Weld step destroyed {face_loss_pct:.1f}% of faces "
                f"({face_count_before_weld} -> {face_count_after_weld}), exceeding the "
                f"{max_weld_face_loss_pct}% threshold. This indicates Weld merged vertices across "
                f"disconnected mesh regions rather than within a single surface. Aborting rather than "
                f"silently continuing with a corrupted mesh."
            )

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.ops.holes_fill(bm, edges=bm.edges, sides=fill_holes_sides)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")
    _checkpoint("holes_fill")

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.ops.dissolve_limit(
        bm, angle_limit=dissolve_angle_limit, use_dissolve_boundaries=False, verts=bm.verts, edges=bm.edges
    )
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")
    bm.free()
    _checkpoint("dissolve_limit")


def _bfs_path(start_vert, target_verts, max_hops=60):
    parent = {start_vert: None}
    depth = {start_vert: 0}
    queue = deque([start_vert])
    while queue:
        current = queue.popleft()
        if current in target_verts and current is not start_vert:
            path = []
            v = current
            while v is not None:
                path.append(v)
                v = parent[v]
            return path
        if depth[current] >= max_hops:
            continue
        for edge in current.link_edges:
            neighbor = edge.other_vert(current)
            if neighbor not in parent:
                parent[neighbor] = current
                depth[neighbor] = depth[current] + 1
                queue.append(neighbor)
    return None


def bridge_nearby_islands(
    bm, vertices_to_keep, min_bridge_island_faces=50, prefilter_gap_multiplier=25, max_bridge_hops=20,
    patch_rings=2,
):
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    kept_faces = [f for f in bm.faces if all(v.index in vertices_to_keep for v in f.verts)]
    unvisited = set(kept_faces)
    islands = []
    while unvisited:
        seed = unvisited.pop()
        island_faces = {seed}
        to_visit = [seed]
        while to_visit:
            current = to_visit.pop()
            for edge in current.edges:
                for linked_face in edge.link_faces:
                    if linked_face in unvisited:
                        unvisited.remove(linked_face)
                        island_faces.add(linked_face)
                        to_visit.append(linked_face)
        island_verts = set()
        for f in island_faces:
            island_verts.update(f.verts)
        islands.append((island_verts, island_faces))

    real_islands = [(verts, faces) for verts, faces in islands if len(faces) >= min_bridge_island_faces]
    if len(real_islands) < 2:
        return 0

    def boundary_verts(verts, faces):
        result = set()
        for f in faces:
            for edge in f.edges:
                linked_in_island = sum(1 for lf in edge.link_faces if lf in faces)
                if linked_in_island < len(edge.link_faces):
                    result.update(edge.verts)
        return result if result else verts

    real_islands = [(verts, boundary_verts(verts, faces)) for verts, faces in real_islands]

    edge_lengths = sorted(e.calc_length() for e in bm.edges)
    median_edge = edge_lengths[len(edge_lengths) // 2] if edge_lengths else 1.0
    prefilter_threshold = prefilter_gap_multiplier * median_edge

    def bbox(verts):
        xs = [v.co.x for v in verts]
        ys = [v.co.y for v in verts]
        zs = [v.co.z for v in verts]
        return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))

    boxes = [bbox(boundary) for _, boundary in real_islands]

    def bbox_gap(box_a, box_b):
        gap = 0.0
        for axis in range(3):
            lo = max(box_a[0][axis], box_b[0][axis]) - min(box_a[1][axis], box_b[1][axis])
            gap += max(lo, 0.0) ** 2
        return math.sqrt(gap)

    bridged = 0
    for i in range(len(real_islands)):
        for j in range(i + 1, len(real_islands)):
            island_a_verts, boundary_a = real_islands[i]
            island_b_verts, boundary_b = real_islands[j]

            if bbox_gap(boxes[i], boxes[j]) > prefilter_threshold:
                continue

            best_dist, best_vert = None, None
            for va in boundary_a:
                for vb in boundary_b:
                    d = (va.co - vb.co).length
                    if best_dist is None or d < best_dist:
                        best_dist, best_vert = d, va

            if best_dist is None or best_dist > prefilter_threshold:
                continue

            path = _bfs_path(best_vert, island_b_verts, max_hops=max_bridge_hops)
            if path is None:
                continue

            patch_verts = set(path)
            frontier = set(path)
            for _ in range(patch_rings):
                next_frontier = set()
                for v in frontier:
                    for edge in v.link_edges:
                        neighbor = edge.other_vert(v)
                        if neighbor not in patch_verts:
                            patch_verts.add(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier

            for v in patch_verts:
                vertices_to_keep.add(v.index)

            bridged += 1
            print(f"Bridged islands ({len(island_a_verts)} and {len(island_b_verts)} verts): "
                  f"gap={best_dist:.2f}, path_hops={len(path) - 1}, patch_verts={len(patch_verts)}")

    return bridged


def clean_internal_geometry(
    obj,
    ray_density=1000,
    secondary_rays=50,
    random_seed=0,
    keep_rings=2,
    min_island_faces=4,
    min_bridge_island_faces=50,
    max_bridge_hops=20,
):
    np.random.seed(random_seed)
    random.seed(random_seed)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_min = Vector(map(min, zip(*bbox_corners)))
    bbox_max = Vector(map(max, zip(*bbox_corners)))

    center = (bbox_max + bbox_min) / 2
    radius = (bbox_max - bbox_min).length * 2

    def cast_ray(origin, direction):
        hit, loc, norm, face_index = obj.ray_cast(obj.matrix_world.inverted() @ origin, direction)
        return hit, face_index

    def add_face_and_connected(face, vertices_to_keep):
        frontier = {face}
        visited_faces = {face}
        for _ in range(keep_rings):
            next_frontier = set()
            for f in frontier:
                for edge in f.edges:
                    for linked_face in edge.link_faces:
                        if linked_face not in visited_faces:
                            visited_faces.add(linked_face)
                            next_frontier.add(linked_face)
            frontier = next_frontier
            if not frontier:
                break
        for f in visited_faces:
            for vert in f.verts:
                vertices_to_keep.add(vert.index)

    vertices_to_keep = set()

    phi = np.linspace(0, 2 * np.pi, int(np.sqrt(ray_density)))
    theta = np.linspace(0, np.pi, int(np.sqrt(ray_density)))

    for p in phi:
        for t in theta:
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)

            origin = center + Vector((x, y, z))
            main_direction = (center - origin).normalized()

            hit, face_index = cast_ray(origin, main_direction)
            if hit and face_index < len(bm.faces):
                face = bm.faces[face_index]
                add_face_and_connected(face, vertices_to_keep)

            for _ in range(secondary_rays):
                azimuth_offset = np.random.uniform(-np.pi / 9, np.pi / 9)
                elevation_offset = np.random.uniform(-np.pi / 9, np.pi / 9)

                offset_direction = main_direction.copy()
                offset_direction.rotate(mathutils.Euler((elevation_offset, 0, azimuth_offset)))

                hit, face_index = cast_ray(origin, offset_direction)
                if hit and face_index < len(bm.faces):
                    face = bm.faces[face_index]
                    add_face_and_connected(face, vertices_to_keep)

    bridged = bridge_nearby_islands(
        bm, vertices_to_keep, min_bridge_island_faces=min_bridge_island_faces, max_bridge_hops=max_bridge_hops
    )
    print(f"Bridged {bridged} nearby island pairs via original-mesh shortest path")

    for vert in bm.verts:
        vert.select = vert.index in vertices_to_keep

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="INVERT")

    bm.verts.ensure_lookup_table()
    verts_to_remove = [v for v in bm.verts if not v.select]
    bmesh.ops.delete(bm, geom=verts_to_remove, context="VERTS")

    bpy.ops.object.mode_set(mode="OBJECT")
    bm.to_mesh(mesh)
    mesh.update()

    bm.free()

    print(f"Vertices kept (pre-island-filter): {len(vertices_to_keep)}")
    print(f"Total vertices after ray-cast cleaning: {len(obj.data.vertices)}")

    removed_islands = filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces)")
    print(f"Total vertices after island filtering: {len(obj.data.vertices)}")


def find_largest_component(obj):
    bpy.ops.object.mode_set(mode="EDIT")

    mesh = bmesh.from_edit_mesh(obj.data)

    for v in mesh.verts:
        v.select = False
    mesh.verts.ensure_lookup_table()

    unvisited = set(mesh.verts)
    largest_component = set()

    while unvisited:
        current = unvisited.pop()
        component = set([current])
        to_visit = set([current])

        while to_visit:
            current = to_visit.pop()
            for edge in current.link_edges:
                neighbor = edge.other_vert(current)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    component.add(neighbor)
                    to_visit.add(neighbor)

        if len(component) > len(largest_component):
            largest_component = component

    for v in largest_component:
        v.select = True

    bmesh.update_edit_mesh(obj.data)

    bpy.ops.mesh.select_all(action="INVERT")
    bpy.ops.mesh.delete(type="VERT")

    bpy.ops.object.mode_set(mode="OBJECT")


def filter_small_components(obj, min_faces=4):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    unvisited = set(bm.faces)
    removed_components = 0
    verts_to_remove = set()

    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        to_visit = [seed]

        while to_visit:
            current = to_visit.pop()
            for edge in current.edges:
                for linked_face in edge.link_faces:
                    if linked_face in unvisited:
                        unvisited.remove(linked_face)
                        component.add(linked_face)
                        to_visit.append(linked_face)

        if len(component) < min_faces:
            removed_components += 1
            for face in component:
                verts_to_remove.update(face.verts)

    if verts_to_remove:
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=list(verts_to_remove), context="VERTS")
        bmesh.update_edit_mesh(obj.data)

    bpy.ops.object.mode_set(mode="OBJECT")
    return removed_components


def export_mesh_to_obj(obj, filepath):
    if obj.type != "MESH":
        raise TypeError("The selected object is not a mesh.")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh = obj.data
    vertices = []
    faces = []

    for vert in mesh.vertices:
        vertices.append(vert.co)

    for poly in mesh.polygons:
        if len(poly.vertices) == 3:
            faces.append(poly.vertices)
        else:
            raise ValueError(f"Face with vertices {poly.vertices} is not a triangle and will be skipped.")

    with open(filepath, "w") as file:
        for vert in vertices:
            file.write(f"v {vert.x} {vert.y} {vert.z}\n")

        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    return filepath


def count_holes(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]

    hole_count = 0
    visited_edges = set()

    for start_edge in boundary_edges:
        if start_edge not in visited_edges:
            current_edge = start_edge
            is_hole = True
            loop_edges = []

            while current_edge not in visited_edges:
                visited_edges.add(current_edge)
                loop_edges.append(current_edge)

                next_vert = (
                    current_edge.verts[1]
                    if current_edge.verts[0] in current_edge.link_faces[0].verts
                    else current_edge.verts[0]
                )
                next_edges = [e for e in next_vert.link_edges if e in boundary_edges and e != current_edge]

                if not next_edges:
                    is_hole = False
                    break

                current_edge = next_edges[0]

                if current_edge == start_edge:
                    break

            if is_hole:
                hole_count += 1

    bpy.ops.object.mode_set(mode="OBJECT")

    return hole_count


def mesh_integrity_report(obj):
    """
    Broader mesh integrity check than count_holes(), specifically because count_holes() has a
    real blind spot: it only ever looks at edges with EXACTLY 1 linked face (a simple open
    boundary). It has zero visibility into edges with 3+ linked faces - which is exactly what
    overlapping/double-skin geometry looks like (two independently-reconstructed surface patches
    covering roughly the same physical location, both real faces, neither one a simple "open"
    edge). That specific gap is why "0 holes" was reported on a mesh that still looked visibly
    broken in the Blender viewport (tile-boundary marching-cubes ambiguity, before voxel remesh
    cleanup was added) - count_holes was never wrong about what it measures, it just wasn't
    measuring the right thing on its own.

    Args:
        obj (bpy.types.Object): The Blender object to analyze.

    Returns:
        dict: {
            "vertex_count", "face_count",
            "boundary_edges": edges with exactly 1 linked face (simple open boundary, what
                              count_holes walks),
            "non_manifold_edges": edges with 3+ linked faces (overlap/double-skin - the blind
                                  spot above),
            "non_manifold_verts": vertices that aren't a single continuous face-fan (bowtie
                                  configurations etc.), via Blender's own BMVert.is_manifold,
            "simple_holes": count_holes()'s own count, kept for continuity with prior logs,
            "is_watertight": True only if boundary_edges == 0 AND non_manifold_edges == 0 AND
                             non_manifold_verts == 0 - a much stricter, more honest definition
                             than "count_holes() returned 0",
            "signed_volume": bmesh's own signed volume calculation. Only strictly meaningful when
                             is_watertight is True, but reported regardless as an extra sanity
                             signal - a nonsensical volume even on a "watertight" mesh points at
                             inverted normals or a self-intersecting shell rather than a genuine
                             simple closed surface.
        }
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    boundary_edges = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    non_manifold_edges = sum(1 for e in bm.edges if len(e.link_faces) >= 3)
    non_manifold_verts = sum(1 for v in bm.verts if not v.is_manifold)

    signed_volume = None
    try:
        signed_volume = bm.calc_volume(signed=True)
    except Exception as e:
        print(f"mesh_integrity_report: volume calc failed: {e}")

    bpy.ops.object.mode_set(mode="OBJECT")

    simple_holes = count_holes(obj)

    is_watertight = (boundary_edges == 0 and non_manifold_edges == 0 and non_manifold_verts == 0)

    report = {
        "vertex_count": len(obj.data.vertices),
        "face_count": len(obj.data.polygons),
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "non_manifold_verts": non_manifold_verts,
        "simple_holes": simple_holes,
        "is_watertight": is_watertight,
        "signed_volume": round(signed_volume, 2) if signed_volume is not None else None,
    }
    print(f"mesh_integrity_report: {report}")
    return report


def count_boundary_edges(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    n = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    return n


def describe_hole_locations(obj, top_n=15):
    """
    Walks each remaining boundary loop (same traversal as count_holes) and reports its edge count
    and centroid distance from the mesh's own centroid, sorted farthest-from-centroid first.

    This exists to answer a specific question cheaply, from the log, without re-inspecting the
    mesh visually each run: are the remaining holes concentrated at the extremities (tips of legs/
    mandibles - large distance from centroid) or scattered across the body (small distance)? A
    "far from centroid" fingerprint supports the tip-taper/self-occlusion hypothesis; a "close to
    centroid, on the body" fingerprint would point at a different cause (e.g. a bridging gap) and
    mean tuning voxel_pitch/keep_rings further isn't the right next step.

    Args:
        obj (bpy.types.Object): The Blender object to analyze.
        top_n (int): Number of farthest-from-centroid holes to print.

    Returns:
        list[dict]: Each hole as {"n_edges", "centroid", "dist_from_mesh_centroid"}, sorted
                    farthest-first.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    mesh_centroid = Vector((0.0, 0.0, 0.0))
    for v in bm.verts:
        mesh_centroid += v.co
    if len(bm.verts) > 0:
        mesh_centroid /= len(bm.verts)

    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    boundary_edge_set = set(boundary_edges)

    holes = []
    visited_edges = set()
    for start_edge in boundary_edges:
        if start_edge in visited_edges:
            continue
        current_edge = start_edge
        is_hole = True
        loop_edges = []
        while current_edge not in visited_edges:
            visited_edges.add(current_edge)
            loop_edges.append(current_edge)
            next_vert = (
                current_edge.verts[1]
                if current_edge.verts[0] in current_edge.link_faces[0].verts
                else current_edge.verts[0]
            )
            next_edges = [e for e in next_vert.link_edges if e in boundary_edge_set and e != current_edge]
            if not next_edges:
                is_hole = False
                break
            current_edge = next_edges[0]
            if current_edge == start_edge:
                break
        if not is_hole:
            continue
        loop_verts = set()
        for e in loop_edges:
            loop_verts.update(e.verts)
        loop_centroid = Vector((0.0, 0.0, 0.0))
        for v in loop_verts:
            loop_centroid += v.co
        loop_centroid /= len(loop_verts)
        holes.append({
            "n_edges": len(loop_edges),
            "centroid": tuple(loop_centroid),
            "dist_from_mesh_centroid": (loop_centroid - mesh_centroid).length,
        })

    bpy.ops.object.mode_set(mode="OBJECT")

    holes.sort(key=lambda h: h["dist_from_mesh_centroid"], reverse=True)
    print(f"describe_hole_locations: {len(holes)} holes total, mesh_centroid={tuple(mesh_centroid)}")
    print(f"  top {min(top_n, len(holes))} farthest-from-centroid (most likely tip/extremity holes):")
    for h in holes[:top_n]:
        print(f"    n_edges={h['n_edges']:>3}  dist_from_centroid={h['dist_from_mesh_centroid']:>7.2f}  "
              f"centroid={tuple(round(c, 1) for c in h['centroid'])}")
    if len(holes) > top_n:
        remaining_dists = [h["dist_from_mesh_centroid"] for h in holes[top_n:]]
        print(f"  remaining {len(holes) - top_n} holes: dist_from_centroid range "
              f"{min(remaining_dists):.2f}-{max(remaining_dists):.2f}")

    return holes


def cap_remaining_holes(obj):
    """
    Last-resort closer: for every remaining boundary loop (same walk as count_holes /
    describe_hole_locations), adds a new vertex at the loop's centroid and fan-triangulates it to
    every edge in the loop. This ALWAYS closes a simple loop, regardless of how non-planar or
    irregular it is - unlike bmesh.ops.holes_fill, which can silently fail to close loops with
    awkward geometry.

    This is an explicit tradeoff, not a fix: the cap is a flat/blunt fan, not a reconstruction of
    the true tip shape. Use this where watertightness matters more than anatomical fidelity at the
    capped spot - e.g. when describe_hole_locations has already shown the remaining holes are
    concentrated at self-occluded extremities (leg/mandible tips) where the raw scan itself likely
    has no real data to reconstruct from, so no amount of upstream resolution tuning will produce
    a better answer than this cap would.

    Args:
        obj (bpy.types.Object): The Blender object to cap in place.

    Returns:
        tuple[int, int]: (holes capped, holes skipped - non-simple loops or degenerate faces).
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    boundary_edge_set = set(boundary_edges)
    visited_edges = set()
    capped = 0
    skipped = 0

    for start_edge in boundary_edges:
        if start_edge in visited_edges:
            continue

        current_edge = start_edge
        is_hole = True
        ring_verts = []
        first_vert_added = False

        while current_edge not in visited_edges:
            visited_edges.add(current_edge)
            pivot = (
                current_edge.verts[0]
                if current_edge.verts[0] in current_edge.link_faces[0].verts
                else current_edge.verts[1]
            )
            next_vert = current_edge.verts[1] if pivot == current_edge.verts[0] else current_edge.verts[0]
            if not first_vert_added:
                ring_verts.append(pivot)
                first_vert_added = True
            ring_verts.append(next_vert)

            next_edges = [e for e in next_vert.link_edges if e in boundary_edge_set and e != current_edge]
            if not next_edges:
                is_hole = False
                break
            current_edge = next_edges[0]
            if current_edge == start_edge:
                break

        if not is_hole:
            skipped += 1
            continue
        if ring_verts and ring_verts[0] == ring_verts[-1]:
            ring_verts = ring_verts[:-1]
        if len(ring_verts) < 3:
            skipped += 1
            continue

        centroid = Vector((0.0, 0.0, 0.0))
        for v in ring_verts:
            centroid += v.co
        centroid /= len(ring_verts)
        center_vert = bm.verts.new(centroid)

        ok = True
        n = len(ring_verts)
        for i in range(n):
            v1 = ring_verts[i]
            v2 = ring_verts[(i + 1) % n]
            try:
                f = bm.faces.new([center_vert, v1, v2])
            except ValueError:
                ok = False  # duplicate face - leave this bit of the loop open
                continue
            if f.calc_area() < 1e-9:
                # bm.faces.new() only raises on an exact duplicate vertex set - a zero-area
                # sliver from near-collinear points (centroid, v1, v2 almost in a line) succeeds
                # silently and leaves a face with an undefined/zero-length normal, which crashes
                # calculate_mesh_smoothness downstream. Remove it explicitly rather than let it
                # through; this bit of the loop is left open, same as the duplicate-face case.
                # MUST be FACES_ONLY, not FACES: center_vert is shared across every triangle in
                # this fan and starts with zero faces of its own (v1/v2 are original boundary
                # verts that always keep >=1 real face elsewhere, but center_vert has none until
                # a face here succeeds). context="FACES" deletes now-unused verts along with the
                # face, so if THIS is center_vert's only face so far, it gets deleted too - then
                # the next iteration's bm.faces.new([center_vert, ...]) crashes with "0 BMVert has
                # been removed". FACES_ONLY deletes just the face and never touches verts.
                bmesh.ops.delete(bm, geom=[f], context="FACES_ONLY")
                ok = False
        if ok:
            capped += 1
        else:
            skipped += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)  # cheap safety net for fan-cap winding direction
    bpy.ops.object.mode_set(mode="OBJECT")

    print(f"cap_remaining_holes: capped {capped} holes with flat centroid-fan patches "
          f"(these are NOT reconstructed anatomy - just guaranteed closure), skipped {skipped} "
          f"(non-simple loops or degenerate geometry)")
    return capped, skipped


def remove_orphan_vertices(obj):
    """
    Deletes vertices with zero linked faces. Confirmed empirically (not assumed): a
    zero-face vertex has BMVert.is_manifold == False in Blender, so these silently inflate
    mesh_integrity_report's non_manifold_verts count even though they carry no geometry at
    all - on Acanthomyrmex_cf.ferox's winding_number run, 21,206 of 52,444 exported vertices
    (40%) turned out to be zero-face orphans, accounting for 91% of that run's reported
    non_manifold_verts=23,285. Most originate from cap_remaining_holes's FACES_ONLY fix
    (see that function's comment) intentionally no longer deleting a degenerate fan's
    center_vert - correct for avoiding the "0 BMVert has been removed" crash, but it leaves
    the orphan behind if every triangle in that fan failed. This is a straightforward
    end-of-pipeline sweep, not a replacement for that fix.

    Args:
        obj (bpy.types.Object): The Blender object to clean in place.

    Returns:
        int: The number of orphan vertices removed.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    orphans = [v for v in bm.verts if not v.link_faces]
    if orphans:
        bmesh.ops.delete(bm, geom=orphans, context="VERTS")
        bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"remove_orphan_vertices: removed {len(orphans)} zero-face orphan vertices")
    return len(orphans)


def calculate_face_size_cov(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    face_areas = [f.calc_area() for f in bm.faces]

    bpy.ops.object.mode_set(mode="OBJECT")

    return np.round(np.std(face_areas) / np.mean(face_areas), 3)


def calculate_mesh_smoothness(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    total_angle = 0
    total_comparisons = 0
    skipped_degenerate = 0

    for face in bm.faces:
        if face.normal.length < 1e-9:
            continue  # degenerate (near-zero-area) face - no well-defined normal, skip rather than crash
        for edge in face.edges:
            adjacent_face = [f for f in edge.link_faces if f != face]
            if adjacent_face:
                if adjacent_face[0].normal.length < 1e-9:
                    skipped_degenerate += 1
                    continue
                angle = face.normal.angle(adjacent_face[0].normal)
                total_angle += math.degrees(angle)
                total_comparisons += 1

    if skipped_degenerate:
        print(f"calculate_mesh_smoothness: skipped {skipped_degenerate} face-pairs adjacent to "
              f"degenerate (zero-area) faces")

    bpy.ops.object.mode_set(mode="OBJECT")

    if total_comparisons > 0:
        average_angle = total_angle / total_comparisons
        return np.round(average_angle, 3)
    else:
        return 0.0


def _find_exact_duplicate_vertex_groups(coords):
    if len(coords) == 0:
        return []

    _, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    return [np.flatnonzero(inverse == group_id) for group_id in np.nonzero(counts > 1)[0]]


def inspect_duplicate_vertex_groups(obj, max_groups=10, max_verts_per_group=2, dup_vert_decimals=7):
    """
    Print a topology-focused summary for a few coordinate-duplicate vertex groups.
    This is meant to answer the core question behind the cleanup regression:
    are the 'duplicate vertices' really the same logical vertex, or are they coincident
    vertices that belong to different face fans / disconnected shells?
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    coords = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    rounded = np.round(coords, dup_vert_decimals)
    _, inverse, counts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    groups = [np.flatnonzero(inverse == group_id) for group_id in np.nonzero(counts > 1)[0]]

    print(f"inspect_duplicate_vertex_groups: {len(groups)} duplicate groups found")
    for gi, group in enumerate(groups[:max_groups]):
        if len(group) == 0:
            continue

        print(f"--- duplicate-group {gi} size={len(group)} ---")
        verts = [bm.verts[int(idx)] for idx in group[:max_verts_per_group]]
        for vert in verts:
            face_ids = [f.index for f in vert.link_faces]
            edge_ids = [e.index for e in vert.link_edges]
            print(f"vertex {vert.index} co={tuple(round(float(c), 8) for c in vert.co)}")
            print(f"  n_faces={len(face_ids)} n_edges={len(edge_ids)}")
            print(f"  faces={face_ids[:20]}")
            print(f"  edges={edge_ids[:20]}")
            print(f"  boundary={any(len(e.link_faces) == 1 for e in vert.link_edges)}")

        if len(verts) >= 2:
            v0, v1 = verts[0], verts[1]
            faces0 = {f.index for f in v0.link_faces}
            faces1 = {f.index for f in v1.link_faces}
            edges0 = {e.index for e in v0.link_edges}
            edges1 = {e.index for e in v1.link_edges}
            print(f"  same_face_fan={faces0 == faces1}")
            print(f"  same_edge_fan={edges0 == edges1}")
            print(f"  face_overlap={sorted(faces0 & faces1)}")
            print(f"  edge_overlap={sorted(edges0 & edges1)}")
            print(f"  face_only_in_v0={sorted(faces0 - faces1)}")
            print(f"  face_only_in_v1={sorted(faces1 - faces0)}")

    bm.free()
    return groups


def log_mesh_state(obj, label):
    print(f"STATE[{label}]")
    report_mesh_degeneracy(obj)
    print(mesh_integrity_report(obj))


def report_mesh_degeneracy(obj, zero_len_eps=1e-8, zero_area_eps=1e-9, dup_vert_decimals=7):
    """
    Checks for GEOMETRIC degeneracy - zero-length edges, zero-area faces, duplicate vertices,
    duplicate faces - which mesh_integrity_report does NOT check at all. Manifoldness
    (boundary_edges/non_manifold_edges/is_watertight) is a purely COMBINATORIAL property (face
    count per edge, vertex fan connectivity); a mesh can be perfectly manifold and watertight by
    that definition while still containing zero-length edges or duplicate vertices sitting on
    top of each other - those are a separate, geometric kind of brokenness the combinatorial
    checks cannot see. Written to answer a specific question raised after decimate_mesh damage
    was traced to pristine-mesh vertices with min_edge_len==0 even though the SAME pristine mesh
    had just been reported is_watertight=True: is that pre-existing degeneracy from
    close_holes_via_winding_number/voxel-remesh, or an artifact of the bad-vertex matching?
    Call this directly on the pristine mesh, independent of any decimation attempt.

    Args:
        obj (bpy.types.Object): The Blender object to inspect (read-only, not modified).
        zero_len_eps (float): Edges shorter than this count as zero-length.
        zero_area_eps (float): Faces smaller than this count as zero-area (matches
                               calculate_mesh_smoothness's own degenerate-face threshold).
        dup_vert_decimals (int): Rounding precision for duplicate-vertex position matching.

    Returns:
        dict: {"n_verts", "n_edges", "n_faces", "n_zero_length_edges", "n_zero_area_faces",
              "n_duplicate_verts", "n_duplicate_faces", "edge_length_percentiles"}.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    edge_lengths = np.array([e.calc_length() for e in bm.edges], dtype=np.float64)
    n_zero_length_edges = int(np.sum(edge_lengths < zero_len_eps))

    face_areas = np.array([f.calc_area() for f in bm.faces], dtype=np.float64)
    n_zero_area_faces = int(np.sum(face_areas < zero_area_eps))

    coords = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    rounded = np.round(coords, dup_vert_decimals)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    n_duplicate_verts = int(np.sum(counts[counts > 1] - 1))  # extra copies beyond the first

    # Faces here are NOT guaranteed to be triangles (holes_fill with sides=0 and dissolve_limit
    # both can produce n-gons), so vertex-index tuples can vary in length - use a Counter, not
    # a fixed-shape numpy array (np.stack on variable-length tuples would raise).
    face_vert_counts = Counter(tuple(sorted(v.index for v in f.verts)) for f in bm.faces)
    n_duplicate_faces = sum(c - 1 for c in face_vert_counts.values() if c > 1)

    percentiles = {}
    if len(edge_lengths):
        for p in (0, 0.1, 1, 5, 50):
            percentiles[p] = float(np.percentile(edge_lengths, p))

    bm.free()

    report = {
        "n_verts": len(coords),
        "n_edges": len(edge_lengths),
        "n_faces": len(face_areas),
        "n_zero_length_edges": n_zero_length_edges,
        "n_zero_area_faces": n_zero_area_faces,
        "n_duplicate_verts": n_duplicate_verts,
        "n_duplicate_faces": n_duplicate_faces,
        "edge_length_percentiles": percentiles,
    }
    print(f"report_mesh_degeneracy: {report}")
    return report


def clean_mesh_degeneracy(obj, dist):
    """
    Removes GEOMETRIC degeneracy (duplicate/near-duplicate vertices, zero-length edges,
    zero-area faces) via Blender's own purpose-built operators, in the standard order:
    remove_doubles first - merges duplicate/near-duplicate vertices, the root cause of most
    zero-length edges - then dissolve_degenerate, which cleans up whatever zero-length edges/
    zero-area faces remain after welding (e.g. a sliver left when two of a triangle's three
    vertices get merged but the third stays distinct). Confirmed necessary empirically, not
    assumed: a mesh mesh_integrity_report reported is_watertight=True (boundary_edges=0,
    non_manifold_edges=0) was independently found (report_mesh_degeneracy) to contain 10,311
    zero-length edges and 7,754 duplicate vertices at the same time - manifoldness is a purely
    combinatorial property and does not imply geometric cleanliness.

    Args:
        obj (bpy.types.Object): The Blender object to clean in place.
        dist (float): Distance threshold for both operators. Tie this to the mesh's own
                      characteristic scale (e.g. voxel_pitch * 0.01, not an arbitrary constant),
                      and check it against the actual edge-length percentile distribution first -
                      confirmed empirically that voxel_pitch * 0.05 sat ABOVE the p5 edge-length
                      percentile, meaning it merged real small-but-legitimate geometry, not just
                      the degenerate cluster, and blew up boundary_edges from 0 to 45,466 as a
                      side effect. Aim below the percentile where degeneracy actually clusters.

    Returns:
        tuple[dict, dict]: (degeneracy report before cleanup, degeneracy report after cleanup).
    """
    print(f"clean_mesh_degeneracy: dist={dist:.4g}")
    print("clean_mesh_degeneracy: before:")
    before = report_mesh_degeneracy(obj)
    integrity_before = mesh_integrity_report(obj)

    # Split into two separately-measured steps (was: run both, then check once) - remove_doubles
    # (pure vertex merging) and dissolve_degenerate (can dissolve/restructure local topology, not
    # just merge points) are different kinds of operations; if remove_doubles alone already
    # clears most of the degeneracy without damaging boundary_edges, that's a materially safer
    # fix than needing dissolve_degenerate's more invasive restructuring too.
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")

    print("clean_mesh_degeneracy: after remove_doubles ALONE (before dissolve_degenerate):")
    report_mesh_degeneracy(obj)
    integrity_after_remove_doubles = mesh_integrity_report(obj)
    print(f"  boundary_edges={integrity_after_remove_doubles['boundary_edges']}, "
          f"non_manifold_edges={integrity_after_remove_doubles['non_manifold_edges']} "
          f"(was boundary_edges={integrity_before['boundary_edges']}, "
          f"non_manifold_edges={integrity_before['non_manifold_edges']} before any cleanup)")

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.ops.dissolve_degenerate(bm, dist=dist, edges=bm.edges)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")

    print("clean_mesh_degeneracy: after remove_doubles + dissolve_degenerate:")
    after = report_mesh_degeneracy(obj)
    integrity_after = mesh_integrity_report(obj)
    print(f"  boundary_edges={integrity_after['boundary_edges']}, "
          f"non_manifold_edges={integrity_after['non_manifold_edges']}")

    if after["n_zero_length_edges"] or after["n_zero_area_faces"] or after["n_duplicate_verts"]:
        print(
            f"clean_mesh_degeneracy: WARNING - mesh still geometrically degenerate before "
            f"decimation (zero_length_edges={after['n_zero_length_edges']}, "
            f"zero_area_faces={after['n_zero_area_faces']}, "
            f"duplicate_verts={after['n_duplicate_verts']}). dist={dist:.4g} may be too small, "
            f"or some degeneracy has a different root cause than remove_doubles/"
            f"dissolve_degenerate can fix."
        )
    else:
        print("clean_mesh_degeneracy: mesh is fully clean (0 zero-length edges, "
              "0 zero-area faces, 0 duplicate verts).")

    return before, after


def clean_mesh_degeneracy_local(obj, dist, expand_rings=1, region_margin_factor=2.0,
                                 zero_len_eps=1e-8, zero_area_eps=1e-9, dup_vert_decimals=7):
    """
    Conservative cleanup for voxel-remesh artifacts: only exact duplicate vertices are merged,
    and zero-area faces are removed. This avoids the topology-restructuring behavior of
    dissolve_degenerate, which was shown to create holes and reshape thin structures even when
    the underlying geometry was already mostly sane.

    Args:
        obj (bpy.types.Object): The Blender object to clean in place.
        dist (float): Kept for API compatibility; exact-duplicate cleanup ignores this threshold.
        expand_rings (int): Kept for API compatibility; the conservative path does not expand a
                            region around flagged vertices.
        region_margin_factor (float): Kept for API compatibility; unused in this conservative path.
        zero_len_eps, zero_area_eps, dup_vert_decimals: match report_mesh_degeneracy's own
                                                         detection thresholds.

    Returns:
        dict: {"n_exact_duplicates", "n_zero_area_faces_removed", "before", "after"}.
    """
    print(f"clean_mesh_degeneracy_local: conservative cleanup (dist={dist:.4g}, expand_rings={expand_rings})")
    before = report_mesh_degeneracy(obj)
    integrity_before = mesh_integrity_report(obj)
    print(f"  boundary_edges={integrity_before['boundary_edges']}, "
          f"non_manifold_edges={integrity_before['non_manifold_edges']} before conservative cleanup")

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # The topology probe showed that many coordinate-duplicate vertices are not safe to merge:
    # their face/edge fans differ, so merging them by position alone breaks the local structure
    # and creates boundary holes. We therefore skip duplicate-vertex merging in this cleanup path
    # and only remove exact zero-area faces.
    coords_all = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    duplicate_groups = _find_exact_duplicate_vertex_groups(coords_all)
    n_exact_duplicates = sum(len(group) - 1 for group in duplicate_groups)
    print(f"  {n_exact_duplicates} exact duplicate vertices found")
    if n_exact_duplicates > 0:
        print("  skipping duplicate-vertex merge: topology probe showed coincident vertices often have different face/edge fans")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    faces_to_remove = [f for f in bm.faces if f.calc_area() < zero_area_eps]
    n_zero_area_faces_removed = len(faces_to_remove)
    print(f"  {n_zero_area_faces_removed} zero-area faces found")

    if faces_to_remove:
        bmesh.ops.delete(bm, geom=faces_to_remove, context="FACES")
        bmesh.update_edit_mesh(obj.data)

    bpy.ops.object.mode_set(mode="OBJECT")

    after = report_mesh_degeneracy(obj)
    integrity_after = mesh_integrity_report(obj)
    print(f"  boundary_edges={integrity_after['boundary_edges']}, "
          f"non_manifold_edges={integrity_after['non_manifold_edges']} after conservative cleanup")

    return {"n_exact_duplicates": n_exact_duplicates,
            "n_zero_area_faces_removed": n_zero_area_faces_removed,
            "before": before,
            "after": after}


def report_bad_edges(obj, limit=20):
    """
    Lists edges with a face count other than 2 (boundary edges: 1 face; non-manifold edges: 3+)
    on obj's CURRENT mesh, with their vertex coordinates - for inspecting WHERE decimation (or
    any other step) introduced topology damage, before deciding how to fix it. Call this while
    the damaged mesh is still live (e.g. right after detecting damage in decimate_mesh, before
    any revert), not after restoring a backup - the whole point is to see the actual damaged
    state.

    Args:
        obj (bpy.types.Object): The Blender object to inspect (read-only, not modified).
        limit (int): Max number of bad edges to print in detail.

    Returns:
        tuple[int, list[tuple[float, float, float]]]: Total bad-edge count, and the flattened
        list of both endpoint coordinates of every bad edge (duplicates included - a vertex
        touching multiple bad edges appears multiple times, which is informative on its own).
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    bad = []
    bad_vert_coords = []
    for e in bm.edges:
        if len(e.link_faces) != 2:
            coords = tuple(v.co[:] for v in e.verts)
            bad.append((e.index, len(e.link_faces), coords))
            bad_vert_coords.extend(coords)

    print(f"report_bad_edges: {len(bad)} bad edges (face count != 2)")
    for item in bad[:limit]:
        print(f"  edge={item[0]} n_faces={item[1]} verts={item[2]}")
    if len(bad) > limit:
        print(f"  ... and {len(bad) - limit} more")

    bm.free()
    return len(bad), bad_vert_coords


def check_bad_verts_edge_length_percentile(pristine_mesh_data, bad_vert_coords, percentile=5):
    """
    Validates (or refutes) the "thin/tip geometry, not self-approach fusion" hypothesis for
    decimation damage: for every vertex in the PRE-decimation mesh, computes its minimum
    incident edge length, then checks where each bad-edge vertex (from the DAMAGED mesh, matched
    to its nearest pristine vertex by 3D position - decimation moves/merges vertices, so indices
    don't correspond between the two meshes) falls in that mesh-wide distribution. If nearly all
    bad vertices match to pristine vertices already in the bottom `percentile`% of edge lengths,
    that's direct, specific evidence the damage concentrates on genuinely thin/tip-like local
    geometry (Decimate collapsing a sharp convergence point past survivability), not two distant
    surfaces fusing (which would show ordinary, not unusually-short, local edge lengths at the
    fusion site itself).

    Args:
        pristine_mesh_data (bpy.types.Mesh): The mesh datablock from BEFORE the damaging
                                             decimation attempt (e.g. decimate_mesh's
                                             pristine_backup, read while still untouched).
        bad_vert_coords (list[tuple[float, float, float]]): Vertex coordinates from the damaged
                                                             mesh's bad edges (report_bad_edges's
                                                             second return value).
        percentile (float): Percentile threshold to check bad vertices against.

    Returns:
        dict: {"threshold", "n_below_threshold", "n_total", "fraction_below"}.
    """
    from scipy.spatial import cKDTree

    bm = bmesh.new()
    bm.from_mesh(pristine_mesh_data)
    bm.verts.ensure_lookup_table()

    n = len(bm.verts)
    all_min_edge_len = np.full(n, np.inf, dtype=np.float64)
    all_coords = np.empty((n, 3), dtype=np.float64)
    for v in bm.verts:
        edge_lens = [e.calc_length() for e in v.link_edges]
        if edge_lens:
            all_min_edge_len[v.index] = min(edge_lens)
        all_coords[v.index] = v.co[:]
    bm.free()

    finite_mask = np.isfinite(all_min_edge_len)
    p_threshold = float(np.percentile(all_min_edge_len[finite_mask], percentile))

    tree = cKDTree(all_coords)
    bad_coords_arr = np.array(bad_vert_coords, dtype=np.float64)
    dists, idxs = tree.query(bad_coords_arr, k=1)
    matched_min_edge_len = all_min_edge_len[idxs]

    n_below = int(np.sum(matched_min_edge_len <= p_threshold))
    n_total = len(bad_vert_coords)

    print(f"check_bad_verts_edge_length_percentile: p{percentile}_min_edge_length_threshold="
          f"{p_threshold:.4g} (mesh-wide, n={n:,} pristine verts)")
    print(f"  {n_below}/{n_total} bad vertices matched to a pristine vertex at/below that "
          f"threshold ({n_below / n_total * 100:.1f}%)" if n_total else "  no bad vertices")
    for coord, dist, min_len in list(zip(bad_coords_arr, dists, matched_min_edge_len))[:15]:
        below = "YES" if min_len <= p_threshold else "no"
        print(f"    bad_vert={tuple(coord.round(3))} nearest_pristine_dist={dist:.4g} "
              f"pristine_min_edge_len={min_len:.4g} in_bottom_{percentile}pct={below}")

    return {
        "threshold": p_threshold,
        "n_below_threshold": n_below,
        "n_total": n_total,
        "fraction_below": n_below / n_total if n_total else float("nan"),
    }


def decimate_mesh(obj, max_vertices, min_ratio_step=0.99, max_retries_per_step=6):
    """
    Args:
        obj (bpy.types.Object): The Blender object to decimate in place.
        max_vertices (int): Target maximum vertex count.
        min_ratio_step (float): Stop retrying a gentler ratio once it's this close to 1.0 (i.e.
                                less than (1-min_ratio_step)*100% reduction) - diminishing
                                returns past this point, and a step this gentle still damaging
                                topology means this specimen genuinely can't be decimated further
                                without hitting a real geometric problem, not a step-size problem.
        max_retries_per_step (int): Cap on gentler-ratio retries per outer iteration, in case
                                    min_ratio_step is never reached (e.g. base_ratio already close
                                    to 1.0).

    Returns:
        int: The final vertex count.
    """
    print("Applying mesh decimation...")

    current_vertices = len(obj.data.vertices)
    iteration_count = 0
    last_vertex_count = current_vertices

    while current_vertices > max_vertices * 1.05:
        # Pristine snapshot of this outer iteration's starting mesh - never mutated directly;
        # each retry works on its own throwaway copy of it, so a damaging attempt at one ratio
        # doesn't corrupt the starting point for the next, gentler retry. The retry loop below
        # always reassigns obj.data to a fresh copy of pristine_backup, so free the mesh that
        # was active on entry now - otherwise it's immediately orphaned (nothing left pointing
        # to it) and leaked for the rest of this Blender session, once per outer iteration.
        outer_input_mesh = obj.data
        pristine_backup = outer_input_mesh.copy()
        obj.data = pristine_backup
        bpy.data.meshes.remove(outer_input_mesh)
        backup_vertex_count = current_vertices

        if current_vertices / max_vertices > 2:
            ratio = 0.8
        else:
            ratio = (max_vertices * 1.05) / current_vertices

        # RETRY WITH A GENTLER RATIO instead of giving up on the whole outer loop after one
        # damaging attempt (was: one damaging step aborted decimation entirely, silently leaving
        # the mesh at whatever vertex count it happened to be at - e.g. one real run left a
        # mesh at 963,681 vertices against a 50,000 target because the very first, most
        # aggressive step (ratio=0.8) hit a topology-damaging collapse and the loop just quit).
        # A single bad ratio doesn't mean this mesh can't be decimated at all - it means THIS
        # step size collapsed an edge across a genuine geometric problem (the same self-approach-
        # gap mechanism diagnosed for Weld/voxel-remesh: COLLAPSE can merge two vertices that are
        # close in 3D but topologically distant, exactly like Weld could). Retrying with a
        # smaller step (halving the remaining distance to ratio=1.0 each time) gives decimation
        # a real chance to make progress around that problem instead of surrendering completely.
        step_succeeded = False
        retry = 0
        while True:
            attempt_mesh = pristine_backup.copy()
            obj.data = attempt_mesh

            modifier = obj.modifiers.new(name="Decimate", type="DECIMATE")
            modifier.decimate_type = "COLLAPSE"
            modifier.use_symmetry = False
            modifier.use_collapse_triangulate = True
            modifier.use_dissolve_boundaries = False
            modifier.ratio = ratio

            bpy.context.view_layer.objects.active = obj
            try:
                bpy.ops.object.modifier_apply(modifier="Decimate")
            except RuntimeError as e:
                if "Modifiers cannot be applied to multi-user data" in str(e):
                    print("Making mesh data single-user and retrying...")
                    bpy.ops.object.make_single_user(
                        object=True, obdata=True, material=False, animation=False
                    )
                    bpy.ops.object.modifier_apply(modifier="Decimate")
                else:
                    raise

            current_vertices = len(obj.data.vertices)
            integrity = mesh_integrity_report(obj)
            print(
                f"Decimation attempt (ratio={ratio:.4f}, retry={retry}): {current_vertices} "
                f"vertices, boundary_edges={integrity['boundary_edges']}, "
                f"non_manifold_edges={integrity['non_manifold_edges']}"
            )

            if integrity["boundary_edges"] == 0 and integrity["non_manifold_edges"] == 0:
                step_succeeded = True
                break

            print(f"WARNING: decimation at ratio={ratio:.4f} damaged topology (retry {retry}).")
            if retry == 0:
                # Only on the first failure per outer iteration - empirically confirmed (see
                # decimate_mesh's docstring/comments above) that the damage is identical across
                # every gentler retry, so this is representative, not one sample of many.
                # pristine_backup is still the untouched pre-decimation mesh at this point (the
                # revert below hasn't happened yet), so this is the one moment both the damaged
                # and pristine states coexist - required to test whether bad vertices correspond
                # to genuinely short-edge (thin/tip) regions in the ORIGINAL geometry.
                _, bad_vert_coords = report_bad_edges(obj)
                if bad_vert_coords:
                    check_bad_verts_edge_length_percentile(pristine_backup, bad_vert_coords)
            bad_mesh = obj.data
            obj.data = pristine_backup
            bpy.data.meshes.remove(bad_mesh)

            retry += 1
            if ratio > min_ratio_step or retry > max_retries_per_step:
                print(
                    f"Decimation stopped: no non-damaging ratio found after {retry} retries "
                    f"(gentlest tried: {ratio:.4f}). Keeping mesh at {backup_vertex_count} "
                    f"vertices - further outer-loop iterations would very likely hit the same "
                    f"geometric problem, not a step-size problem."
                )
                current_vertices = backup_vertex_count
                break

            ratio = 1.0 - (1.0 - ratio) / 2.0
            print(f"Retrying decimation with gentler ratio={ratio:.4f}...")

        if step_succeeded:
            bpy.data.meshes.remove(pristine_backup)
        else:
            # obj.data is already pristine_backup (set by the last failed retry above) - it's
            # the live mesh now, do not remove it. No progress was possible this iteration.
            break

        iteration_count += 1

        if current_vertices >= last_vertex_count:
            print("Decimation stopped: no progress.")
            break

        if iteration_count > 10:
            print("Decimation stopped: iteration limit reached.")
            break

        last_vertex_count = current_vertices

    return current_vertices


def _clear_scene_mesh_objects():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            obj.select_set(True)
    bpy.ops.object.delete()


def _triangulated_verts_faces(obj):
    """
    Returns a fresh (verts, faces) numpy pair for obj's current mesh, triangulated. Works on a
    throwaway bmesh copy so it never mutates obj.data.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    verts = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    faces = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int64)
    bm.free()
    return verts, faces


def _min_self_approach_gap(verts, faces, hops=4, k=100):
    """
    Distance from every vertex to the nearest OTHER vertex that is not within `hops`
    mesh-edges of it, minimized over all vertices - i.e. how close two anatomically distinct
    but spatially close parts of the surface (e.g. a leg resting near the body) actually get.
    This is the real predictor of whether voxel-remesh (OpenVDB, via _voxel_remesh_cleanup)
    will incorrectly fuse them into one solid: it happens whenever this gap drops below the
    voxel_pitch used for reconstruction, regardless of the specimen's overall size.

    Empirically confirmed on 4 real specimens (see diagnostics/measure_self_approach_gap.py
    and diagnostics/PROBE_export_precleaning_mesh.py, run on the actual pre-hole-fill mesh
    both hole_fill_method paths share): the one specimen whose min_self_approach_gap fell
    BELOW its own edge-percentile-derived voxel_pitch (ratio 0.68, all others 1.2-2.3) was the
    same specimen with a catastrophic 37.4% Weld face-loss after voxel-remesh cleanup (all
    others: 1.5-13.0%). Bounding-box-relative proxies (voxel_pitch / bbox_min etc.) do NOT
    predict this - confirmed not to correlate on the same 4 specimens - because self-contact
    gaps are a local geometric fact with no necessary relationship to whole-specimen scale.

    Algorithm validated on a hand-built synthetic mesh with a known true minimum distance
    (a leg-like strip curling back near a flat plate, connected to it only via a distant
    topological joint): computed value matched the independently hand-calculated true nearest
    distance exactly. Filters zero-face orphan vertices first - confirmed on real pipeline
    output that these exist (e.g. one real specimen's export was 7.4% orphans) and, being
    unreachable via any mesh-hop, would otherwise trivially register as spuriously touching
    whatever real vertex happens to sit nearest them - caught exactly this failure mode on an
    early, sloppier version of the synthetic test mesh before this filter was added.

    Args:
        verts (np.ndarray): (N, 3) vertex positions.
        faces (np.ndarray): (M, 3) triangle vertex indices.
        hops (int): Mesh-edge hop radius to exclude as "locally adjacent, not a real
                    self-approach" - i.e. normal surface curvature, not two distinct parts.
        k (int): Nearest-neighbor candidates to check per vertex before giving up. Validated
                 empirically that k=40 already finds the same minimum as k=250 on all 4 real
                 specimens tested (the vertices that exhausted k=40 without a match were never
                 the ones holding the true minimum) - k=100 here is a safety margin, not a
                 value known to be necessary.

    Returns:
        float: The minimum self-approach gap, or float('inf') if it cannot be determined
              (e.g. every vertex's k nearest neighbors were all within `hops`).
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix

    referenced = np.unique(faces.ravel())
    if len(referenced) < len(verts):
        remap = -np.ones(len(verts), dtype=np.int64)
        remap[referenced] = np.arange(len(referenced))
        verts = verts[referenced]
        faces = remap[faces]

    n = len(verts)
    edges = set()
    for f in faces:
        a, b, c = f
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    edges = np.array(list(edges))

    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row))
    A = csr_matrix((data, (row, col)), shape=(n, n))

    R = A.copy()
    cur = A.copy()
    for _ in range(hops - 1):
        cur = cur.dot(A)
        R = R + cur
    R = (R > 0)

    tree = cKDTree(verts)
    min_gap = float("inf")
    n_exhausted = 0
    for i in range(n):
        dists, idxs = tree.query(verts[i], k=k)
        excl = set(R.getrow(i).indices.tolist())
        excl.add(i)
        found = False
        for d, j in zip(dists, idxs):
            if j not in excl:
                if d < min_gap:
                    min_gap = float(d)
                found = True
                break
        if not found:
            n_exhausted += 1

    if n_exhausted:
        print(f"_min_self_approach_gap: WARNING - {n_exhausted}/{n} vertices exhausted "
              f"k={k} nearest neighbors without finding one outside hops={hops} (dense local "
              f"topology); if this is a large fraction of n, increase k.")
    return min_gap


def _winding_number_naive(verts, faces, query_points, tri_batch_size=2000):
    """
    Exact GWN via the Van Oosterom-Strackee signed solid-angle formula. O(query_points * faces) -
    only used per-tile as a fallback when igl isn't installed, on small per-tile face counts.
    """
    wn = np.zeros(len(query_points), dtype=np.float64)
    q = query_points[:, None, :]
    for start in range(0, len(faces), tri_batch_size):
        batch = faces[start:start + tri_batch_size]
        a = verts[batch[:, 0]][None, :, :] - q
        b = verts[batch[:, 1]][None, :, :] - q
        c = verts[batch[:, 2]][None, :, :] - q
        a_len = np.linalg.norm(a, axis=-1)
        b_len = np.linalg.norm(b, axis=-1)
        c_len = np.linalg.norm(c, axis=-1)
        numerator = np.sum(a * np.cross(b, c), axis=-1)
        denominator = (
            a_len * b_len * c_len
            + np.sum(a * b, axis=-1) * c_len
            + np.sum(b * c, axis=-1) * a_len
            + np.sum(c * a, axis=-1) * b_len
        )
        solid_angle = 2.0 * np.arctan2(numerator, denominator)
        wn += np.sum(solid_angle, axis=-1)
    return wn / (4.0 * np.pi)


def _voxel_remesh_cleanup(obj, voxel_size):
    """
    Runs Blender's built-in Voxel Remesh (OpenVDB-backed) as a robustness pass after
    close_holes_via_winding_number's own tile reconstruction + weld.

    Independent per-tile marching cubes can leave small non-manifold artifacts specifically at
    tile boundaries: adjacent tiles can resolve the SAME locally-ambiguous marching-cubes cube
    configuration differently (a known ambiguity in the algorithm itself), producing overlapping
    double-skin geometry rather than a clean simple hole. This is most common exactly where our
    own results have been worst - thin, high-curvature regions (leg/mandible/antenna tips) - and
    neither vertex welding (the vertices aren't duplicates, they're genuinely different local
    reconstructions) nor loop-based hole capping (which needs a well-formed simple boundary loop
    to walk, and double-skin overlap isn't that) can clean it up.

    OpenVDB's voxel remesh doesn't care about input topology - it derives its own signed-distance
    estimate from whatever geometry exists nearby and re-meshes from scratch, so it's robust to
    exactly this kind of mess, without needing us to hand-solve tile-boundary MC-ambiguity
    consistency ourselves.

    Args:
        obj (bpy.types.Object): The Blender object to remesh in place.
        voxel_size (float): Should be close to close_holes_via_winding_number's own voxel_pitch -
                            matching resolution avoids reintroducing a "different mesh styles
                            stitched together" mismatch (the visual problem from earlier remesh
                            attempts in this pipeline's history, which used an unrelated fixed
                            voxel_size of 1.5 regardless of the surrounding mesh's actual density).

    Returns:
        None
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")
    mod = obj.modifiers.new(name="VoxelRemeshCleanup", type="REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel_size
    mod.adaptivity = 0.0
    try:
        bpy.ops.object.modifier_apply(modifier="VoxelRemeshCleanup")
    except RuntimeError as e:
        if "Modifiers cannot be applied to multi-user data" in str(e):
            bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)
            bpy.ops.object.modifier_apply(modifier="VoxelRemeshCleanup")
        else:
            raise


def close_holes_via_winding_number(
    obj,
    voxel_pitch=None,
    voxel_pitch_edge_percentile=10,
    voxel_pitch_percentile_factor=0.4,
    voxel_pitch_gap_safety_factor=0.5,
    self_approach_gap_hops=4,
    self_approach_gap_k=100,
    tile_voxels=20,
    tile_halo_voxels=4,
    wn_threshold=0.5,
    weld_threshold_factor=0.3,
    max_naive_faces=4000,
    max_tiles=6000,
    max_tile_voxels=100,
    igl_query_batch_size=200_000,
    progress_every=100,
    voxel_remesh_cleanup=True,
    voxel_remesh_size_factor=1.0,
    debug_export_checkpoint_a_path=None,
):
    """
    Rebuilds obj's mesh as a watertight surface via tiled/blocked generalized-winding-number +
    marching cubes. This is the fix for a specific, real failure mode of a SINGLE dense grid over
    the whole bounding box: voxel_pitch there has to be derived from the mesh's overall (median)
    edge length, which is dominated by large flat body faces - so it ends up far too coarse to
    resolve thin cylindrical features (legs, mandibles, antennae), which need ~2-3 voxels across
    their diameter to survive marching cubes at all. Making that global pitch finer directly is
    not viable either: a dense array fine enough for legs, spanning the WHOLE bounding box, blows
    past any reasonable memory budget.

    The fix: only build a dense grid where surface actually exists. The narrow band is
    partitioned into cubic tiles of tile_voxels^3 interior voxels (plus a tile_halo_voxels-voxel
    halo on every side so adjacent tiles' reconstructed surfaces overlap and can be welded
    seamlessly). Only tiles that a triangle's bounding box overlaps are ever processed - this is
    what keeps a genuinely fine, leg-resolving voxel_pitch tractable: total cost scales with
    SURFACE AREA at fine resolution, not bounding-box VOLUME at fine resolution.

    voxel_pitch is derived from a LOW percentile of edge length (default: 10th percentile), not
    the median - the median is dominated by big flat body faces and says nothing about how fine
    the legs/mandibles actually are. The low percentile is a proxy for the finest real features in
    the scan. If legs/mandibles are still incomplete after this runs, first try lowering
    voxel_pitch_edge_percentile (e.g. to 5) or voxel_pitch_percentile_factor (e.g. to 0.3) before
    concluding the approach doesn't work - undersized voxel_pitch is the most common cause.

    Each tile is welded to its neighbors afterward (weld_threshold_factor * voxel_pitch) to merge
    the duplicate geometry produced in the overlapping halo zones into one continuous surface, and
    literal duplicate faces left over from that overlap are removed.

    Uses libigl's fast_winding_number_for_meshes when available (this is what makes per-tile calls
    cheap regardless of total face count - it's a global hierarchical query, not something that
    needs pre-restricting to nearby triangles). Falls back to an exact numpy implementation only
    when the mesh is small enough (<= max_naive_faces) for that to be tractable.

    Args:
        obj (bpy.types.Object): The Blender object to rebuild in place.
        voxel_pitch (float): Grid spacing. If None, derived per voxel_pitch_edge_percentile/factor.
        voxel_pitch_edge_percentile (float): Percentile (0-100) of edge lengths used as the basis
                                             for voxel_pitch. Low values target the finest features.
        voxel_pitch_percentile_factor (float): voxel_pitch = percentile_edge_length * this.
        voxel_pitch_gap_safety_factor (float): After the edge-percentile voxel_pitch above, also
                                               computed against this specimen's own minimum
                                               self-approach gap (see _min_self_approach_gap) -
                                               voxel_pitch is shrunk to min_gap * this factor if
                                               that's finer than the percentile-derived value, so
                                               a tight real gap (e.g. leg resting near body) can
                                               still be resolved as open space rather than fused.
                                               None disables this check entirely.
        self_approach_gap_hops (int): Mesh-edge hop radius passed to _min_self_approach_gap.
        self_approach_gap_k (int): Nearest-neighbor candidate count passed to
                                   _min_self_approach_gap.
        tile_voxels (int): Interior voxels per tile edge.
        tile_halo_voxels (int): Extra halo voxels per side, for seamless inter-tile stitching.
        wn_threshold (float): Isosurface level for marching cubes (0.5 is standard for GWN).
        weld_threshold_factor (float): Vertex-weld distance = this * voxel_pitch, applied across
                                       the whole rebuilt mesh to merge tile-seam duplicates.
        max_naive_faces (int): Face-count ceiling for the numpy fallback (see _winding_number_naive).
        max_tiles (int): Target cap on the number of occupied tiles. If exceeded, tile_voxels is
                         grown adaptively (see max_tile_voxels) to bring tile count down without
                         coarsening voxel_pitch/feature resolution; if still exceeded after that,
                         proceeds anyway with a warning rather than raising - every specimen gets
                         an attempt rather than being silently excluded.
        max_tile_voxels (int): Ceiling on how far tile_voxels can grow while adapting to
                               max_tiles. Bounds per-tile memory/compute (grid_side**3 points per
                               tile) since a single igl/marching_cubes call must hold that many
                               points at once.
        igl_query_batch_size (int): Batches queries to igl.fast_winding_number_for_meshes (mitigates
                                    a known libigl-python-bindings crash on very large single batches).
        progress_every (int): Print a progress line every this many tiles processed.
        debug_export_checkpoint_a_path (str): If given, exports the mesh to this path
                                              immediately after voxel remesh cleanup, before any
                                              degeneracy cleanup - lets follow-up experiments
                                              iterate on that exact intermediate state directly
                                              without re-running the whole pipeline up to it.

    Returns:
        dict: Diagnostics - {"voxel_pitch", "n_tiles", "engine", "output_vertices", "output_faces"}.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise ImportError(
            "close_holes_via_winding_number requires scipy. Install it into the SAME Python "
            "environment Blender's process is actually using (check by seeing which traceback "
            "path scipy/igl import errors show) - e.g. `pip install scipy scikit-image`."
        ) from e
    try:
        from skimage import measure as skimage_measure
    except ImportError as e:
        raise ImportError(
            "close_holes_via_winding_number requires scikit-image for marching_cubes. "
            "`pip install scikit-image scipy` into the same Python environment Blender is using."
        ) from e

    try:
        import igl
        has_igl = hasattr(igl, "fast_winding_number_for_meshes")
        if not has_igl:
            print(
                "WARNING: igl is installed but igl.fast_winding_number_for_meshes is missing "
                "(dropped in libigl 2.6.0/2.6.1 - github.com/libigl/libigl/issues/2477). "
                "Pin pip install \"libigl==2.5.1\". Falling back to per-tile exact numpy winding "
                "number, which is only viable on very small tile face-counts."
            )
    except ImportError:
        igl = None
        has_igl = False
        print(
            "WARNING: igl not installed - using per-tile exact numpy winding number fallback. "
            "This works but is much slower; `pip install \"libigl==2.5.1\"` for real use "
            "(2.6.0/2.6.1 dropped fast_winding_number_for_meshes, pin 2.5.1)."
        )

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")

    verts, faces = _triangulated_verts_faces(obj)
    if len(faces) == 0:
        print("close_holes_via_winding_number: mesh has no faces, skipping.")
        return {"engine": None, "output_vertices": 0, "output_faces": 0}

    edge_lengths = np.linalg.norm(verts[faces[:, [1, 2, 0]]] - verts[faces], axis=-1).ravel()
    p_edge = float(np.percentile(edge_lengths, voxel_pitch_edge_percentile)) if len(edge_lengths) else 1.0
    if p_edge <= 0:
        p_edge = float(np.median(edge_lengths)) if len(edge_lengths) else 1.0
    if p_edge <= 0:
        p_edge = 1.0

    if voxel_pitch is None:
        voxel_pitch = p_edge * voxel_pitch_percentile_factor
    if voxel_pitch <= 0:
        raise ValueError(f"Computed voxel_pitch={voxel_pitch} <= 0; pass voxel_pitch explicitly.")

    # SELF-APPROACH GAP CHECK - was missing entirely before; voxel_pitch was derived purely
    # from the edge-length percentile, with no awareness of how close two anatomically distinct
    # parts of THIS specimen actually get to each other. See _min_self_approach_gap's docstring
    # for the empirical evidence (on real specimens, min_self_approach_gap/voxel_pitch < 1
    # predicted a 37.4% Weld face-loss cascade; every specimen with ratio > 1.2 stayed under
    # 13%). If the edge-percentile-derived voxel_pitch is coarser than this specimen's own
    # tightest real gap, OpenVDB physically cannot represent that gap - it isn't a "Weld is too
    # aggressive" problem, the two surfaces are already fused before Weld ever runs. This only
    # ever makes voxel_pitch FINER, never coarser - it's an additional constraint on top of the
    # existing edge-percentile logic, not a replacement for it. Any resulting tile-count
    # increase is handled by the adaptive tile-budget growth below (grows tile_voxels, not
    # voxel_pitch, so this fix's resolution gain is never silently given back).
    if voxel_pitch_gap_safety_factor is not None:
        min_gap = _min_self_approach_gap(verts, faces, hops=self_approach_gap_hops,
                                          k=self_approach_gap_k)
        if np.isfinite(min_gap):
            gap_bound_pitch = min_gap * voxel_pitch_gap_safety_factor
            print(f"close_holes_via_winding_number: min_self_approach_gap={min_gap:.4g}, "
                  f"gap/voxel_pitch={min_gap / voxel_pitch:.3g}")
            if gap_bound_pitch < voxel_pitch:
                print(f"close_holes_via_winding_number: self-approach gap requires finer "
                      f"voxel_pitch than the edge-percentile alone gave - shrinking "
                      f"{voxel_pitch:.4g} -> {gap_bound_pitch:.4g} (min_gap={min_gap:.4g} * "
                      f"safety_factor={voxel_pitch_gap_safety_factor}) so this specimen's "
                      f"tightest real self-contact gap can still be resolved as open space.")
                voxel_pitch = gap_bound_pitch
        else:
            print("close_holes_via_winding_number: WARNING - could not determine "
                  "min_self_approach_gap (see warning above); proceeding with the "
                  "edge-percentile-only voxel_pitch, unchecked against self-contact gaps.")

    face_min = verts[faces].min(axis=1)
    face_max = verts[faces].max(axis=1)

    def _occupied_tiles_for(tile_voxels_local):
        # Occupied tiles: every tile whose world-space extent overlaps a triangle's bounding
        # box - NOT just tiles containing a vertex. Long/sparse triangles (common after
        # decimation on thin cylindrical geometry) can span a tile without having a single
        # vertex inside it; vertex-only occupancy silently drops such tiles and leaves gaps in
        # exactly the thin-feature regions this function exists to fix.
        tile_world_local = tile_voxels_local * voxel_pitch
        tile_min_idx = np.floor(face_min / tile_world_local).astype(int)
        tile_max_idx = np.floor(face_max / tile_world_local).astype(int)
        occ = set()
        for lo, hi in zip(tile_min_idx, tile_max_idx):
            for tx in range(lo[0], hi[0] + 1):
                for ty in range(lo[1], hi[1] + 1):
                    for tz in range(lo[2], hi[2] + 1):
                        occ.add((tx, ty, tz))
        return occ, tile_world_local

    print(
        f"close_holes_via_winding_number: p{voxel_pitch_edge_percentile}_edge={p_edge:.4g}, "
        f"voxel_pitch={voxel_pitch:.4g} (fixed - thin-feature resolution is never sacrificed "
        f"for tile budget, only tile_voxels is grown, see below)"
    )

    occupied, tile_world = _occupied_tiles_for(tile_voxels)
    n_tiles = len(occupied)
    print(f"close_holes_via_winding_number: {n_tiles:,} occupied tiles at tile_voxels={tile_voxels}")

    # ADAPTIVE TILE BUDGET (was: hard ValueError here, silently excluding any specimen whose
    # tiling exceeded max_tiles from winding_number entirely). Growing tile_voxels enlarges
    # each tile's world-space footprint at the SAME voxel_pitch, so it cuts the NUMBER of tiles
    # (the actual wall-clock cost driver - one igl/marching_cubes call per tile) without
    # coarsening voxel_pitch itself, i.e. without sacrificing the thin-feature (leg/antenna)
    # resolution that is the whole reason this function exists. Tile count scales roughly with
    # 1/tile_world^2 = 1/(tile_voxels*voxel_pitch)^2 for a fixed surface area, so each growth
    # step targets sqrt(n_tiles/max_tiles) directly instead of guessing a step size.
    original_tile_voxels = tile_voxels
    while n_tiles > max_tiles and tile_voxels < max_tile_voxels:
        growth = max(1.15, math.sqrt(n_tiles / max_tiles))
        tile_voxels = min(max_tile_voxels, int(math.ceil(tile_voxels * growth)))
        occupied, tile_world = _occupied_tiles_for(tile_voxels)
        n_tiles = len(occupied)
        print(f"close_holes_via_winding_number: adaptive tile budget - grew tile_voxels "
              f"{original_tile_voxels}->{tile_voxels}, now {n_tiles:,} occupied tiles")

    if n_tiles > max_tiles:
        print(
            f"close_holes_via_winding_number: WARNING - {n_tiles:,} occupied tiles still "
            f"exceeds max_tiles={max_tiles:,} after growing tile_voxels to the max_tile_voxels="
            f"{max_tile_voxels} ceiling. Proceeding anyway (adaptive-budget policy: attempt "
            f"every specimen rather than silently exclude it) - this run will be slow."
        )
    if n_tiles == 0:
        raise RuntimeError("No occupied tiles found - mesh may be empty or degenerate.")

    halo_world = tile_halo_voxels * voxel_pitch
    grid_side = tile_voxels + 2 * tile_halo_voxels + 1
    print(
        f"close_holes_via_winding_number: final tile_voxels={tile_voxels}, tile_world={tile_world:.4g}, "
        f"per-tile grid={grid_side}^3 ({grid_side**3:,} pts)"
    )

    verts_f64 = np.ascontiguousarray(verts, dtype=np.float64)
    faces_i32 = np.ascontiguousarray(faces, dtype=np.int32)
    if len(faces) > max_naive_faces and not has_igl:
        raise ImportError(
            f"igl is not installed and this mesh has {len(faces):,} faces, above "
            f"max_naive_faces={max_naive_faces:,} for the numpy fallback. Install libigl:\n"
            f"  pip install \"libigl==2.5.1\"\n"
            f"into the Python environment Blender's process is actually using."
        )

    all_v_chunks = []
    all_f_chunks = []
    v_offset = 0
    engine = "igl" if has_igl else "naive_numpy"
    tiles_with_surface = 0

    for i, (tx, ty, tz) in enumerate(occupied):
        if progress_every and i > 0 and i % progress_every == 0:
            print(f"close_holes_via_winding_number: tile {i:,}/{n_tiles:,} "
                  f"({tiles_with_surface:,} produced surface so far)")

        tile_min = np.array([tx, ty, tz]) * tile_world - halo_world
        xs = tile_min[0] + np.arange(grid_side) * voxel_pitch
        ys = tile_min[1] + np.arange(grid_side) * voxel_pitch
        zs = tile_min[2] + np.arange(grid_side) * voxel_pitch
        g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

        if has_igl:
            chunks = []
            for start in range(0, len(g), igl_query_batch_size):
                q = np.ascontiguousarray(g[start:start + igl_query_batch_size], dtype=np.float64)
                chunks.append(igl.fast_winding_number_for_meshes(verts_f64, faces_i32, q))
            wn_t = np.concatenate(chunks) if chunks else np.zeros(0)
        else:
            wn_t = _winding_number_naive(verts, faces, g)

        wn_t = wn_t.reshape(grid_side, grid_side, grid_side)
        if wn_t.max() < wn_threshold:
            continue

        try:
            tv, tf, _, _ = skimage_measure.marching_cubes(
                wn_t, level=wn_threshold, spacing=(voxel_pitch, voxel_pitch, voxel_pitch)
            )
        except (ValueError, RuntimeError):
            continue  # degenerate local field (e.g. isolated single-voxel blob) - skip this tile
        tv = tv + tile_min

        all_v_chunks.append(tv)
        all_f_chunks.append(tf + v_offset)
        v_offset += len(tv)
        tiles_with_surface += 1

    if not all_v_chunks:
        raise RuntimeError(
            "No tile produced any surface. voxel_pitch or band placement is likely wrong - check "
            "the p{}_edge value printed above against the mesh's actual scale.".format(
                voxel_pitch_edge_percentile
            )
        )

    raw_v = np.concatenate(all_v_chunks, axis=0)
    raw_f = np.concatenate(all_f_chunks, axis=0)
    print(
        f"close_holes_via_winding_number: {tiles_with_surface:,}/{n_tiles:,} tiles produced surface, "
        f"raw (pre-weld) {len(raw_v):,} verts, {len(raw_f):,} faces via {engine}"
    )

    # Weld duplicate/near-duplicate vertices from tile-halo overlaps, via bmesh (not the numpy
    # union-find used for standalone testing) so this stays consistent with the rest of the
    # pipeline's mesh representation.
    new_bm = bmesh.new()
    bm_verts = [new_bm.verts.new(v) for v in raw_v]
    new_bm.verts.ensure_lookup_table()
    skipped_faces = 0
    for tri in raw_f:
        try:
            new_bm.faces.new([bm_verts[i] for i in tri])
        except ValueError:
            skipped_faces += 1  # literal duplicate face from tile overlap

    bmesh.ops.remove_doubles(new_bm, verts=new_bm.verts, dist=weld_threshold_factor * voxel_pitch)

    # Tile-seam cleanup pass: even after the general weld above, gaps between adjacent tiles can
    # leave a small number of near-duplicate vertices unmerged - not because they're far apart, but
    # because there are simply a lot of seams (one per tile boundary) and each is an independent
    # chance for a vertex pair to land just outside weld_threshold_factor * voxel_pitch. These show
    # up downstream as many tiny (often 1-2 edge) holes scattered across the surface - a "swiss
    # cheese" look - which is a different failure mode from genuinely missing geometry and doesn't
    # get better with finer voxel_pitch (more tiles -> more seams -> if anything, worse).
    # Scoping to boundary-edge vertices only (not the whole mesh) makes a much looser threshold
    # (a full voxel_pitch, vs. weld_threshold_factor*voxel_pitch above) safe to use: it can only
    # merge vertices that are already on an open edge, so it cannot silently fuse unrelated closed
    # interior surfaces the way loosening the global threshold could.
    boundary_verts = [
        v for v in new_bm.verts
        if any(len(e.link_faces) == 1 for e in v.link_edges)
    ]
    n_boundary_before = len(boundary_verts)
    if boundary_verts:
        bmesh.ops.remove_doubles(new_bm, verts=boundary_verts, dist=voxel_pitch)
        bmesh.ops.dissolve_degenerate(new_bm, dist=1e-6, edges=new_bm.edges)
    n_boundary_after = sum(1 for v in new_bm.verts if any(len(e.link_faces) == 1 for e in v.link_edges))
    print(
        f"close_holes_via_winding_number: boundary-seam reweld: {n_boundary_before:,} -> "
        f"{n_boundary_after:,} boundary verts"
    )

    new_bm.normal_update()

    new_bm.to_mesh(obj.data)
    obj.data.update()
    n_final_verts = len(new_bm.verts)
    n_final_faces = len(new_bm.faces)
    new_bm.free()

    print(
        f"close_holes_via_winding_number: after weld -> {n_final_verts:,} verts, "
        f"{n_final_faces:,} faces (skipped {skipped_faces:,} exact-duplicate faces during build; "
        f"was {len(verts):,} verts, {len(faces):,} faces before this step)"
    )
    print("close_holes_via_winding_number: degeneracy check after tiled MC rebuild + tile-seam weld:")
    report_mesh_degeneracy(obj)
    log_mesh_state(obj, "after_tile_weld")

    if voxel_remesh_cleanup:
        remesh_voxel_size = voxel_pitch * voxel_remesh_size_factor
        print(f"close_holes_via_winding_number: voxel remesh cleanup at voxel_size={remesh_voxel_size:.4g} "
              f"(cleans up tile-boundary MC-ambiguity artifacts - see docstring)")
        _voxel_remesh_cleanup(obj, remesh_voxel_size)
        n_final_verts = len(obj.data.vertices)
        n_final_faces = len(obj.data.polygons)
        print(f"close_holes_via_winding_number: after voxel remesh -> {n_final_verts:,} verts, "
              f"{n_final_faces:,} faces")
        print("close_holes_via_winding_number: degeneracy check A (after voxel remesh cleanup):")
        report_mesh_degeneracy(obj)
        log_mesh_state(obj, "after_voxel_remesh")

        if debug_export_checkpoint_a_path:
            # Saves this exact intermediate state (post-voxel-remesh, pre-any-cleanup) to disk
            # so future degeneracy-cleanup experiments can iterate on it directly (plain
            # bpy/bmesh, no scipy/igl needed) without re-paying the ~5-8 min cost of import +
            # ray-cast cleaning + tiled MC + voxel remesh each time.
            #
            # MUST export a throwaway DUPLICATE, not obj itself: export_mesh_to_obj calls
            # bpy.ops.mesh.quads_convert_to_tris() as part of its normal (correct) job for a
            # real final export - caught this the hard way (twice - see below), it silently
            # triangulated obj in place here too (roughly doubled n_faces, and inflated
            # n_zero_length_edges/n_zero_area_faces as a side effect of splitting already-
            # degenerate quads), corrupting the very state the cleanup experiment right after
            # this was supposed to be testing.
            #
            # A first fix attempt created debug_obj but never made it ACTIVE - export_mesh_to_obj
            # itself never sets bpy.context.view_layer.objects.active (it assumes the caller
            # already did), so its bpy.ops.object.mode_set(mode="EDIT")/quads_convert_to_tris()
            # silently operated on whatever was PREVIOUSLY active (obj, still active from
            # _voxel_remesh_cleanup earlier in this function) instead of debug_obj - same
            # corruption, just via a less obvious path. Explicitly managing the active object
            # around the call, and restoring it afterward, is required.
            # Second bug caught here (empirically reproduced + fixed in a 5-second standalone
            # test before touching this pipeline again, not guessed): setting active alone is
            # NOT enough. bpy.ops.object.mode_set(mode="EDIT") enters multi-object edit mode for
            # every currently SELECTED object, not just the active one - obj was still selected
            # from earlier steps in this function, so both obj and debug_obj entered edit mode
            # together and BOTH got triangulated by quads_convert_to_tris(), reproducing the
            # exact same corruption via a second path. Explicit selection management (deselect
            # everything, select only debug_obj) is required, not just setting .active.
            debug_mesh_copy = obj.data.copy()
            debug_obj = obj.copy()
            debug_obj.data = debug_mesh_copy
            bpy.context.collection.objects.link(debug_obj)
            bpy.ops.object.select_all(action="DESELECT")
            debug_obj.select_set(True)
            bpy.context.view_layer.objects.active = debug_obj
            export_mesh_to_obj(debug_obj, debug_export_checkpoint_a_path)
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.data.objects.remove(debug_obj)
            bpy.data.meshes.remove(debug_mesh_copy)
            print(f"close_holes_via_winding_number: DEBUG - exported checkpoint-A mesh to "
                  f"{debug_export_checkpoint_a_path} (obj itself left untouched)")

        # Cleanup moved HERE (was: right before decimate_mesh in process_stl, much later) after
        # that placement made decimation drastically worse (boundary_edges 24-31 -> 49,335+) even
        # though it did reach 0/0/0 degeneracy - almost certainly because remove_doubles/
        # dissolve_degenerate at that late stage restructured local topology across the WHOLE
        # apply_modifiers-processed mesh at once, which Decimate's COLLAPSE then handled worse
        # than the original degenerate-but-locally-simple geometry (echoes the same over-merging
        # mechanism diagnosed for Weld earlier). Cleaning immediately at the source, before
        # apply_modifiers' holes_fill/dissolve_limit and filter_small_components get a chance to
        # build additional topology around the degenerate geometry, is the more targeted fix.
        #
        # Conservative cleanup is used here: exact duplicate vertices are merged, and zero-area
        # faces are removed. This avoids the topology-restructuring behavior that was creating new
        # holes and boundary edges downstream.
        print("close_holes_via_winding_number: cleaning degeneracy conservatively at its source (voxel remesh)...")
        inspect_duplicate_vertex_groups(obj, max_groups=8, max_verts_per_group=2)
        clean_mesh_degeneracy_local(obj, dist=voxel_pitch * 0.01, expand_rings=1)
        n_final_verts = len(obj.data.vertices)
        n_final_faces = len(obj.data.polygons)
        print("close_holes_via_winding_number: degeneracy check B (after conservative cleanup):")
        report_mesh_degeneracy(obj)
        log_mesh_state(obj, "after_conservative_cleanup")

    return {
        "voxel_pitch": voxel_pitch,
        "n_tiles": n_tiles,
        "tiles_with_surface": tiles_with_surface,
        "engine": engine,
        "output_vertices": n_final_verts,
        "output_faces": n_final_faces,
        "voxel_remesh_cleanup": voxel_remesh_cleanup,
    }


def process_stl(
    stl_path,
    output_dir=None,
    max_vertices=20000,
    ray_density=1000,
    secondary_rays=5,
    random_seed=42,
    min_island_faces=4,
    keep_rings=2,
    min_bridge_island_faces=50,
    max_bridge_hops=20,
    hole_fill_method="winding_number",
    winding_number_kwargs=None,
    cap_remaining_holes_flag=True,
):
    """
    Processes an STL file by importing, cleaning, simplifying, and decimating the mesh.

    hole_fill_method: "winding_number" (default) rebuilds a watertight surface via
    close_holes_via_winding_number (tiled GWN + marching cubes) BEFORE apply_modifiers.
    "loop_fill" skips that and relies solely on bmesh.ops.holes_fill inside apply_modifiers (the
    original behavior), kept for A/B comparison.

    winding_number_kwargs: extra kwargs forwarded to close_holes_via_winding_number, e.g.
    {"voxel_pitch_edge_percentile": 5} if legs/mandibles are still incomplete.
    """
    if hole_fill_method not in ("winding_number", "loop_fill"):
        raise ValueError(f"hole_fill_method must be 'winding_number' or 'loop_fill', got {hole_fill_method!r}")

    process_stl_start_time = time.time()

    # Start from a clean scene so process_stl can be called more than once in the same session
    # (e.g. to A/B loop_fill vs winding_number back-to-back) without leftover objects interfering.
    _clear_scene_mesh_objects()

    # Import the STL file
    bpy.ops.wm.stl_import(filepath=stl_path)
    obj = bpy.context.selected_objects[0]

    initial_vertices = len(obj.data.vertices)
    if initial_vertices > 2000000:
        print(f"Initial vertex count: {initial_vertices}. Reducing vertices...")
        reduced_vertices = reduce_vertices_by_distance(obj)
        print(f"Reduced vertex count: {reduced_vertices}")

    find_largest_component(obj)

    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")
    obj.location = Vector((0, 0, 0))

    print("Cleaning internal geometry...")
    clean_internal_geometry(
        obj,
        ray_density,
        secondary_rays,
        random_seed,
        keep_rings=keep_rings,
        min_island_faces=min_island_faces,
        min_bridge_island_faces=min_bridge_island_faces,
        max_bridge_hops=max_bridge_hops,
    )

    if hole_fill_method == "winding_number":
        print("Closing holes via tiled generalized winding number + marching cubes...")
        wn_stats = close_holes_via_winding_number(obj, **(winding_number_kwargs or {}))
        print(f"close_holes_via_winding_number stats: {wn_stats}")
        print(f"DIAGNOSTIC: holes after tiled MC rebuild: {count_holes(obj)}, "
              f"boundary_edges: {count_boundary_edges(obj)}")

        print("Applying simplification modifiers...")
        # run_weld=False: apply_modifiers' own mesh-wide Weld is skipped entirely here, not just
        # "back in" as an earlier version of this comment claimed. Root cause traced empirically
        # (diagnostics/measure_self_approach_gap.py + PROBE_export_precleaning_mesh.py, 4-specimen
        # comparison): Weld's threshold, applied to the freshly tiled-MC-reconstructed dense mesh,
        # doesn't distinguish "duplicate vertex from tile overlap" from "two anatomically distinct
        # surfaces that happen to sit close together" (e.g. a leg resting near the body) - it fuses
        # both indiscriminately, which is what caused holes/non-manifold verts to appear at Weld
        # even on meshes close_holes_via_winding_number had already left genuinely watertight.
        # Tile-seam duplicates are already handled correctly and specifically by
        # close_holes_via_winding_number's own internal remove_doubles + boundary-seam-reweld pass
        # (voxel_pitch-scaled, not mesh-wide), so apply_modifiers' Weld is pure redundant risk here,
        # not a needed cleanup step. fill_holes_sides=0 is a no-op safety net, not the primary
        # mechanism - real closure now depends on voxel_pitch actually resolving this specimen's
        # tightest self-approach gap (see close_holes_via_winding_number's own gap-safety check).
        apply_modifiers(
            obj,
            fill_holes_sides=0,
            min_island_faces=min_island_faces,
            weld_merge_threshold=0.0,
            verbose_diagnostics=True,
            run_edge_split=False,
            run_weld=False,
        )
        print(f"DIAGNOSTIC: holes after apply_modifiers: {count_holes(obj)}, "
              f"boundary_edges: {count_boundary_edges(obj)}")

        removed_islands = filter_small_components(obj, min_faces=min_island_faces)
        print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) after apply_modifiers")

        # Checkpoint C - was already cleaned at its source right after voxel remesh (checkpoint
        # B, inside close_holes_via_winding_number); this checks whether holes_fill/
        # dissolve_limit (run above, inside apply_modifiers) or filter_small_components
        # reintroduced any degeneracy. If C comes back dirty despite B being clean, that
        # pinpoints one of THOSE operators as a second source, not decimate_mesh itself.
        print("close_holes_via_winding_number: degeneracy check C (after apply_modifiers):")
        report_mesh_degeneracy(obj)

        remaining_vertices = decimate_mesh(obj, max_vertices)
        print(f"DIAGNOSTIC: holes after decimate: {count_holes(obj)}, "
              f"boundary_edges: {count_boundary_edges(obj)}")
    else:
        print("Applying simplification modifiers...")
        apply_modifiers(obj, fill_holes_sides=0, min_island_faces=min_island_faces)
        removed_islands = filter_small_components(obj, min_faces=min_island_faces)
        print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) after apply_modifiers")
        remaining_vertices = decimate_mesh(obj, max_vertices)

    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    cov_matrix = np.cov(vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    rotation_matrix = Matrix(eigenvectors).to_4x4().inverted()
    obj.matrix_world = rotation_matrix @ obj.matrix_world
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    print("Aligned model with X-axis based on principal component analysis.")

    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    y_variance = np.var(vertices[:, 1])
    z_variance = np.var(vertices[:, 2])
    if y_variance < z_variance:
        rotation_matrix = Matrix.Rotation(np.pi / 2, 4, "X")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 90 degrees around X-axis to put legs down.")

    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_center = (z_min + z_max) / 2
    z_median = np.median(vertices[:, 2])
    if z_median < z_center:
        rotation_matrix = Matrix.Rotation(np.pi, 4, "X")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Flipped model 180 degrees around X-axis to ensure positive Z is up.")

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    print("Ensured positive Z is up.")

    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    num_slices = 20
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    slice_width = (x_max - x_min) / num_slices
    slice_densities = []
    for i in range(num_slices):
        slice_start = x_min + i * slice_width
        slice_end = slice_start + slice_width
        slice_vertices = vertices[(vertices[:, 0] >= slice_start) & (vertices[:, 0] < slice_end)]
        slice_volume = (
            slice_width
            * (slice_vertices[:, 1].max() - slice_vertices[:, 1].min())
            * (slice_vertices[:, 2].max() - slice_vertices[:, 2].min())
        )
        slice_density = len(slice_vertices) / slice_volume if slice_volume > 0 else 0
        slice_densities.append(slice_density)

    head_end = "start" if np.mean(slice_densities[:3]) < np.mean(slice_densities[-3:]) else "end"
    if head_end == "end":
        rotation_matrix = Matrix.Rotation(np.pi, 4, "Z")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 180 degrees around Z-axis to ensure head is in positive X direction.")

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    remaining_vertices = len(obj.data.vertices)
    print(f"Number of remaining vertices: {remaining_vertices}")

    hole_count = count_holes(obj)
    print(f"Number of holes in the processed mesh (before capping): {hole_count}")
    if hole_fill_method == "winding_number" and hole_count > 0:
        describe_hole_locations(obj, top_n=15)
        if cap_remaining_holes_flag:
            capped, cap_skipped = cap_remaining_holes(obj)
            remaining_vertices = len(obj.data.vertices)
            hole_count = count_holes(obj)
            print(f"After cap_remaining_holes: {remaining_vertices} vertices, "
                  f"{hole_count} holes remaining (capped={capped}, skipped={cap_skipped})")

    n_orphans_removed = remove_orphan_vertices(obj)
    if n_orphans_removed:
        remaining_vertices = len(obj.data.vertices)
        print(f"After remove_orphan_vertices: {remaining_vertices} vertices")

    face_size_cov = calculate_face_size_cov(obj)
    print(f"Coefficient of variation of face sizes: {face_size_cov}")

    mesh_smoothness = calculate_mesh_smoothness(obj)
    print(f"Average angle between face normals: {mesh_smoothness} degrees")

    integrity = mesh_integrity_report(obj)
    
    process_stl_elapsed = time.time() - process_stl_start_time
    specimen_name = os.path.splitext(os.path.basename(stl_path))[0]
    print(
        f"FINAL_SUMMARY: specimen={specimen_name} approach={hole_fill_method} "
        f"vertices={integrity['vertex_count']} faces={integrity['face_count']} "
        f"simple_holes={integrity['simple_holes']} boundary_edges={integrity['boundary_edges']} "
        f"non_manifold_edges={integrity['non_manifold_edges']} "
        f"non_manifold_verts={integrity['non_manifold_verts']} "
        f"is_watertight={integrity['is_watertight']} signed_volume={integrity['signed_volume']} "
        f"face_size_cov={face_size_cov} mesh_smoothness={mesh_smoothness} "
        f"time_sec={process_stl_elapsed:.1f}"
    )

    if output_dir is None:
        output_dir = os.path.dirname(stl_path)
    os.makedirs(output_dir, exist_ok=True)

    original_file_name = os.path.splitext(os.path.basename(stl_path))[0]
    export_path = os.path.join(output_dir, f"{original_file_name}_processed.obj")

    print(f"Exporting the processed mesh to {export_path}...")
    export_mesh_to_obj(obj, export_path)
    print("Mesh exported successfully.")

    json_path = os.path.splitext(stl_path)[0] + ".json"
    if os.path.exists(json_path):
        print(f"Updating JSON file: {json_path}")
        with open(json_path, "r") as f:
            json_data = json.load(f)
        json_data["processed_vertex_count"] = remaining_vertices
        json_data["processed_hole_count"] = hole_count
        json_data["processed_face_size_cov"] = face_size_cov
        json_data["processed_mesh_smoothness"] = mesh_smoothness
        json_data["processed_mesh_integrity"] = integrity
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print("JSON file updated successfully.")
    else:
        print(f"Warning: Corresponding JSON file not found at {json_path}")

    return (remaining_vertices, hole_count, face_size_cov, mesh_smoothness)


def compare_hole_fill_methods(stl_path, output_dir, **kwargs):
    """
    Runs process_stl twice on the same specimen - once with hole_fill_method="loop_fill" (original
    behavior), once with "winding_number" (tiled GWN + marching cubes) - into separate subfolders,
    and prints a side-by-side comparison. Each call starts from a clean scene
    (_clear_scene_mesh_objects, called inside process_stl), so this is safe to run in one session.
    """
    results = {}
    for method in ("loop_fill", "winding_number"):
        print(f"\n{'=' * 70}\nRunning hole_fill_method={method!r}\n{'=' * 70}")
        method_out_dir = os.path.join(output_dir, method)
        os.makedirs(method_out_dir, exist_ok=True)
        vertex_count, hole_count, face_size_cov, mesh_smoothness = process_stl(
            stl_path, output_dir=method_out_dir, hole_fill_method=method, **kwargs
        )
        results[method] = {
            "vertex_count": vertex_count,
            "hole_count": hole_count,
            "face_size_cov": face_size_cov,
            "mesh_smoothness": mesh_smoothness,
        }

    print(f"\n{'=' * 70}\nCOMPARISON: loop_fill vs winding_number\n{'=' * 70}")
    header = f"{'metric':<20} {'loop_fill':>15} {'winding_number':>18}"
    print(header)
    print("-" * len(header))
    for metric in ("vertex_count", "hole_count", "face_size_cov", "mesh_smoothness"):
        lf = results["loop_fill"][metric]
        wn = results["winding_number"][metric]
        print(f"{metric:<20} {lf:>15} {wn:>18}")

    return results


# RESULT OF THE PREVIOUS EXPERIMENT (keep_rings=4, voxel_pitch_edge_percentile=3, factor=0.25):
# both knobs made things WORSE, not better - runtime 4min->43min, final hole count 51->205, and
# bridged-island count 6->70. Diagnosis from that log:
#  - keep_rings=4 didn't produce fewer/cleaner islands, it produced MORE (70 bridge pairs vs 6) -
#    wider ring-expansion around each ray hit also pulls in more small isolated fragments that
#    aren't contiguous with anything at sparse/self-occluded tip regions, and several of the
#    resulting bridges were marginal (gap=46, path_hops=18, right at max_bridge_hops=20).
#  - Finer voxel_pitch means more tiles (854->5266), and MORE TILES MEANS MORE SEAMS: the
#    remaining holes were overwhelmingly tiny (n_edges 1-4) - not missing anatomy, but unwelded
#    tile-boundary slivers. Finer resolution doesn't fix that, it multiplies the number of seams
#    that can produce one.
# Both knobs are reverted back to the settings that actually worked (percentile=10, factor=0.4,
# keep_rings=2). The one new, isolated fix is the boundary-seam reweld pass added inside
# close_holes_via_winding_number, which targets the actual seam-slivers mechanism directly instead
# of trying to out-resolve it.
WINDING_NUMBER_KWARGS = {
    "voxel_pitch_edge_percentile": 10,
    "voxel_pitch_percentile_factor": 0.4,
}


SCRIPT_VERSION = "2026-08-integrity-report-batch-v4"


def main():
    print(f"SCRIPT_VERSION={SCRIPT_VERSION}")
    start_time = time.time()

    mode = "compare"  # default: run both loop_fill and winding_number, as before

    if bpy.context.space_data is not None and bpy.context.space_data.type == "TEXT_EDITOR":
        stl_path = bpy.path.abspath(
            "/home/fabi/dev/SMILify/custom_processing/antscan_data/Vollenhovia_sp._CASENT0745625/Vollenhovia_sp._CASENT0745625.stl"
        )
        output_dir = "/home/fabi/dev/SMILify/custom_processing/antscan_processed_test_winding_number"
    else:
        # Optional 3rd trailing arg "wn_only" skips the internal loop_fill comparison run and just
        # runs winding_number once - for batch comparisons against a SEPARATE baseline script
        # (e.g. the real production pipeline), where re-running loop_fill here would be redundant
        # and roughly double the time per specimen for no new information.
        if len(sys.argv) >= 2 and sys.argv[-1] == "wn_only":
            mode = "wn_only"
            if len(sys.argv) < 4:
                print(
                    "Usage: blender --background --python prepare_antscan_data_for_mesh_fitting_test.py "
                    "-- <input_stl_path> <output_dir> wn_only"
                )
                sys.exit(1)
            stl_path = sys.argv[-3]
            output_dir = sys.argv[-2]
        else:
            if len(sys.argv) < 3:
                print(
                    "Usage: blender --background --python prepare_antscan_data_for_mesh_fitting_test.py "
                    "-- <input_stl_path> <output_dir> [wn_only]"
                )
                sys.exit(1)
            stl_path = sys.argv[-2]
            output_dir = sys.argv[-1]

    print(f"mode={mode}, stl_path={stl_path}, output_dir={output_dir}")

    if mode == "wn_only":
        method_out_dir = os.path.join(output_dir, "winding_number")
        os.makedirs(method_out_dir, exist_ok=True)
        process_stl(
            stl_path,
            output_dir=method_out_dir,
            max_vertices=50000,
            ray_density=1000,
            secondary_rays=10000,
            random_seed=0,
            keep_rings=2,
            hole_fill_method="winding_number",
            winding_number_kwargs=WINDING_NUMBER_KWARGS,
        )
    else:
        compare_hole_fill_methods(
            stl_path,
            output_dir,
            max_vertices=50000,
            ray_density=1000,
            secondary_rays=10000,
            random_seed=0,
            keep_rings=2,
            winding_number_kwargs=WINDING_NUMBER_KWARGS,
        )

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Total processing time: {processing_time:.2f} seconds")


if __name__ == "__main__":
    main()

