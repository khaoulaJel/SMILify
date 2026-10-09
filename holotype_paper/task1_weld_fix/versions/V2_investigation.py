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
import tempfile
import shutil
from collections import deque

# DIAGNOSTICS ADDED (2026-08-14): reuse prepare_antscan_data_for_mesh_fitting_manifold's
# validated general-purpose checks (_min_self_approach_gap, verify_gap_preservation,
# verify_reconstruction_fidelity, report_mesh_degeneracy, _triangulated_verts_faces) rather than
# duplicating them - none of these are method-specific (they only need a mesh snapshot path/a
# vertex-face pair), so they apply just as well to this Weld-based pipeline as they do to
# prepare_antscan_data_for_mesh_fitting_alphawrap.py. Every OTHER function in this file
# (find_largest_component, clean_internal_geometry, apply_modifiers, decimate_mesh, etc.) is left
# completely untouched - this file's actual pipeline behavior does not change, only new
# diagnostics are added around it, so its output on any given specimen is bit-for-bit identical to
# before this instrumentation pass.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_antscan_data_for_mesh_fitting_manifold as base


def _median_edge_length(obj):
    """
    Computes the median edge length of the object's current mesh - a measure of local mesh density that,
    unlike overall bounding-box size, reflects how finely/sparsely triangulated the mesh actually is.

    Args:
        obj (bpy.types.Object): The Blender object to measure.

    Returns:
        float: The median edge length, or 0.0 if the mesh has no edges.
    """
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
):
    """
    Applies a series of modifiers and mesh operations to the given object to simplify and clean up the mesh.

    Args:
        obj (bpy.types.Object): The Blender object to modify.
        edge_split_angle (float): The angle threshold for the Edge Split modifier (in radians).
        weld_merge_threshold (float): If explicitly provided, the distance threshold for the Weld modifier.
                                      If None, weld_merge_threshold is derived from the mesh itself (see below).
        dissolve_angle_limit (float): The angle limit for the Limited Dissolve operation.
        fill_holes_sides (int): The maximum number of sides a hole can have to be filled. 0 means no limit.
        min_island_faces (int): Before welding, any connected component with fewer faces than this is
                                discarded. Prevents small debris fragments (e.g. left over from upstream
                                ray-cast cleaning) from being merged into real geometry by Weld.
        max_weld_face_loss_pct (float): If the Weld step destroys more than this percentage of faces, it
                                        means it merged vertices across what should have been disconnected
                                        mesh regions (a sign the input was a fragmented/non-manifold mesh
                                        and welding produced degenerate geometry). Raises RuntimeError
                                        instead of silently continuing with a corrupted mesh.

    Returns:
        None
    """
    if weld_merge_threshold is None:
        # A threshold based only on the overall bounding box assumes a solid, densely and uniformly
        # triangulated mesh. A sparser or patchier mesh (e.g. after ray-cast based cleaning) has a much
        # smaller "correct" weld distance for the same bbox size, so also derive a threshold from the
        # mesh's own local density (median edge length) and use whichever is more conservative.
        bbox_size = obj.dimensions
        max_dimension = max(bbox_size)
        bbox_threshold = max_dimension * 0.002  # 0.2% of the largest dimension
        print(f"Bounding box size: {bbox_size}, bbox-based threshold: {bbox_threshold}")

        median_edge = _median_edge_length(obj)
        local_threshold = median_edge * 0.3  # 30% of local median edge length
        print(f"Median edge length: {median_edge}, local-density-based threshold: {local_threshold}")

        if median_edge > 0:
            weld_merge_threshold = min(bbox_threshold, local_threshold)
        else:
            weld_merge_threshold = bbox_threshold

        if weld_merge_threshold <= 0:
            print("Warning: Calculated weld_merge_threshold is 0 or negative. Using default value.")
            weld_merge_threshold = 2

        print(f"Using weld_merge_threshold: {weld_merge_threshold}")

    # Apply Edge Split Modifier
    edge_split = obj.modifiers.new(name="EdgeSplit", type="EDGE_SPLIT")
    edge_split.split_angle = edge_split_angle
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Drop small debris islands before welding, so Weld only ever merges vertices within real geometry
    # rather than bridging gaps between disconnected fragments.
    removed_islands = filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) before Weld")

    # Apply Weld Modifier
    face_count_before_weld = len(obj.data.polygons)
    weld = obj.modifiers.new(name="Weld", type="WELD")
    weld.merge_threshold = weld_merge_threshold
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Weld")
    face_count_after_weld = len(obj.data.polygons)

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

    # Switch to Edit Mode for mesh operations
    bpy.ops.object.mode_set(mode="EDIT")

    # Create a BMesh
    bm = bmesh.from_edit_mesh(obj.data)

    # Fill Holes operation using bmesh.ops
    bmesh.ops.holes_fill(bm, edges=bm.edges, sides=fill_holes_sides)

    # Limited Dissolve operation
    bmesh.ops.dissolve_limit(
        bm, angle_limit=dissolve_angle_limit, use_dissolve_boundaries=False, verts=bm.verts, edges=bm.edges
    )

    # Update the mesh
    bmesh.update_edit_mesh(obj.data)

    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Free the BMesh
    bm.free()


def _bfs_path(start_vert, target_verts, max_hops=60):
    """
    Breadth-first search through the mesh's edge graph from start_vert, stopping as soon as a
    vertex in target_verts is reached (or after max_hops).

    Args:
        start_vert (BMVert): The vertex to search from.
        target_verts (set): Set of BMVert to search for.
        max_hops (int): Maximum edge-hops to search before giving up.

    Returns:
        list[BMVert] or None: The path from start_vert to the reached target vertex
                              (inclusive of both ends), or None if unreachable within max_hops.
    """
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
    """
    Some anatomically real but thin connections (e.g. a head-thorax neck) can end up split into
    separate islands because sparse ray sampling under-covers narrow bottlenecks, even with
    multi-ring expansion around each hit. Rather than lowering the global weld threshold (which
    risks fusing unrelated nearby surfaces elsewhere, the original catastrophic failure mode) or
    inventing new connecting geometry, this finds the real shortest path through the ORIGINAL,
    not-yet-deleted mesh graph between islands, and restores that path's real scanned faces into
    vertices_to_keep - i.e. a targeted, local reconnection rather than a global one.

    The decision to bridge is gated on REAL GRAPH PATH LENGTH (max_bridge_hops), not on Euclidean
    proximity - two islands can be spatially close but only reachable via a long, circuitous route
    through the original mesh (e.g. unrelated debris that merely floats nearby), and a naive
    distance-only gate would bridge those just as readily as a genuine severed joint. A short real
    path is what distinguishes "the ray-cast missed a thin neck" from "two unrelated fragments
    happen to be close in space". Euclidean distance is used only as a cheap prefilter to avoid
    running an expensive nearest-point search between every pair of islands.

    Only islands at or above min_bridge_island_faces are considered candidates: this is
    deliberately a much larger threshold than the general debris filter (min_island_faces), since
    indiscriminately path-bridging hundreds of small debris fragments to each other (which a low
    threshold does) recreates a milder version of the original problem - lots of small, arbitrary
    merges - even when each individual merge is short-hop. Bridging is for reconnecting real,
    substantial anatomy, not for stitching debris.

    A genuinely severed piece (e.g. a leg that broke off in the raw scan, not merely an artifact of
    ray sampling) has no short path through the original mesh graph by construction - the BFS will
    either find no path at all (truly disconnected in the source data) or only a long, circuitous
    one, both of which correctly fail the max_bridge_hops gate.

    Args:
        bm (BMesh): The full mesh (before any deletion), with all original faces intact.
        vertices_to_keep (set): Vertex indices marked to survive; mutated in place.
        min_bridge_island_faces (int): Minimum face count for an island to be considered a
                                       bridging candidate at all.
        prefilter_gap_multiplier (float): Cheap bounding-box gap prefilter, in multiples of median
                                          edge length - only a speed optimization to skip obviously
                                          unrelated island pairs; not the correctness-critical gate.
        max_bridge_hops (int): The real accept/reject gate - only bridge if a path of at most this
                               many edges exists through the original mesh graph.
        patch_rings (int): After finding the shortest path, also pull in this many extra
                           topological rings around it, so the recovered geometry is a real patch
                           of surface rather than a single-vertex-wide wire.

    Returns:
        int: The number of island pairs bridged.
    """
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
        """Vertices on the open edge of this island's face subset (where the original mesh's
        other face on that edge belongs to a different island) - the only vertices that can
        plausibly be nearest to a separate island."""
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
                continue  # no short real path - proximity alone doesn't justify bridging

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
    """
    Cleans internal geometry of the object using ray casting.

    Args:
        obj (bpy.types.Object): The Blender object to clean.
        ray_density (int): The density of primary rays to cast.
        secondary_rays (int): The number of secondary rays to cast for each primary ray.
        random_seed (int): Seed for random number generation to ensure consistent results.
        keep_rings (int): Topological hops to expand around each ray-hit face when marking vertices to
                          keep. A single ray hit only guarantees one face is on the surface; expanding
                          several rings makes neighbouring hits overlap into contiguous patches instead of
                          isolated single-face islands, which downstream steps (e.g. Weld in
                          apply_modifiers) cannot safely merge back together.
        min_island_faces (int): After deleting unselected geometry, any remaining connected component with
                                fewer faces than this is discarded as ray-cast noise/debris.
        min_bridge_island_faces (int): Minimum face count for an island to be considered a candidate for
                                       bridge_nearby_islands - deliberately much larger than
                                       min_island_faces, so only substantial (anatomically plausible)
                                       islands get bridged, not small debris fragments.
        max_bridge_hops (int): Passed through to bridge_nearby_islands - the real accept/reject gate for
                               whether two islands get reconnected (see that function's docstring).

    Returns:
        None
    """
    # Set the random seed for numpy and Python's random module
    np.random.seed(random_seed)
    random.seed(random_seed)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")  # Ensure we're in Object mode

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Get the bounding box in world space
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_min = Vector(map(min, zip(*bbox_corners)))
    bbox_max = Vector(map(max, zip(*bbox_corners)))

    # Calculate bounding sphere
    center = (bbox_max + bbox_min) / 2
    radius = (bbox_max - bbox_min).length * 2  # four as large, so hard to sample corners are hit

    def cast_ray(origin, direction):
        """
        Casts a ray from the given origin in the given direction and returns hit information.

        Args:
            origin (Vector): The starting point of the ray.
            direction (Vector): The direction of the ray.

        Returns:
            tuple: A tuple containing hit (bool), location (Vector), and face_index (int).
        """
        hit, loc, norm, face_index = obj.ray_cast(obj.matrix_world.inverted() @ origin, direction)
        return hit, face_index

    def add_face_and_connected(face, vertices_to_keep):
        """
        Adds the given face and all faces within keep_rings topological hops of it to the set of
        vertices to keep.

        Args:
            face (BMFace): The face to add.
            vertices_to_keep (set): The set of vertex indices to keep.

        Returns:
            None
        """
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

    # Set to store indices of vertices to keep
    vertices_to_keep = set()

    # Generate spherical distribution of rays
    phi = np.linspace(0, 2 * np.pi, int(np.sqrt(ray_density)))
    theta = np.linspace(0, np.pi, int(np.sqrt(ray_density)))

    for p in phi:
        for t in theta:
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)

            origin = center + Vector((x, y, z))
            main_direction = (center - origin).normalized()

            # Cast main ray
            hit, face_index = cast_ray(origin, main_direction)
            if hit and face_index < len(bm.faces):
                face = bm.faces[face_index]
                add_face_and_connected(face, vertices_to_keep)

            # Cast secondary rays
            for _ in range(secondary_rays):
                # Generate random offset angles
                azimuth_offset = np.random.uniform(-np.pi / 9, np.pi / 9)  # ±20 degrees
                elevation_offset = np.random.uniform(-np.pi / 9, np.pi / 9)  # ±20 degrees

                # Apply rotation to the main direction
                offset_direction = main_direction.copy()
                offset_direction.rotate(mathutils.Euler((elevation_offset, 0, azimuth_offset)))

                hit, face_index = cast_ray(origin, offset_direction)
                if hit and face_index < len(bm.faces):
                    face = bm.faces[face_index]
                    add_face_and_connected(face, vertices_to_keep)

    # Reconnect real-but-thin anatomical bottlenecks (e.g. the head-thorax neck) that ended up
    # split into separate islands purely because sparse ray sampling under-covers narrow
    # regions - using the still-intact original mesh graph, not synthetic bridging geometry.
    bridged = bridge_nearby_islands(
        bm, vertices_to_keep, min_bridge_island_faces=min_bridge_island_faces, max_bridge_hops=max_bridge_hops
    )
    print(f"Bridged {bridged} nearby island pairs via original-mesh shortest path")

    # Select vertices to keep
    for vert in bm.verts:
        vert.select = vert.index in vertices_to_keep

    # Invert selection
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="INVERT")

    # Delete unselected vertices
    bm.verts.ensure_lookup_table()
    verts_to_remove = [v for v in bm.verts if not v.select]
    bmesh.ops.delete(bm, geom=verts_to_remove, context="VERTS")

    # Update the mesh
    bpy.ops.object.mode_set(mode="OBJECT")  # Ensure we're in Object mode before updating
    bm.to_mesh(mesh)
    mesh.update()

    # Clean up
    bm.free()

    print(f"Vertices kept (pre-island-filter): {len(vertices_to_keep)}")
    print(f"Total vertices after ray-cast cleaning: {len(obj.data.vertices)}")

    removed_islands = filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces)")
    print(f"Total vertices after island filtering: {len(obj.data.vertices)}")


def find_largest_component(obj):
    """
    Finds and keeps only the largest connected component of the mesh.

    Args:
        obj (bpy.types.Object): The Blender object to process.

    Returns:
        None
    """
    bpy.ops.object.mode_set(mode="EDIT")

    mesh = bmesh.from_edit_mesh(obj.data)

    # Select all vertices
    for v in mesh.verts:
        v.select = False
    mesh.verts.ensure_lookup_table()

    # Find the largest coherent mesh
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

    # Select the largest component
    for v in largest_component:
        v.select = True

    # Update the mesh
    bmesh.update_edit_mesh(obj.data)

    # Invert selection and delete vertices
    bpy.ops.mesh.select_all(action="INVERT")
    bpy.ops.mesh.delete(type="VERT")

    bpy.ops.object.mode_set(mode="OBJECT")


def filter_small_components(obj, min_faces=4):
    """
    Removes every connected component (by face adjacency) with fewer than min_faces faces, keeping all
    components that meet or exceed the threshold. Unlike find_largest_component, which keeps only the single
    largest island, this keeps every sufficiently large island - appropriate for meshes (e.g. after ray-cast
    based cleaning) that legitimately consist of several disjoint but valid regions, where discarding
    everything but the largest would delete real geometry (e.g. antennae, mandibles).

    Args:
        obj (bpy.types.Object): The Blender object to process.
        min_faces (int): Minimum face count for a component to survive.

    Returns:
        int: The number of components removed.
    """
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

    # Convert mesh to triangles
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
    """
    Counts the number of holes in the given mesh object.

    Args:
        obj (bpy.types.Object): The Blender object to analyze.

    Returns:
        int: The number of holes in the mesh.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]

    hole_count = 0
    visited_edges = set()

    for start_edge in boundary_edges:
        if start_edge not in visited_edges:
            # Start of a new hole
            current_edge = start_edge
            is_hole = True
            loop_edges = []

            while current_edge not in visited_edges:
                visited_edges.add(current_edge)
                loop_edges.append(current_edge)

                # Find the next edge in the boundary loop
                next_vert = (
                    current_edge.verts[1]
                    if current_edge.verts[0] in current_edge.link_faces[0].verts
                    else current_edge.verts[0]
                )
                next_edges = [e for e in next_vert.link_edges if e in boundary_edges and e != current_edge]

                if not next_edges:
                    # We've reached an open end, not a hole
                    is_hole = False
                    break

                current_edge = next_edges[0]

                if current_edge == start_edge:
                    # We've completed a loop
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
    broken in the Blender viewport - count_holes was never wrong about what it measures, it just
    wasn't measuring the right thing on its own.

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


def calculate_face_size_cov(obj):
    """
    Calculates the coefficient of variation of face sizes in the given mesh object.

    Args:
        obj (bpy.types.Object): The Blender object to analyze.

    Returns:
        float: The standard deviation of face sizes.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    face_areas = [f.calc_area() for f in bm.faces]

    bpy.ops.object.mode_set(mode="OBJECT")

    return np.round(np.std(face_areas) / np.mean(face_areas), 3)


def calculate_mesh_smoothness(obj):
    """
    Calculates the average angle between face normals as a measure of mesh smoothness.

    Args:
        obj (bpy.types.Object): The Blender object to analyze.

    Returns:
        float: The average angle between face normals in degrees.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    total_angle = 0
    total_comparisons = 0

    for face in bm.faces:
        for edge in face.edges:
            adjacent_face = [f for f in edge.link_faces if f != face]
            if adjacent_face:
                angle = face.normal.angle(adjacent_face[0].normal)
                total_angle += math.degrees(angle)
                total_comparisons += 1

    bpy.ops.object.mode_set(mode="OBJECT")

    if total_comparisons > 0:
        average_angle = total_angle / total_comparisons
        return np.round(average_angle, 3)
    else:
        return 0.0


def decimate_mesh(obj, max_vertices):
    """
    Decimates the mesh to reduce the number of vertices.

    Args:
        obj (bpy.types.Object): The Blender object to decimate.
        max_vertices (int): The target maximum number of vertices.

    Returns:
        int: The number of remaining vertices after decimation.
    """
    print("Applying mesh decimation...")
    initial_vertices = len(obj.data.vertices)
    current_vertices = initial_vertices
    iteration_count = 0
    last_vertex_count = current_vertices

    while current_vertices > max_vertices:
        modifier = obj.modifiers.new(name="Decimate", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.use_symmetry = False
        modifier.use_collapse_triangulate = True

        # Apply decimation with a ratio of 0.5 if more than twice the target vertices
        if current_vertices / max_vertices > 2:
            modifier.ratio = 0.5
        else:
            modifier.ratio = max_vertices / current_vertices

        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.modifier_apply(modifier="Decimate")
        except RuntimeError as e:
            if "Modifiers cannot be applied to multi-user data" in str(e):
                print("Making mesh data single-user and retrying...")
                bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)
                bpy.ops.object.modifier_apply(modifier="Decimate")
            else:
                raise

        current_vertices = len(obj.data.vertices)
        print(f"Current vertices after decimation: {current_vertices}")

        # Edge case detection
        iteration_count += 1
        if current_vertices >= last_vertex_count or iteration_count > 10:
            print(f"Decimation stopped after {iteration_count} iterations.")
            break
        last_vertex_count = current_vertices

    return current_vertices


def reduce_vertices_by_distance(obj, target_vertices=1000000, max_iterations=100):
    """
    Iteratively increases the merge distance to reduce the number of vertices.

    Args:
        obj (bpy.types.Object): The Blender object to process.
        target_vertices (int): The target number of vertices.
        max_iterations (int): Maximum number of iterations to attempt.

    Returns:
        int: The final number of vertices.
    """
    initial_vertices = len(obj.data.vertices)
    if initial_vertices <= target_vertices:
        return initial_vertices

    bpy.context.view_layer.objects.active = obj

    for i in range(max_iterations):
        merge_distance = 1 * (2**i)  # Exponentially increase merge distance
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        bpy.ops.object.mode_set(mode="OBJECT")

        current_vertices = len(obj.data.vertices)
        print(f"Iteration {i + 1}: Merge distance = {merge_distance:.6f}, Vertices = {current_vertices}")

        if current_vertices <= target_vertices:
            break

    return current_vertices


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
):
    """
    Processes an STL file by importing, cleaning, simplifying, and decimating the mesh.

    Args:
        stl_path (str): The file path of the STL file to process.
        output_dir (str, optional): The directory to save the processed mesh. If None, saves in the same directory as the input file.
        max_vertices (int): The maximum number of vertices to keep after decimation.
        ray_density (int): The density of primary rays for internal geometry cleaning.
        secondary_rays (int): The number of secondary rays for internal geometry cleaning.
        random_seed (int): Seed for random number generation to ensure consistent results.
        min_island_faces (int): Minimum face count for a connected component to be kept whenever the
                                pipeline filters out small islands (ray-cast cleaning, pre-Weld, and after
                                apply_modifiers). Kept as a single value across all three call sites so
                                "real geometry" vs. "debris" means the same thing throughout the pipeline.
        keep_rings (int): Topological hops to expand around each ray-hit face in clean_internal_geometry.
                          Higher values produce fewer, larger islands at the cost of raycast runtime.
        min_bridge_island_faces (int): Minimum face count for an island to be considered a candidate for
                                       bridge_nearby_islands (real anatomy, not debris).
        max_bridge_hops (int): Real accept/reject gate for bridge_nearby_islands - only reconnect islands
                               with a short real path through the original mesh graph.

    Returns:
        int: The number of remaining vertices after processing.
    """
    process_stl_start_time = time.time()  # ADDED FOR INSTRUMENTATION - feeds FINAL_SUMMARY's time_sec

    # DIAGNOSTICS ADDED (2026-08-14): also accept .obj input (worker_ALT specimens, downloaded
    # from Drive - see prepare_antscan_data_for_mesh_fitting_alphawrap.py's identical fix for the
    # full backstory) alongside the original .stl-only behavior, so the same specimen set can run
    # through both pipelines for a direct comparison. Existing .stl behavior is byte-for-byte
    # unchanged.
    input_ext = os.path.splitext(stl_path)[1].lower()
    if input_ext == ".stl":
        bpy.ops.wm.stl_import(filepath=stl_path)
    elif input_ext == ".obj":
        bpy.ops.object.select_all(action="DESELECT")
        try:
            bpy.ops.wm.obj_import(filepath=stl_path)
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=stl_path)
    else:
        raise ValueError(
            f"process_stl: unsupported input extension {input_ext!r} for {stl_path} - expected "
            f".stl or .obj."
        )
    obj = bpy.context.selected_objects[0]

    # Reduce vertices if necessary
    initial_vertices = len(obj.data.vertices)
    if initial_vertices > 2000000:
        print(f"Initial vertex count: {initial_vertices}. Reducing vertices...")
        reduced_vertices = reduce_vertices_by_distance(obj)
        print(f"Reduced vertex count: {reduced_vertices}")

    # Find and keep only the largest component
    find_largest_component(obj)

    # Set the origin to the center of mass
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")

    # Set the object's location to the world origin
    obj.location = Vector((0, 0, 0))

    # Clean internal geometry using ray casting
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

    # DIAGNOSTICS ADDED (2026-08-14): pre-Weld snapshot + self-approach gap landmark - PRE_WELD
    # is the direct analogue of prepare_antscan_data_for_mesh_fitting_alphawrap.py's
    # PRE_RECONSTRUCTION snapshot: the mesh state immediately before the step this pipeline was
    # originally suspected of silently fusing anatomically-distinct close parts (Weld - see this
    # file's own apply_modifiers docstring and prepare_antscan_data_for_mesh_fitting_alphawrap.py's
    # module docstring for that history). Recording the tightest self-approach gap HERE, before
    # Weld runs, is what makes it possible to empirically test that suspicion on this specimen
    # rather than continue treating it as untested theory.
    diagnostic_dir = os.path.join(tempfile.gettempdir(), "original_pipeline_diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    diagnostic_specimen_name = os.path.splitext(os.path.basename(stl_path))[0]
    pre_weld_diagnostic_path = os.path.join(diagnostic_dir, f"{diagnostic_specimen_name}_PRE_WELD.obj")
    export_mesh_to_obj(obj, pre_weld_diagnostic_path)
    verts, faces = base._triangulated_verts_faces(obj)
    pre_gap, target_a, target_b = base._min_self_approach_gap(verts, faces, return_pair=True)
    print(f"DIAGNOSTIC: pre-Weld mesh saved to {pre_weld_diagnostic_path}. "
          f"min_self_approach_gap={pre_gap}")

    # Apply simplification modifiers
    print("Applying simplification modifiers...")
    apply_modifiers(obj, fill_holes_sides=0, min_island_faces=min_island_faces)  # 0 means fill all holes regardless of size

    # DIAGNOSTICS ADDED (2026-08-14): checkpoint immediately after Weld/EdgeSplit/holes_fill/
    # dissolve_limit (apply_modifiers), BEFORE filter_small_components/decimate_mesh run -
    # isolates Weld's own effect on the tracked gap from whatever decimate_mesh does afterward,
    # same isolation principle prepare_antscan_data_for_mesh_fitting_alphawrap.py's own
    # "DIAGNOSTIC CHECKPOINT" uses for its reconstruction step.
    post_weld_diagnostic_path = os.path.join(
        diagnostic_dir, f"{diagnostic_specimen_name}_POST_WELD_PRE_DECIMATE.obj"
    )
    export_mesh_to_obj(obj, post_weld_diagnostic_path)
    post_weld_gap_check = base.verify_gap_preservation(obj, target_a, target_b, pre_gap,
                                                         correspondence_gate_factor=20)
    print(f"DIAGNOSTIC CHECKPOINT (immediately after Weld, before filter_small_components/"
          f"decimation): verify_gap_preservation={post_weld_gap_check}")
    print(f"DIAGNOSTIC: post-Weld mesh saved to {post_weld_diagnostic_path}.")

    # Remove any remaining debris islands. Uses the same island-size threshold as the earlier filtering
    # steps rather than find_largest_component, since the specimen legitimately consists of several
    # disjoint regions (legs, antennae, mandibles) that a "keep only the single largest component" filter
    # would otherwise delete.
    removed_islands = filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) after apply_modifiers")

    # Apply mesh decimation
    remaining_vertices = decimate_mesh(obj, max_vertices)

    # DIAGNOSTICS ADDED (2026-08-14): final gap-preservation + whole-surface fidelity check,
    # placed HERE (right after decimate_mesh, BEFORE the PCA/orientation block below) rather than
    # at the very end of process_stl - both base.verify_gap_preservation and
    # base.verify_reconstruction_fidelity require the SAME coordinate frame the pre-Weld landmark
    # was recorded in, and the orientation block below bakes rotation into vertex coordinates via
    # bpy.ops.object.transform_apply - running these checks after that point would silently
    # compare two different orientations of the same shape as if they were different geometry
    # (this exact bug was found and fixed in prepare_antscan_data_for_mesh_fitting_alphawrap.py's
    # own fidelity gate - same fix applied here from the start rather than repeating the mistake).
    # target_cell_size has no natural equivalent here (no bisected/derived alpha, unlike
    # alpha_wrap) - derived from the pre-Weld mesh's own p10 edge percentile instead, the same
    # fallback prepare_antscan_data_for_mesh_fitting_alphawrap.py's
    # _decimate_with_fidelity_stopping uses for its manifold_plus case, so the two files' fidelity
    # numbers stay on a directly comparable scale.
    final_gap_check = base.verify_gap_preservation(obj, target_a, target_b, pre_gap,
                                                     correspondence_gate_factor=20)
    print(f"DIAGNOSTIC: final (post-decimation) verify_gap_preservation={final_gap_check}")

    edge_lengths = np.linalg.norm(verts[faces[:, [1, 2, 0]]] - verts[faces], axis=-1).ravel()
    target_cell_size = (float(np.percentile(edge_lengths, 10)) if len(edge_lengths) else 1.0)
    final_fidelity = base.verify_reconstruction_fidelity(pre_weld_diagnostic_path, obj)
    worst_fidelity_deviation = max(final_fidelity["pre_to_post_p99.9"],
                                    final_fidelity["post_to_pre_p99.9"])
    fidelity_gate_factor = 3.0  # same default prepare_antscan_data_for_mesh_fitting_alphawrap.py uses
    fidelity_gate_passed = worst_fidelity_deviation <= fidelity_gate_factor * target_cell_size
    print(f"DIAGNOSTIC: final verify_reconstruction_fidelity={final_fidelity}, "
          f"target_cell_size(p10 edge, pre-Weld)={target_cell_size:.4g}, "
          f"worst_fidelity_deviation={worst_fidelity_deviation:.4g}, "
          f"fidelity_gate_passed={fidelity_gate_passed} "
          f"(deviation <= {fidelity_gate_factor}x target_cell_size)")

    final_degeneracy = base.report_mesh_degeneracy(obj)

    # Get mesh data
    mesh = obj.data

    # Convert vertices to numpy array for easier computation
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])

    # Calculate the covariance matrix
    cov_matrix = np.cov(vertices.T)

    # Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Create rotation matrix to align principal axis with X-axis
    rotation_matrix = Matrix(eigenvectors).to_4x4().inverted()

    # Apply rotation
    obj.matrix_world = rotation_matrix @ obj.matrix_world

    # Apply the rotation to make it permanent
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    print("Aligned model with X-axis based on principal component analysis.")

    # Get mesh data again after the initial alignment
    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])

    # Calculate the variance along Y and Z axes
    y_variance = np.var(vertices[:, 1])
    z_variance = np.var(vertices[:, 2])

    # Determine if we need to rotate 90 degrees around X-axis
    if y_variance < z_variance:
        rotation_matrix = Matrix.Rotation(np.pi / 2, 4, "X")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 90 degrees around X-axis to put legs down.")

    # Ensure the "up" direction is positive Z
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_center = (z_min + z_max) / 2
    z_median = np.median(vertices[:, 2])

    if z_median < z_center:
        rotation_matrix = Matrix.Rotation(np.pi, 4, "X")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Flipped model 180 degrees around X-axis to ensure positive Z is up.")

    # Apply the rotations to make them permanent
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # After ensuring positive Z is up
    print("Ensured positive Z is up.")

    # Get mesh data again
    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])

    # Divide the model into slices along the X-axis
    num_slices = 20
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    slice_width = (x_max - x_min) / num_slices

    slice_densities = []
    for i in range(num_slices):
        slice_start = x_min + i * slice_width
        slice_end = slice_start + slice_width
        slice_vertices = vertices[(vertices[:, 0] >= slice_start) & (vertices[:, 0] < slice_end)]

        # Calculate the density of the slice (number of vertices / volume)
        slice_volume = (
            slice_width
            * (slice_vertices[:, 1].max() - slice_vertices[:, 1].min())
            * (slice_vertices[:, 2].max() - slice_vertices[:, 2].min())
        )
        slice_density = len(slice_vertices) / slice_volume if slice_volume > 0 else 0
        slice_densities.append(slice_density)

    # The end with lower density is likely to be the antennae end (head)
    head_end = "start" if np.mean(slice_densities[:3]) < np.mean(slice_densities[-3:]) else "end"

    if head_end == "end":
        rotation_matrix = Matrix.Rotation(np.pi, 4, "Z")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 180 degrees around Z-axis to ensure head is in positive X direction.")

    # Apply the rotation to make it permanent
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Report the number of remaining vertices
    remaining_vertices = len(obj.data.vertices)
    print(f"Number of remaining vertices: {remaining_vertices}")

    # After all processing steps and before exporting
    hole_count = count_holes(obj)
    print(f"Number of holes in the processed mesh: {hole_count}")

    # Calculate standard deviation of face sizes
    face_size_cov = calculate_face_size_cov(obj)
    print(f"Coefficient of variation of face sizes: {face_size_cov}")

    # Calculate mesh smoothness
    mesh_smoothness = calculate_mesh_smoothness(obj)
    print(f"Average angle between face normals: {mesh_smoothness} degrees")

    # ADDED FOR INSTRUMENTATION - see mesh_integrity_report docstring. This is the "original"
    # baseline's own honest is_watertight/FINAL_SUMMARY line, in the same format the test script
    # uses (approach=original vs. approach=loop_fill/winding_number there), so runs can be
    # compared directly by grepping FINAL_SUMMARY out of both scripts' logs.
    integrity = mesh_integrity_report(obj)
    process_stl_elapsed = time.time() - process_stl_start_time
    specimen_name = os.path.splitext(os.path.basename(stl_path))[0]
    print(
        f"FINAL_SUMMARY: specimen={specimen_name} approach=original "
        f"vertices={integrity['vertex_count']} faces={integrity['face_count']} "
        f"simple_holes={integrity['simple_holes']} boundary_edges={integrity['boundary_edges']} "
        f"non_manifold_edges={integrity['non_manifold_edges']} "
        f"non_manifold_verts={integrity['non_manifold_verts']} "
        f"is_watertight={integrity['is_watertight']} signed_volume={integrity['signed_volume']} "
        f"face_size_cov={face_size_cov} mesh_smoothness={mesh_smoothness} "
        f"pre_gap={pre_gap:.4g} "
        f"post_weld_gap_verdict={post_weld_gap_check['verdict']!r} "
        f"final_gap_verdict={final_gap_check['verdict']!r} "
        f"fidelity_gate_passed={fidelity_gate_passed} "
        f"worst_fidelity_deviation={worst_fidelity_deviation:.4g} "
        f"target_cell_size={target_cell_size:.4g} "
        f"time_sec={process_stl_elapsed:.1f}"
    )

    # Determine the output directory
    if output_dir is None:
        output_dir = os.path.dirname(stl_path)
    os.makedirs(output_dir, exist_ok=True)

    if output_dir is None:
        output_dir = os.path.dirname(stl_path)
    os.makedirs(output_dir, exist_ok=True)

    original_file_name = os.path.splitext(os.path.basename(stl_path))[0]
    export_path = os.path.join(output_dir, f"{original_file_name}_processed.obj")

    print(f"Exporting the processed mesh to {export_path}...")
    export_mesh_to_obj(obj, export_path)
    print("Mesh exported successfully.")

    # Update the corresponding JSON file with vertex and hole count
    json_path = os.path.splitext(stl_path)[0] + ".json"
    if os.path.exists(json_path):
        print(f"Updating JSON file: {json_path}")
        with open(json_path, "r") as f:
            json_data = json.load(f)

        json_data["processed_vertex_count"] = remaining_vertices
        json_data["processed_hole_count"] = hole_count
        json_data["processed_face_size_cov"] = face_size_cov
        json_data["processed_mesh_smoothness"] = mesh_smoothness
        json_data["processed_mesh_integrity"] = integrity  # ADDED FOR INSTRUMENTATION
        # DIAGNOSTICS ADDED (2026-08-14): pre_gap/target_cell_size derived from the pre-Weld
        # snapshot; post_weld_gap_check isolates Weld's own effect; final_gap_check/
        # verify_reconstruction_fidelity isolate the whole pipeline's combined effect (Weld +
        # decimate_mesh). Field names deliberately match
        # prepare_antscan_data_for_mesh_fitting_alphawrap.py's JSON fields where the concept is
        # the same, so both files' JSON records can be compared directly.
        json_data["processed_pre_gap"] = pre_gap
        json_data["processed_post_weld_gap_check"] = post_weld_gap_check
        json_data["processed_gap_check"] = final_gap_check
        json_data["processed_fidelity"] = final_fidelity
        json_data["processed_worst_fidelity_deviation"] = worst_fidelity_deviation
        json_data["processed_target_cell_size"] = target_cell_size
        json_data["processed_fidelity_gate_passed"] = fidelity_gate_passed
        json_data["processed_mesh_degeneracy"] = final_degeneracy

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print("JSON file updated successfully.")
    else:
        print(f"Warning: Corresponding JSON file not found at {json_path}")

    return (
        remaining_vertices,
        hole_count,
        face_size_cov,
        mesh_smoothness,
    )  # Return vertex count, hole count, face size cov, and mesh smoothness


SCRIPT_VERSION = "2026-08-original-instrumented-v2-gap-fidelity"  # ADDED FOR INSTRUMENTATION


def main():
    print(f"SCRIPT_VERSION={SCRIPT_VERSION}")  # ADDED FOR INSTRUMENTATION
    start_time = time.time()  # Start the timer

    # Check if the script is run from Blender's text editor
    if bpy.context.space_data is not None and bpy.context.space_data.type == "TEXT_EDITOR":
        # Running within Blender
        stl_path = bpy.path.abspath(
            "/home/fabi/dev/SMILify/custom_processing/antscan_data/Acanthomyrmex_glabfemoralis_CASENT0744002/Acanthomyrmex_glabfemoralis_CASENT0744002.stl"
        )  # Update this path
        stl_path = bpy.path.abspath(
            "/home/fabi/dev/SMILify/custom_processing/antscan_data/Platythyrea_MG01_CASENT0840864-D4/Platythyrea_MG01_CASENT0840864-D4.stl"
        )  # Update this path
        output_dir = "/home/fabi/dev/SMILify/custom_processing/antscan_processed"  # if not provided, saves in the same directory as the input file
    else:
        # Running as a standalone script
        if len(sys.argv) < 3:
            print(
                "Usage: blender --background --python prepare_antscan_data_for_mesh_fitting.py -- <input_stl_path> <output_dir>"
            )
            sys.exit(1)
        stl_path = sys.argv[-2]
        output_dir = sys.argv[-1]

    vertex_count, hole_count, face_size_cov, mesh_smoothness = process_stl(
        stl_path, output_dir=output_dir, max_vertices=50000, ray_density=1000, secondary_rays=10000, random_seed=0
    )
    print(f"Processed STL file. Final vertex count: {vertex_count}")
    print(f"Number of holes in the processed mesh: {hole_count}")
    print(f"Coefficient of variation of face sizes: {face_size_cov}")
    print(f"Mesh smoothness (average angle between face normals): {mesh_smoothness} degrees")

    end_time = time.time()  # Stop the timer
    processing_time = end_time - start_time
    print(f"Total processing time: {processing_time:.2f} seconds")


if __name__ == "__main__":
    main()