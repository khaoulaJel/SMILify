"""
Merged production pipeline for antscan STL -> processed OBJ conversion.

This is the ORIGINAL pipeline (loop_fill hole-closing via bmesh.ops.holes_fill inside
apply_modifiers, PCA-based reorientation, decimation to a target vertex count) with exactly
THREE additions, each independently verified during the adaptive-pitch investigation
(ADAPTIVE_PITCH_INVESTIGATION_REPORT.md) to have no drawback on any specimen:

  1. DETERMINISM FIX (find_largest_component, bridge_nearby_islands): both functions
     previously used Python set.pop()/unvisited.pop() for traversal order, which depends on
     per-process hash randomization rather than the pipeline's own random_seed parameter.
     Confirmed empirically to move a real specimen's tightest self-approach gap by ~40%
     between identical runs (Bothroponera: 2.928 vs 4.109). Fixed by replacing with
     deterministic sorted-by-index selection. Verified bit-identical across 5 repeated runs on
     all 10 specimens tested, and the corrected values independently confirmed *correct* (not
     just stable) via a from-scratch spatial-hash-grid algorithm sharing no code with the fix.

  2. HARDENED decimate_mesh: the original made one decimation pass per outer iteration at a
     single fixed ratio, with no check that the pass didn't damage mesh topology, and no error
     if it made zero progress. Replaced with a version that verifies boundary_edges==0 and
     non_manifold_edges==0 after every attempt, retries at a gentler ratio if damaged (rather
     than silently keeping a corrupted result or giving up on the whole call), and now RAISES
     clearly if it ever makes zero net progress instead of returning the mesh unchanged with no
     signal a caller could distinguish from "target already met".

  3. DIAGNOSTIC TOOLKIT (report_mesh_degeneracy, count_boundary_edges, remove_orphan_vertices,
     _find_exact_duplicate_vertex_groups): read-only (except remove_orphan_vertices, which only
     ever deletes literal zero-face vertices - a no-op on output from this pipeline's own
     loop_fill path) additions that give visibility mesh_integrity_report alone doesn't have -
     geometric degeneracy (zero-length edges, zero-area faces, duplicate vertices) is a
     completely separate property from combinatorial manifoldness and can be present even on a
     mesh mesh_integrity_report reports as fully watertight.

Deliberately NOT included, because each has a confirmed, specific drawback and is not yet safe
to run unattended over a full batch (see ADAPTIVE_PITCH_INVESTIGATION_REPORT.md for the full
evidence trail):
  - hole_fill_method="winding_number" (tiled generalized-winding-number reconstruction): even in
    its plain, non-adaptive form, this was found to carry a pipeline-wide tile-seam overlap
    defect (a near-duplicate-face-pair signature on 95.9%/82.3% of non-manifold edges on two
    specimens tested), invisible to mesh_integrity_report's own is_watertight check because it
    predates the investigation that built tools able to see it.
  - adaptive_pitch_targets / corner-unification patch / T-junction healing: real, partial
    progress, but twice regressed under generalization and confirmed structurally incomplete.
  - hole_fill_method="manifold_external": needed its own fallback verification apparatus
    specifically because the naive version silently fused tight self-approach gaps on 3 of 6
    tested specimens; not safe to run without that whole apparatus alongside it.

This file's default behavior (hole_fill_method concept doesn't exist here - it's always the
original loop_fill path) is otherwise BYTE-FOR-BYTE the same pipeline Fabian's original script
runs, on any specimen that doesn't hit the (now much rarer, and now loudly-reported instead of
silent) decimation edge case.
"""

import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
import math
import addon_utils
import numpy as np
import os
import time
import sys
import random
import json
from collections import deque, Counter


def ensure_addon_enabled(addon_name):
    """
    Ensures that the specified Blender addon is enabled.

    Args:
        addon_name (str): The name of the addon to enable.

    Returns:
        None
    """
    if not addon_utils.check(addon_name)[0]:
        addon_utils.enable(addon_name, default_set=True)
        print(f"Enabled addon: {addon_name}")


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

    # Drop small debris islands before welding, so Weld only ever merges vertices within real
    # geometry rather than bridging gaps between disconnected fragments.
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

    DETERMINISM: seed selection and vertex-pair iteration are both explicitly ordered by index
    (min(unvisited, key=lambda f: f.index), sorted(boundary_a/b, key=lambda v: v.index)) rather
    than relying on Python's set.pop()/plain set iteration, which depends on per-process hash
    randomization. Confirmed empirically (ADAPTIVE_PITCH_INVESTIGATION_REPORT.md) that the
    unordered version could produce different island-bridging decisions - and thus a different
    tightest self-approach gap - across identical runs of the same specimen (Bothroponera:
    2.928 vs 4.109 gap, purely from hash-seed variance). Verified bit-identical across 5 runs on
    10 specimens post-fix, with the corrected values independently confirmed correct (not just
    stable) via a from-scratch algorithm.

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
        seed = min(unvisited, key=lambda f: f.index)
        unvisited.discard(seed)
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
            for va in sorted(boundary_a, key=lambda v: v.index):
                for vb in sorted(boundary_b, key=lambda v: v.index):
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
    # Set the random seed for numpy and Python's random module. NOTE: this seeds numpy.random /
    # random only - it does NOT make find_largest_component/bridge_nearby_islands deterministic
    # on its own, since those previously depended on Python's per-process hash randomization
    # instead. Both are now fixed to use explicit sorted-by-index ordering (see their own
    # docstrings) so this seed is sufficient again.
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

    DETERMINISM: seed and traversal-frontier selection are both explicitly ordered by vertex
    index (min(unvisited/to_visit, key=lambda v: v.index)) rather than set.pop(), which depends
    on per-process hash randomization. See bridge_nearby_islands' docstring for the empirical
    evidence this mattered in practice (a real specimen's tightest self-approach gap moved ~40%
    across identical runs before this fix).

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
        current = min(unvisited, key=lambda v: v.index)
        unvisited.discard(current)
        component = set([current])
        to_visit = set([current])

        while to_visit:
            current = min(to_visit, key=lambda v: v.index)
            to_visit.discard(current)
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

    Not touched by the determinism fix: this function's traversal order affects which SET of
    small components get removed together in a batch, but not the final KEPT geometry (every
    component below threshold is removed regardless of visit order, and every component at/above
    threshold survives regardless of visit order) - so plain set.pop() here has no bearing on
    the output, unlike find_largest_component/bridge_nearby_islands where visit order directly
    picks which geometry among several plausible choices gets kept.

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

    NOTE: only sees edges with exactly 1 linked face (simple open boundary) - has zero
    visibility into non-manifold overlap (3+ linked faces). Use mesh_integrity_report for a
    complete picture; this is kept for continuity with prior logs and as the fast/simple check.

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
    edge). That specific gap is why "0 holes" could be reported on a mesh that still looked
    visibly broken - count_holes was never wrong about what it measures, it just wasn't
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
    """
    Cheap standalone count of boundary edges (1 linked face) - a lighter-weight companion to
    mesh_integrity_report when only this one number is needed (e.g. before/after a specific
    operation, without paying for the full report's other checks each time).

    Args:
        obj (bpy.types.Object): The Blender object to analyze.

    Returns:
        int: The number of boundary edges.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    n = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    return n


def remove_orphan_vertices(obj):
    """
    Deletes vertices with zero linked faces. A zero-face vertex has BMVert.is_manifold == False
    in Blender, so these would otherwise silently inflate mesh_integrity_report's
    non_manifold_verts count even though they carry no real geometry.

    Defensive end-of-pipeline sweep - a no-op (returns 0, changes nothing) on any mesh from this
    file's own loop_fill path, which has no operation that produces zero-face vertices. Kept as
    a cheap safety net in case that ever changes, and because mesh_integrity_report's numbers
    are only trustworthy if orphan vertices aren't quietly contaminating them.

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
    if orphans:
        print(f"remove_orphan_vertices: removed {len(orphans)} zero-face orphan vertices")
    return len(orphans)


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

    Skips face-pairs adjacent to a degenerate (near-zero-area) face, which has no well-defined
    normal and would otherwise crash this calculation rather than just being excluded from it.

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
    """
    Groups vertex indices (by position, into `coords`) into sets of exact-duplicate-position
    groups. Helper for report_mesh_degeneracy.

    Args:
        coords (np.ndarray): (N, 3) vertex positions.

    Returns:
        list[np.ndarray]: One array of indices per group of 2+ coincident vertices.
    """
    if len(coords) == 0:
        return []

    _, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    return [np.flatnonzero(inverse == group_id) for group_id in np.nonzero(counts > 1)[0]]


def report_mesh_degeneracy(obj, zero_len_eps=1e-8, zero_area_eps=1e-9, dup_vert_decimals=7):
    """
    Checks for GEOMETRIC degeneracy - zero-length edges, zero-area faces, duplicate vertices,
    duplicate faces - which mesh_integrity_report does NOT check at all. Manifoldness
    (boundary_edges/non_manifold_edges/is_watertight) is a purely COMBINATORIAL property (face
    count per edge, vertex fan connectivity); a mesh can be perfectly manifold and watertight by
    that definition while still containing zero-length edges or duplicate vertices sitting on
    top of each other - those are a separate, geometric kind of brokenness the combinatorial
    checks cannot see.

    Read-only diagnostic - never modifies the mesh. Useful to call before/after any operation
    whose safety depends on the input actually being geometrically clean, not just manifold.

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


def report_bad_edges(obj, limit=20):
    """
    Lists edges with a face count other than 2 (boundary edges: 1 face; non-manifold edges: 3+)
    on obj's CURRENT mesh, with their vertex coordinates - for inspecting WHERE a decimation
    attempt introduced topology damage, before deciding how to fix it. Called internally by
    decimate_mesh on the first damaging retry per outer iteration; also usable standalone.

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
    Checks whether decimation damage concentrates on genuinely thin/tip-like local geometry
    (a sharp convergence point collapsing past survivability) rather than two distant surfaces
    fusing: for every vertex in the PRE-decimation mesh, computes its minimum incident edge
    length, then checks where each bad-edge vertex (from the DAMAGED mesh, matched to its
    nearest pristine vertex by 3D position - decimation moves/merges vertices, so indices don't
    correspond between the two meshes) falls in that mesh-wide distribution. If nearly all bad
    vertices match to pristine vertices already in the bottom `percentile`% of edge lengths,
    that's specific evidence for the thin-tip explanation over the fusion one (which would show
    ordinary, not unusually-short, local edge lengths at the fusion site itself).

    Requires scipy (from scipy.spatial import cKDTree). Called internally by decimate_mesh's
    diagnostic path on the first damaging retry; also usable standalone.

    Args:
        pristine_mesh_data (bpy.types.Mesh): The mesh datablock from BEFORE the damaging
                                             decimation attempt.
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


def decimate_mesh(obj, max_vertices, min_ratio_step=0.99, max_retries_per_step=6, max_iterations=10):
    """
    Decimates the mesh to reduce the number of vertices, verifying after every attempt that the
    result is still fully manifold (boundary_edges==0 and non_manifold_edges==0) rather than
    trusting a single decimation pass blindly.

    HARDENED vs. the original single-ratio version: Decimate's COLLAPSE mode can merge two
    vertices that are close in 3D but topologically distant (the same mechanism that made Weld
    risky near tight self-approach gaps) - a step that does this damages the mesh silently
    unless checked for. If a step damages topology, this retries at a gentler ratio (halving the
    remaining distance to ratio=1.0 each time) rather than giving up on the whole call or
    keeping the damaged result. On the first damaging retry per outer iteration, logs which
    vertices were affected and whether they correspond to genuinely thin/tip-like geometry in
    the original mesh (via report_bad_edges + check_bad_verts_edge_length_percentile), since
    that distinguishes "this specimen's anatomy genuinely can't decimate further here" from a
    step-size problem.

    Requires scipy (used inside check_bad_verts_edge_length_percentile, only invoked on the
    diagnostic path when a step is damaging).

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
        max_iterations (int): Cap on outer-loop iterations. Each non-damaging iteration reduces
                              by ~0.8x, so reaching a target that requires more than ~10x total
                              reduction needs more than the default of 10.

    Returns:
        int: The final vertex count.

    Raises:
        RuntimeError: if the mesh started above max_vertices*1.05 and decimation made ZERO net
                     progress across the entire call - every ratio tried was rejected because
                     boundary_edges/non_manifold_edges were already nonzero BEFORE decimation
                     ever ran (decimation only removes geometry, it cannot heal pre-existing
                     topology defects from an earlier pipeline stage - such a mesh should never
                     reach this function in the first place if apply_modifiers ran correctly
                     upstream). The original version silently returned the mesh unchanged in
                     this case, indistinguishable from "target already met" - this raises
                     instead so the failure is never silent.
    """
    print("Applying mesh decimation...")

    current_vertices = len(obj.data.vertices)
    starting_vertices = current_vertices
    iteration_count = 0
    last_vertex_count = current_vertices

    while current_vertices > max_vertices * 1.05:
        # Pristine snapshot of this outer iteration's starting mesh - never mutated directly;
        # each retry works on its own throwaway copy of it, so a damaging attempt at one ratio
        # doesn't corrupt the starting point for the next, gentler retry.
        outer_input_mesh = obj.data
        pristine_backup = outer_input_mesh.copy()
        obj.data = pristine_backup
        bpy.data.meshes.remove(outer_input_mesh)
        backup_vertex_count = current_vertices

        if current_vertices / max_vertices > 2:
            ratio = 0.8
        else:
            ratio = (max_vertices * 1.05) / current_vertices

        # Retry with a gentler ratio instead of giving up on the whole outer loop after one
        # damaging attempt - a single bad ratio doesn't mean this mesh can't be decimated at
        # all, it means THIS step size collapsed an edge across a genuine geometric problem.
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
                # Only on the first failure per outer iteration - representative, not one sample
                # of many. pristine_backup is still untouched at this point (revert hasn't
                # happened yet), so this is the one moment both damaged and pristine states
                # coexist - required to test whether bad vertices correspond to genuinely
                # short-edge (thin/tip) regions in the ORIGINAL geometry.
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

        if iteration_count > max_iterations:
            print(f"Decimation stopped: iteration limit ({max_iterations}) reached.")
            break

        last_vertex_count = current_vertices

    if current_vertices >= starting_vertices and starting_vertices > max_vertices * 1.05:
        # Zero net progress across the ENTIRE call - every ratio tried, down to min_ratio_step,
        # was rejected because the mesh was not fully manifold BEFORE decimation ever ran.
        # Decimation cannot heal pre-existing topology defects, it can only remove geometry, so
        # this indicates a problem upstream (apply_modifiers not closing the mesh fully) rather
        # than anything decimate_mesh itself can fix by retrying differently.
        integrity = mesh_integrity_report(obj)
        raise RuntimeError(
            f"decimate_mesh: made ZERO progress - started and ended at {current_vertices:,} "
            f"vertices (target was {max_vertices:,}). Every decimation ratio tried was rejected "
            f"because the mesh is not fully manifold BEFORE decimation even ran: "
            f"boundary_edges={integrity['boundary_edges']}, "
            f"non_manifold_edges={integrity['non_manifold_edges']} (both must be 0 for any step "
            f"to be accepted - decimation cannot heal pre-existing topology defects, it can only "
            f"remove geometry). Close those defects upstream (check apply_modifiers' output) "
            f"before calling decimate_mesh - no ratio or iteration count can fix an unsatisfiable "
            f"starting condition."
        )

    return current_vertices


def reduce_vertices_by_distance(obj, target_vertices=1000000, max_iterations=100):
    """
    Iteratively increases the merge distance to reduce the number of vertices. Only invoked for
    very large raw scans (>2M vertices) before any other processing, as a coarse pre-reduction.

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

    last_vertices = initial_vertices
    stalled_iterations = 0
    for i in range(max_iterations):
        merge_distance = 1 * (2**i)  # Exponentially increase merge distance
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        bpy.ops.object.mode_set(mode="OBJECT")

        current_vertices = len(obj.data.vertices)
        print(f"Iteration {i + 1}: Merge distance = {merge_distance:.6f}, Vertices = {current_vertices}")

        if current_vertices <= target_vertices:
            break

        # Some specimens' geometry never yields to remove_doubles regardless of threshold
        # (e.g. Polyergus_samurai_CASENT0743790: 0 vertices removed through iteration 84,
        # merge_distance ~9.6e24 - larger than any physically meaningful unit). Without this
        # guard the loop burns all max_iterations doublings - and, inside a SLURM batch job,
        # can consume the entire remaining job time on a single specimen. Bail out once two
        # consecutive iterations make no progress; the caller falls through to decimate_mesh,
        # which has its own convergence handling.
        if current_vertices == last_vertices:
            stalled_iterations += 1
            if stalled_iterations >= 2:
                print(
                    f"reduce_vertices_by_distance: no progress for {stalled_iterations} "
                    f"consecutive iterations at merge_distance={merge_distance:.6f} - stopping "
                    f"early rather than continuing to double toward max_iterations={max_iterations}."
                )
                break
        else:
            stalled_iterations = 0
        last_vertices = current_vertices

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

    Pipeline: import -> keep largest component -> center at origin -> ray-cast internal-
    geometry cleaning (with bridging for thin real anatomy) -> apply_modifiers (EdgeSplit,
    debris filter, Weld, holes_fill, limited dissolve) -> decimate to max_vertices -> PCA-based
    reorientation (X-axis = principal axis, Z-axis = up, head = +X) -> export.

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
        tuple: (remaining_vertices, hole_count, face_size_cov, mesh_smoothness).
    """
    process_stl_start_time = time.time()

    # Import the STL file
    try:
        bpy.ops.wm.stl_import(filepath=stl_path)
    except AttributeError:
        ensure_addon_enabled("io_mesh_stl")
        bpy.ops.import_mesh.stl(filepath=stl_path)
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

    # Apply simplification modifiers
    print("Applying simplification modifiers...")
    apply_modifiers(obj, fill_holes_sides=0, min_island_faces=min_island_faces)  # 0 means fill all holes regardless of size

    # Remove any remaining debris islands. Uses the same island-size threshold as the earlier
    # filtering steps rather than find_largest_component, since the specimen legitimately
    # consists of several disjoint regions (legs, antennae, mandibles) that a "keep only the
    # single largest component" filter would otherwise delete.
    removed_islands = filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) after apply_modifiers")

    # Apply mesh decimation
    remaining_vertices = decimate_mesh(obj, max_vertices)

    # Defensive sweep - a no-op on this pipeline's own output, kept so mesh_integrity_report's
    # non_manifold_verts count below is never silently contaminated by zero-face vertices.
    n_orphans_removed = remove_orphan_vertices(obj)
    if n_orphans_removed:
        remaining_vertices = len(obj.data.vertices)

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

    # Full integrity report (boundary_edges, non_manifold_edges, non_manifold_verts,
    # is_watertight, signed_volume) plus geometric-degeneracy report (zero-length edges,
    # zero-area faces, duplicate verts/faces) - two independent properties, both logged.
    integrity = mesh_integrity_report(obj)
    degeneracy = report_mesh_degeneracy(obj)

    process_stl_elapsed = time.time() - process_stl_start_time
    specimen_name = os.path.splitext(os.path.basename(stl_path))[0]
    print(
        f"FINAL_SUMMARY: specimen={specimen_name} approach=loop_fill "
        f"vertices={integrity['vertex_count']} faces={integrity['face_count']} "
        f"simple_holes={integrity['simple_holes']} boundary_edges={integrity['boundary_edges']} "
        f"non_manifold_edges={integrity['non_manifold_edges']} "
        f"non_manifold_verts={integrity['non_manifold_verts']} "
        f"is_watertight={integrity['is_watertight']} signed_volume={integrity['signed_volume']} "
        f"n_zero_length_edges={degeneracy['n_zero_length_edges']} "
        f"n_zero_area_faces={degeneracy['n_zero_area_faces']} "
        f"n_duplicate_verts={degeneracy['n_duplicate_verts']} "
        f"face_size_cov={face_size_cov} mesh_smoothness={mesh_smoothness} "
        f"time_sec={process_stl_elapsed:.1f}"
    )

    # Determine the output directory
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
        json_data["processed_mesh_integrity"] = integrity
        json_data["processed_mesh_degeneracy"] = degeneracy

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print("JSON file updated successfully.")
    else:
        print(f"Warning: Corresponding JSON file not found at {json_path}")

    return (remaining_vertices, hole_count, face_size_cov, mesh_smoothness)


def _clear_scene_mesh_objects():
    """Deletes every mesh object currently in the scene, so process_stl can be called more than
    once in the same Blender session (e.g. batch-processing a directory) without leftover
    objects from a previous specimen interfering."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            obj.select_set(True)
    bpy.ops.object.delete()


def process_directory(input_dir, output_dir, **kwargs):
    """
    Batch entry point: runs process_stl on every .stl file directly inside input_dir. Clears the
    scene between specimens (via _clear_scene_mesh_objects) so this is safe to run over an
    entire directory in one Blender session rather than one subprocess per file.

    A RuntimeError from any single specimen (e.g. apply_modifiers' Weld-face-loss guard, or
    decimate_mesh's zero-progress guard) is caught, logged, and the batch continues with the
    next specimen - a bad or unusually damaged raw scan should not abort processing for every
    specimen after it in the directory. Failures are collected and reported at the end.

    Args:
        input_dir (str): Directory containing .stl files to process.
        output_dir (str): Directory to write processed .obj files (and updated .json sidecars) to.
        **kwargs: Forwarded to process_stl (max_vertices, ray_density, secondary_rays,
                 random_seed, min_island_faces, keep_rings, min_bridge_island_faces,
                 max_bridge_hops).

    Returns:
        dict: {"succeeded": [specimen_name, ...], "failed": [(specimen_name, error_str), ...]}.
    """
    stl_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".stl"))
    print(f"process_directory: found {len(stl_files)} .stl files in {input_dir}")

    succeeded = []
    failed = []
    for i, filename in enumerate(stl_files):
        specimen_name = os.path.splitext(filename)[0]
        stl_path = os.path.join(input_dir, filename)
        print(f"\n{'=' * 70}\n[{i + 1}/{len(stl_files)}] Processing {specimen_name}\n{'=' * 70}")

        _clear_scene_mesh_objects()
        try:
            process_stl(stl_path, output_dir=output_dir, **kwargs)
            succeeded.append(specimen_name)
        except Exception as e:
            print(f"FAILED: {specimen_name}: {e}")
            failed.append((specimen_name, str(e)))

    print(f"\n{'=' * 70}\nprocess_directory summary: {len(succeeded)} succeeded, "
          f"{len(failed)} failed\n{'=' * 70}")
    if failed:
        print("Failed specimens:")
        for name, err in failed:
            print(f"  {name}: {err}")

    return {"succeeded": succeeded, "failed": failed}


SCRIPT_VERSION = "2026-08-merged-safe-baseline-v1"


def main():
    print(f"SCRIPT_VERSION={SCRIPT_VERSION}")
    start_time = time.time()

    if bpy.context.space_data is not None and bpy.context.space_data.type == "TEXT_EDITOR":
        # Running within Blender's text editor - edit these paths for a single-specimen test.
        stl_path = bpy.path.abspath(
            "/home/fabi/dev/SMILify/custom_processing/antscan_data/Acanthomyrmex_glabfemoralis_CASENT0744002/Acanthomyrmex_glabfemoralis_CASENT0744002.stl"
        )
        output_dir = "/home/fabi/dev/SMILify/custom_processing/antscan_processed"
        process_stl(stl_path, output_dir=output_dir, max_vertices=50000, ray_density=1000,
                    secondary_rays=10000, random_seed=0, keep_rings=2)
    else:
        # Running as a standalone script:
        #   blender --background --python <this file> -- <input_path_or_dir> <output_dir>
        # If <input_path_or_dir> is a directory, every .stl file inside it is processed
        # (process_directory); if it's a single .stl file, only that specimen is processed.
        if len(sys.argv) < 3:
            print(
                "Usage: blender --background --python prepare_antscan_data_for_mesh_fitting_merged.py "
                "-- <input_stl_path_or_directory> <output_dir>"
            )
            sys.exit(1)
        input_path = sys.argv[-2]
        output_dir = sys.argv[-1]

        common_kwargs = dict(
            max_vertices=50000, ray_density=1000, secondary_rays=10000, random_seed=0, keep_rings=2,
        )

        if os.path.isdir(input_path):
            process_directory(input_path, output_dir, **common_kwargs)
        else:
            process_stl(input_path, output_dir=output_dir, **common_kwargs)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Total processing time: {processing_time:.2f} seconds")


if __name__ == "__main__":
    main()