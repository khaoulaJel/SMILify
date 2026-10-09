"""For a given specimen's pre-reconstruction mesh, find the actual sample point that produced the
p5-percentile thickness value, and check the TOPOLOGICAL (mesh-graph hop) distance between that
point and its ray-hit target - not just 3D Euclidean distance. A genuine thin single-tube feature
(e.g. a leg segment) has its "opposite wall" only a few hops away in the mesh graph, even though
it's geometrically close. A crossing/interlacing appendage (two DIFFERENT legs/antennae that
happen to pass near each other in this pose) has its nearest ray-hit on a topologically DISTANT
part of the mesh (many hops away) despite being geometrically close - the same distinguishing
signature _min_self_approach_gap's own hops=4 exclusion is designed around, applied here to
diagnose whether the SDF p5 metric is measuring true material thickness or an air-gap between
crossing parts.

Usage: python3 probe_p5_crossing_appendages.py <PRE_RECONSTRUCTION.obj>
"""
import sys
import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def largest_component(mesh):
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda p: len(p.faces))


def analyze(path, n_samples=1500, seed=0):
    mesh = trimesh.load(path, process=False)
    mesh = largest_component(mesh)
    rng = np.random.default_rng(seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_idx]
    origins = points - normals * 1e-4
    directions = -normals
    locations, index_ray, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
    dists = np.linalg.norm(locations - origins[index_ray], axis=1)

    p5_val = np.percentile(dists, 5)
    # find the sample closest to the actual p5 percentile value
    closest_idx = np.argmin(np.abs(dists - p5_val))
    src_point = origins[index_ray[closest_idx]]
    hit_point = locations[closest_idx]
    euclid_dist = dists[closest_idx]

    # nearest mesh vertex to each
    v = mesh.vertices
    ia = int(np.argmin(np.linalg.norm(v - src_point, axis=1)))
    ib = int(np.argmin(np.linalg.norm(v - hit_point, axis=1)))

    # build sparse edge graph from faces, weighted by edge length, BFS/dijkstra hop-ish distance
    edges = mesh.edges_unique
    edge_len = mesh.edges_unique_length
    n = len(v)
    graph = csr_matrix((edge_len, (edges[:, 0], edges[:, 1])), shape=(n, n))
    graph = graph + graph.T
    try:
        d = shortest_path(graph, method="D", directed=False, indices=[ia], unweighted=False)
        geo_dist = float(d[0][ib])
    except Exception as e:
        geo_dist = float("nan")

    ratio = geo_dist / euclid_dist if euclid_dist > 0 and np.isfinite(geo_dist) else float("inf")
    print(f"specimen={path} p5={p5_val:.4g} euclid_dist={euclid_dist:.4g} "
          f"geodesic_dist={geo_dist:.4g} geo/euclid_ratio={ratio:.2f} "
          f"interpretation={'CROSSING-LIKE(geo>>euclid)' if ratio > 20 else 'SAME-LOCAL-SURFACE(geo~euclid)'}")


if __name__ == "__main__":
    analyze(sys.argv[1])
