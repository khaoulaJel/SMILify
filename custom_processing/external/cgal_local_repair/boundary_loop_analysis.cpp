// Diagnostic-only follow-up (2026-08-17) to patch_intersection_analysis.cpp. Purely diagnostic:
// does NOT repair, weld, bridge, smooth, or run Alpha Wrap. Purpose: characterize every boundary
// loop's PRE-TRIANGULATION geometry (before triangulate_hole ever touches it), then run the exact
// same deterministic fill used by patch_intersection_analysis.cpp so each loop's outcome (rejected
// / patch_id) can be joined in Python against that tool's patches.csv / pairs.csv (patch_id
// assignment is deterministic and identical given the same input file and same loop-enumeration
// order - both are reproduced verbatim here).
//
// Does NOT attempt repair of any kind. Only measures.
//
// Usage: boundary_loop_analysis <input.obj> <loops_csv_out>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Bbox_3.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <sstream>
#include <cmath>
#include <limits>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Vector_3 = K::Vector_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;

using Primitive = CGAL::AABB_face_graph_triangle_primitive<Mesh>;
using Traits = CGAL::AABB_traits_3<K, Primitive>;
using Tree = CGAL::AABB_tree<Traits>;

struct Vec3 { double x, y, z;
  Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
  Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
  Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
  double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
  double norm() const { return std::sqrt(dot(*this)); }
};
static Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static Vec3 P(const Point_3& p) { return {CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())}; }

// Standard closest-distance-between-two-3D-segments (Ericson, "Real-Time Collision Detection").
static double segment_segment_distance(const Vec3& p1, const Vec3& q1, const Vec3& p2, const Vec3& q2)
{
  Vec3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
  double a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
  double s, t;
  const double eps = 1e-15;
  if (a <= eps && e <= eps) { s = t = 0.0; }
  else if (a <= eps) { s = 0.0; t = std::min(1.0, std::max(0.0, f / e)); }
  else {
    double c = d1.dot(r);
    if (e <= eps) { t = 0.0; s = std::min(1.0, std::max(0.0, -c / a)); }
    else {
      double b = d1.dot(d2);
      double denom = a * e - b * b;
      if (denom != 0.0) s = std::min(1.0, std::max(0.0, (b * f - c * e) / denom));
      else s = 0.0;
      t = (b * s + f) / e;
      if (t < 0.0) { t = 0.0; s = std::min(1.0, std::max(0.0, -c / a)); }
      else if (t > 1.0) { t = 1.0; s = std::min(1.0, std::max(0.0, (b - c) / a)); }
    }
  }
  Vec3 c1 = p1 + d1 * s, c2 = p2 + d2 * t;
  return (c1 - c2).norm();
}

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <loops_csv_out>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string loops_csv_path = argv[2];

  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  if (!CGAL::IO::read_polygon_soup(input_path, points, polygons) || polygons.empty()) {
    std::cerr << "Invalid input: " << input_path << std::endl;
    return EXIT_FAILURE;
  }
  PMP::orient_polygon_soup(points, polygons);
  Mesh mesh;
  PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);
  std::cout << "MESH_AFTER_CONVERSION: n_vertices=" << num_vertices(mesh)
            << " n_faces=" << num_faces(mesh) << std::endl;

  CGAL::Bbox_3 mesh_bbox;
  for (vertex_descriptor v : vertices(mesh)) mesh_bbox += mesh.point(v).bbox();
  const double mesh_diag = std::sqrt(
      std::pow(mesh_bbox.xmax()-mesh_bbox.xmin(),2) +
      std::pow(mesh_bbox.ymax()-mesh_bbox.ymin(),2) +
      std::pow(mesh_bbox.zmax()-mesh_bbox.zmin(),2));
  std::cout << "MESH_DIAG: " << mesh_diag << std::endl;

  typedef boost::graph_traits<Mesh>::faces_size_type faces_size_type;
  Mesh::Property_map<face_descriptor, faces_size_type> orig_component =
      mesh.add_property_map<face_descriptor, faces_size_type>("f:orig_cc", 0).first;
  std::size_t n_orig_components = PMP::connected_components(mesh, orig_component);
  std::cout << "ORIGINAL_COMPONENTS: n=" << n_orig_components << std::endl;

  // Global AABB tree over ALL original faces (pre-fill), for loop-to-surface clearance queries.
  std::vector<face_descriptor> all_faces_vec(faces(mesh).begin(), faces(mesh).end());
  Tree global_tree(all_faces_vec.begin(), all_faces_vec.end(), mesh);
  global_tree.accelerate_distance_queries();

  // --- Enumerate loops (IDENTICAL order/logic to patch_intersection_analysis.cpp) and compute
  // pre-fill geometric features per loop in the same pass (vertex positions never change until
  // the later fill pass, so this is safe to compute now). ---
  struct LoopGeom {
    halfedge_descriptor rep;
    std::size_t length;
    faces_size_type home_component;
    std::vector<vertex_descriptor> loop_verts; // ordered
    double perimeter;
    double bbox_dx, bbox_dy, bbox_dz, bbox_diag;
    double max_planarity_dev, rms_planarity_dev;
    bool loop_self_intersects;
    std::size_t n_loop_self_intersecting_segment_pairs;
    double min_dist_to_nonadjacent_surface;
    std::size_t n_other_components_projected_inside;
    std::string other_components_inside_ids;
    bool is_large_boundary;
    double diag_ratio;
  };
  std::vector<LoopGeom> loops;
  {
    std::unordered_set<std::size_t> visited;
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (!CGAL::is_border(h, mesh)) continue;
      std::size_t hid = static_cast<std::size_t>(h);
      if (visited.count(hid)) continue;

      std::vector<vertex_descriptor> loop_verts;
      halfedge_descriptor start = h;
      halfedge_descriptor cur = h;
      do {
        visited.insert(static_cast<std::size_t>(cur));
        loop_verts.push_back(target(cur, mesh)); // NOTE: target(border halfedge) walks the loop
        cur = next(cur, mesh);
      } while (cur != start);

      halfedge_descriptor opp = opposite(h, mesh);
      face_descriptor adj_f = face(opp, mesh);
      faces_size_type home_cc = (adj_f != Mesh::null_face()) ? get(orig_component, adj_f) : static_cast<faces_size_type>(-1);

      LoopGeom lg;
      lg.rep = h;
      lg.length = loop_verts.size();
      lg.home_component = home_cc;
      lg.loop_verts = loop_verts;
      loops.push_back(lg);
    }
  }
  std::cout << "N_BOUNDARY_LOOPS_FOUND: " << loops.size() << std::endl;

  for (LoopGeom& lg : loops) {
    const std::size_t n = lg.length;
    std::vector<Vec3> pts;
    pts.reserve(n);
    for (vertex_descriptor v : lg.loop_verts) pts.push_back(P(mesh.point(v)));

    // perimeter
    double perim = 0.0;
    for (std::size_t i = 0; i < n; ++i) perim += (pts[(i+1)%n] - pts[i]).norm();
    lg.perimeter = perim;

    // bbox
    double bmin[3] = {1e300,1e300,1e300}, bmax[3] = {-1e300,-1e300,-1e300};
    for (const Vec3& p : pts) {
      double c[3] = {p.x, p.y, p.z};
      for (int i=0;i<3;++i) { bmin[i]=std::min(bmin[i],c[i]); bmax[i]=std::max(bmax[i],c[i]); }
    }
    lg.bbox_dx = bmax[0]-bmin[0]; lg.bbox_dy = bmax[1]-bmin[1]; lg.bbox_dz = bmax[2]-bmin[2];
    lg.bbox_diag = std::sqrt(lg.bbox_dx*lg.bbox_dx + lg.bbox_dy*lg.bbox_dy + lg.bbox_dz*lg.bbox_dz);
    lg.diag_ratio = mesh_diag > 0 ? lg.bbox_diag / mesh_diag : 0.0;
    lg.is_large_boundary = (lg.diag_ratio > 0.3);

    // Newell-method area-weighted normal + centroid -> best-fit plane (well-defined for
    // non-planar/non-convex closed polygons; standard technique, not a least-squares PCA fit).
    Vec3 centroid{0,0,0};
    for (const Vec3& p : pts) centroid = centroid + p;
    centroid = centroid * (1.0 / n);
    Vec3 normal{0,0,0};
    for (std::size_t i = 0; i < n; ++i) {
      const Vec3& a = pts[i]; const Vec3& b = pts[(i+1)%n];
      normal.x += (a.y - b.y) * (a.z + b.z);
      normal.y += (a.z - b.z) * (a.x + b.x);
      normal.z += (a.x - b.x) * (a.y + b.y);
    }
    double nnorm = normal.norm();
    if (nnorm > 1e-12) normal = normal * (1.0 / nnorm);
    else normal = {0,0,1}; // degenerate (near-collinear loop); arbitrary axis, deviations reported as 0

    double sum_sq_dev = 0.0, max_dev = 0.0;
    for (const Vec3& p : pts) {
      double dev = std::fabs((p - centroid).dot(normal));
      sum_sq_dev += dev * dev;
      if (dev > max_dev) max_dev = dev;
    }
    lg.max_planarity_dev = max_dev;
    lg.rms_planarity_dev = std::sqrt(sum_sq_dev / n);

    // loop self-intersection: pairwise non-adjacent segment distance test (O(n^2), fine at this scale)
    const double self_int_eps = std::max(1e-6, 1e-9 * mesh_diag);
    std::size_t n_self_int_pairs = 0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        // skip adjacent segments (share an endpoint) - they legitimately touch at that shared vertex
        if (j == i + 1 || (i == 0 && j == n - 1)) continue;
        double d = segment_segment_distance(pts[i], pts[(i+1)%n], pts[j], pts[(j+1)%n]);
        if (d < self_int_eps) ++n_self_int_pairs;
      }
    }
    lg.n_loop_self_intersecting_segment_pairs = n_self_int_pairs;
    lg.loop_self_intersects = (n_self_int_pairs > 0);

    // min distance from loop vertices to non-adjacent existing surface (excludes faces sharing a
    // loop vertex - those are the trivially-adjacent home faces, not informative "nearby anatomy").
    std::set<vertex_descriptor> loop_vert_set(lg.loop_verts.begin(), lg.loop_verts.end());
    double min_clear = std::numeric_limits<double>::infinity();
    for (const Vec3& p : pts) {
      Point_3 qp(p.x, p.y, p.z);
      Tree::Point_and_primitive_id pp = global_tree.closest_point_and_primitive(qp);
      face_descriptor closest_f = pp.second;
      bool adjacent = false;
      for (vertex_descriptor v : vertices_around_face(halfedge(closest_f, mesh), mesh))
        if (loop_vert_set.count(v)) { adjacent = true; break; }
      if (!adjacent) {
        double d = std::sqrt(CGAL::to_double(CGAL::squared_distance(qp, pp.first)));
        min_clear = std::min(min_clear, d);
      }
    }
    lg.min_dist_to_nonadjacent_surface = min_clear;

    // components projected "inside" the loop footprint: project other-component face centroids
    // onto the loop's best-fit plane (centroid/normal above), point-in-polygon test (ray casting
    // in the plane's local 2D basis) against the loop's own projected boundary, restricted to
    // centroids within one bbox_diag of the plane (both in-plane extent and out-of-plane distance)
    // so this stays a LOCAL test, not a whole-mesh one.
    Vec3 u_axis = (std::fabs(normal.x) < 0.9) ? cross(normal, {1,0,0}) : cross(normal, {0,1,0});
    { double un = u_axis.norm(); u_axis = un > 1e-12 ? u_axis * (1.0/un) : Vec3{1,0,0}; }
    Vec3 v_axis = cross(normal, u_axis);

    std::vector<std::pair<double,double>> poly2d;
    poly2d.reserve(n);
    for (const Vec3& p : pts) {
      Vec3 rel = p - centroid;
      poly2d.emplace_back(rel.dot(u_axis), rel.dot(v_axis));
    }
    auto point_in_poly = [&](double px, double py) -> bool {
      bool inside = false;
      for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        double xi = poly2d[i].first, yi = poly2d[i].second;
        double xj = poly2d[j].first, yj = poly2d[j].second;
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi))
          inside = !inside;
      }
      return inside;
    };

    std::set<faces_size_type> comps_inside;
    const double search_radius = std::max(lg.bbox_diag, 1.0);
    for (face_descriptor f : faces(mesh)) {
      faces_size_type cc = get(orig_component, f);
      if (cc == lg.home_component) continue; // only interested in OTHER components
      Vec3 fc{0,0,0}; int nv=0;
      for (vertex_descriptor v : vertices_around_face(halfedge(f, mesh), mesh)) { fc = fc + P(mesh.point(v)); ++nv; }
      fc = fc * (1.0/nv);
      Vec3 rel = fc - centroid;
      double w = rel.dot(normal);
      if (std::fabs(w) > search_radius) continue; // out-of-plane range gate (local test)
      double u = rel.dot(u_axis), v = rel.dot(v_axis);
      if (std::fabs(u) > search_radius || std::fabs(v) > search_radius) continue; // in-plane range gate
      if (point_in_poly(u, v)) comps_inside.insert(cc);
    }
    lg.n_other_components_projected_inside = comps_inside.size();
    std::ostringstream oss;
    bool first = true;
    for (faces_size_type cc : comps_inside) { if (!first) oss << ";"; oss << cc; first = false; }
    lg.other_components_inside_ids = oss.str();
  }

  // --- Now perform the IDENTICAL deterministic fill as patch_intersection_analysis.cpp, to get
  // outcome (patch_id or rejection reason) per loop, in the same order. ---
  std::vector<int> outcome_patch_id(loops.size(), -1);
  std::vector<std::string> outcome_status(loops.size(), "");
  int next_patch_id = 0;
  for (std::size_t i = 0; i < loops.size(); ++i) {
    const LoopGeom& lg = loops[i];
    if (lg.length < 3) { outcome_status[i] = "REJECTED_LEN_LT3"; continue; }
    if (!CGAL::is_border(lg.rep, mesh)) { outcome_status[i] = "REJECTED_NOT_BORDER_ANYMORE"; continue; }
    std::vector<face_descriptor> patch_faces;
    try {
      PMP::triangulate_hole(mesh, lg.rep, CGAL::parameters::face_output_iterator(std::back_inserter(patch_faces)));
    } catch (...) {
      outcome_status[i] = "EXCEPTION";
      continue;
    }
    if (patch_faces.empty()) { outcome_status[i] = "REJECTED_EMPTY_RESULT"; continue; }
    outcome_status[i] = "SUCCESS";
    outcome_patch_id[i] = next_patch_id++;
  }
  std::cout << "N_PATCHES_CREATED_BY_FILL_PASS: " << next_patch_id << std::endl;

  std::ofstream csv(loops_csv_path);
  csv << "loop_id,patch_id,outcome,length,home_component,perimeter,bbox_dx,bbox_dy,bbox_dz,bbox_diag,"
         "diag_ratio_to_mesh,is_large_boundary,max_planarity_dev,rms_planarity_dev,"
         "loop_self_intersects,n_loop_self_intersecting_segment_pairs,"
         "min_dist_to_nonadjacent_surface,n_other_components_projected_inside,other_components_inside_ids\n";
  for (std::size_t i = 0; i < loops.size(); ++i) {
    const LoopGeom& lg = loops[i];
    csv << i << "," << outcome_patch_id[i] << "," << outcome_status[i] << "," << lg.length << ","
        << lg.home_component << "," << lg.perimeter << ","
        << lg.bbox_dx << "," << lg.bbox_dy << "," << lg.bbox_dz << "," << lg.bbox_diag << ","
        << lg.diag_ratio << "," << (lg.is_large_boundary ? "True" : "False") << ","
        << lg.max_planarity_dev << "," << lg.rms_planarity_dev << ","
        << (lg.loop_self_intersects ? "True" : "False") << "," << lg.n_loop_self_intersecting_segment_pairs << ","
        << (std::isinf(lg.min_dist_to_nonadjacent_surface) ? std::string("inf") : std::to_string(lg.min_dist_to_nonadjacent_surface)) << ","
        << lg.n_other_components_projected_inside << "," << lg.other_components_inside_ids
        << "\n";
  }
  csv.close();
  std::cout << "Wrote " << loops_csv_path << std::endl;
  return EXIT_SUCCESS;
}
