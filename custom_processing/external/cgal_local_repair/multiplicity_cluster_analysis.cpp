// Priority 1 (2026-08-17): characterize every >2-way exact-duplicate-position vertex cluster on
// the PRISTINE mesh (read+orient_polygon_soup+polygon_soup_to_polygon_mesh only - matches exactly
// what duplicate_seam_characterization.cpp and explicit_A_only_stitch.cpp operate on; NOT the
// post-triangulate_hole analysis_mesh.off used by earlier patch-intersection tooling, which would
// misreport border status since those holes are already filled there).
//
// For each cluster member: border status, predecessor/successor along its border chain (if any),
// degree, average face normal, incident edge lengths, connected component, one-ring neighbor
// positions (for cross-cluster chain-adjacency checks done in Python afterward).
//
// Diagnostic only. No repair.
//
// Usage: multiplicity_cluster_analysis <input.obj> <clusters_csv_out>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/measure.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cmath>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;
typedef boost::graph_traits<Mesh>::faces_size_type faces_size_type;

struct Vec3 { double x,y,z;
  Vec3 operator-(const Vec3& o) const { return {x-o.x,y-o.y,z-o.z}; }
  double norm() const { return std::sqrt(x*x+y*y+z*z); }
};
static Vec3 P(const Point_3& p) { return {CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())}; }

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <clusters_csv_out>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string csv_path = argv[2];

  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  if (!CGAL::IO::read_polygon_soup(input_path, points, polygons) || polygons.empty()) {
    std::cerr << "Invalid input: " << input_path << std::endl;
    return EXIT_FAILURE;
  }
  PMP::orient_polygon_soup(points, polygons);
  Mesh mesh;
  PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);
  std::cout << "MESH_AFTER_CONVERSION: n_vertices=" << num_vertices(mesh) << " n_faces=" << num_faces(mesh) << std::endl;

  Mesh::Property_map<face_descriptor, faces_size_type> orig_component =
      mesh.add_property_map<face_descriptor, faces_size_type>("f:orig_cc", 0).first;
  std::size_t n_components = PMP::connected_components(mesh, orig_component);
  std::cout << "N_COMPONENTS: " << n_components << std::endl;

  // group vertices by exact position
  std::map<std::array<double,3>, std::vector<vertex_descriptor>> pos_clusters;
  for (vertex_descriptor v : vertices(mesh)) {
    Vec3 p = P(mesh.point(v));
    pos_clusters[{p.x, p.y, p.z}].push_back(v);
  }

  auto vertex_component = [&](vertex_descriptor v) -> faces_size_type {
    for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh))
      if (f != Mesh::null_face()) return get(orig_component, f);
    return static_cast<faces_size_type>(-1);
  };
  auto is_border_vertex = [&](vertex_descriptor v) -> halfedge_descriptor {
    // returns the unique incoming border halfedge, or null if none / not exactly one
    halfedge_descriptor found = Mesh::null_halfedge(); int count = 0;
    for (halfedge_descriptor h : halfedges_around_target(halfedge(v, mesh), mesh))
      if (CGAL::is_border(h, mesh)) { found = h; ++count; }
    return (count == 1) ? found : Mesh::null_halfedge();
  };
  auto avg_normal = [&](vertex_descriptor v) -> Vec3 {
    Vec3 n{0,0,0}; int cnt=0;
    for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh)) {
      if (f == Mesh::null_face()) continue;
      std::vector<Vec3> pts;
      for (vertex_descriptor fv : vertices_around_face(halfedge(f, mesh), mesh)) pts.push_back(P(mesh.point(fv)));
      if (pts.size() == 3) {
        Vec3 e1 = pts[1]-pts[0], e2 = pts[2]-pts[0];
        Vec3 c{e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z, e1.x*e2.y-e1.y*e2.x};
        double l = c.norm();
        if (l > 1e-12) { n.x += c.x/l; n.y += c.y/l; n.z += c.z/l; ++cnt; }
      }
    }
    if (cnt) { double l = n.norm(); if (l>1e-12) { n.x/=l; n.y/=l; n.z/=l; } }
    return n;
  };

  std::ofstream csv(csv_path);
  csv << "cluster_id,cluster_size,v,degree,component,is_border,pred_v,succ_v,pred_dist,succ_dist,"
         "normal_x,normal_y,normal_z,one_ring_positions\n";

  int cluster_id = 0;
  std::size_t n_clusters = 0;
  for (auto& kv : pos_clusters) {
    if (kv.second.size() <= 2) continue;
    ++n_clusters;
    for (vertex_descriptor v : kv.second) {
      int deg = 0;
      for (halfedge_descriptor h : halfedges_around_target(halfedge(v, mesh), mesh)) ++deg;
      faces_size_type cc = vertex_component(v);
      halfedge_descriptor bh = is_border_vertex(v);
      bool border = (bh != Mesh::null_halfedge());
      long pred_v = -1, succ_v = -1;
      double pred_dist = -1, succ_dist = -1;
      if (border) {
        vertex_descriptor pred = source(bh, mesh);
        vertex_descriptor succ = target(next(bh, mesh), mesh);
        pred_v = static_cast<long>(static_cast<std::uint32_t>(pred));
        succ_v = static_cast<long>(static_cast<std::uint32_t>(succ));
        pred_dist = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(v), mesh.point(pred))));
        succ_dist = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(v), mesh.point(succ))));
      }
      Vec3 n = avg_normal(v);
      std::ostringstream ring;
      bool first = true;
      for (vertex_descriptor nb : vertices_around_target(halfedge(v, mesh), mesh)) {
        if (!first) ring << ";"; first = false;
        Vec3 p = P(mesh.point(nb));
        ring << p.x << " " << p.y << " " << p.z;
      }
      csv << cluster_id << "," << kv.second.size() << "," << v << "," << deg << "," << cc << ","
          << (border?"True":"False") << "," << pred_v << "," << succ_v << ","
          << pred_dist << "," << succ_dist << ","
          << n.x << "," << n.y << "," << n.z << ","
          << "\"" << ring.str() << "\"\n";
    }
    ++cluster_id;
  }
  csv.close();
  std::cout << "N_MULTIWAY_CLUSTERS(>2): " << n_clusters << std::endl;
  std::cout << "Wrote " << csv_path << std::endl;
  return EXIT_SUCCESS;
}
