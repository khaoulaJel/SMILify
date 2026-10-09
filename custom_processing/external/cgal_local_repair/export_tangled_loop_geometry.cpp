// Diagnostic-only export (2026-08-17) for Experiment B (tangled-loop decomposition analysis).
// For every boundary loop flagged loop_self_intersects=True by boundary_loop_analysis.cpp
// (recomputed identically here), dumps its ordered 3D vertex coordinates and the full list of
// self-intersecting segment index pairs, so the decomposition-into-candidate-simple-cycles
// analysis can be done in Python. Does NOT triangulate, repair, or modify anything.
//
// Usage: export_tangled_loop_geometry <input.obj> <verts_csv_out> <pairs_csv_out>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Bbox_3.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <algorithm>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;

struct Vec3 { double x,y,z;
  Vec3 operator-(const Vec3& o) const { return {x-o.x,y-o.y,z-o.z}; }
  Vec3 operator+(const Vec3& o) const { return {x+o.x,y+o.y,z+o.z}; }
  Vec3 operator*(double s) const { return {x*s,y*s,z*s}; }
  double dot(const Vec3& o) const { return x*o.x+y*o.y+z*o.z; }
  double norm() const { return std::sqrt(dot(*this)); }
};
static Vec3 P(const Point_3& p) { return {CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())}; }

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
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <verts_csv_out> <pairs_csv_out>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string verts_csv_path = argv[2];
  const std::string pairs_csv_path = argv[3];

  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  if (!CGAL::IO::read_polygon_soup(input_path, points, polygons) || polygons.empty()) {
    std::cerr << "Invalid input: " << input_path << std::endl;
    return EXIT_FAILURE;
  }
  PMP::orient_polygon_soup(points, polygons);
  Mesh mesh;
  PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);

  CGAL::Bbox_3 mesh_bbox;
  for (vertex_descriptor v : vertices(mesh)) mesh_bbox += mesh.point(v).bbox();
  const double mesh_diag = std::sqrt(
      std::pow(mesh_bbox.xmax()-mesh_bbox.xmin(),2) +
      std::pow(mesh_bbox.ymax()-mesh_bbox.ymin(),2) +
      std::pow(mesh_bbox.zmax()-mesh_bbox.zmin(),2));
  const double self_int_eps = std::max(1e-6, 1e-9 * mesh_diag);

  std::vector<std::vector<vertex_descriptor>> loops;
  {
    std::unordered_set<std::size_t> visited;
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (!CGAL::is_border(h, mesh)) continue;
      std::size_t hid = static_cast<std::size_t>(h);
      if (visited.count(hid)) continue;
      std::vector<vertex_descriptor> lv;
      halfedge_descriptor start = h, cur = h;
      do { visited.insert(static_cast<std::size_t>(cur)); lv.push_back(target(cur, mesh)); cur = next(cur, mesh); } while (cur != start);
      loops.push_back(lv);
    }
  }
  std::cout << "N_BOUNDARY_LOOPS_FOUND: " << loops.size() << std::endl;

  std::ofstream verts_csv(verts_csv_path);
  verts_csv << "loop_id,idx_in_loop,x,y,z\n";
  std::ofstream pairs_csv(pairs_csv_path);
  pairs_csv << "loop_id,i,j,distance\n";

  std::size_t n_tangled = 0;
  for (std::size_t lid = 0; lid < loops.size(); ++lid) {
    const auto& lv = loops[lid];
    const std::size_t n = lv.size();
    std::vector<Vec3> pts(n);
    for (std::size_t i = 0; i < n; ++i) pts[i] = P(mesh.point(lv[i]));

    std::vector<std::tuple<std::size_t,std::size_t,double>> self_pairs;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        if (j == i + 1 || (i == 0 && j == n - 1)) continue;
        double d = segment_segment_distance(pts[i], pts[(i+1)%n], pts[j], pts[(j+1)%n]);
        if (d < self_int_eps) self_pairs.emplace_back(i, j, d);
      }
    }
    if (self_pairs.empty()) continue; // only export tangled loops
    ++n_tangled;

    for (std::size_t i = 0; i < n; ++i)
      verts_csv << lid << "," << i << "," << pts[i].x << "," << pts[i].y << "," << pts[i].z << "\n";
    for (const auto& t : self_pairs)
      pairs_csv << lid << "," << std::get<0>(t) << "," << std::get<1>(t) << "," << std::get<2>(t) << "\n";
  }
  verts_csv.close();
  pairs_csv.close();
  std::cout << "N_TANGLED_LOOPS_EXPORTED: " << n_tangled << std::endl;
  std::cout << "Wrote " << verts_csv_path << ", " << pairs_csv_path << std::endl;
  return EXIT_SUCCESS;
}
