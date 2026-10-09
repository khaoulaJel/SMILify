// Follow-up (2026-08-17): fully characterize the giant 2-way duplicate border chains associated
// with the 9 SIMPLE + min_dist=0 loops, uncapped (previous chain-match walks were capped at 200
// steps per direction for the classifier; this walks to the true stopping point and records WHY
// it stopped at every step). Diagnostic only - no stitching.
//
// Usage: giant_chain_diagnosis <input.obj> <seed_v1> <seed_v2> <specimen_tag> <steps_csv_out>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
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
static double dot(const Vec3&a,const Vec3&b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static Vec3 cross(const Vec3&a,const Vec3&b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}

// returns (halfedge, multiplicity) - multiplicity = number of incoming border halfedges at v
// (1 = clean/unique, >1 = local multiplicity ambiguity at this specific vertex)
static std::pair<halfedge_descriptor,int> incoming_border_info(const Mesh& mesh, vertex_descriptor v)
{
  halfedge_descriptor found = Mesh::null_halfedge(); int count = 0;
  for (halfedge_descriptor h : halfedges_around_target(halfedge(v, mesh), mesh))
    if (CGAL::is_border(h, mesh)) { found = h; ++count; }
  return {found, count};
}

static Vec3 avg_normal(const Mesh& mesh, vertex_descriptor v)
{
  Vec3 n{0,0,0}; int cnt=0;
  for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh)) {
    if (f == Mesh::null_face()) continue;
    std::vector<Vec3> pts;
    for (vertex_descriptor fv : vertices_around_face(halfedge(f, mesh), mesh)) pts.push_back(P(mesh.point(fv)));
    if (pts.size() == 3) {
      Vec3 e1 = pts[1]-pts[0], e2 = pts[2]-pts[0];
      Vec3 c = cross(e1,e2);
      double l = c.norm();
      if (l > 1e-12) { n.x += c.x/l; n.y += c.y/l; n.z += c.z/l; ++cnt; }
    }
  }
  if (cnt) { double l = n.norm(); if (l>1e-12) { n.x/=l; n.y/=l; n.z/=l; } }
  return n;
}

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <seed_v1> <seed_v2> <specimen_tag> <steps_csv_out>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  std::uint32_t seed_v1 = std::stoul(argv[2]);
  std::uint32_t seed_v2 = std::stoul(argv[3]);
  const std::string specimen_tag = argv[4];
  const std::string csv_path = argv[5];

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
  PMP::connected_components(mesh, orig_component);
  auto vcomp = [&](vertex_descriptor v) -> faces_size_type {
    for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh))
      if (f != Mesh::null_face()) return get(orig_component, f);
    return static_cast<faces_size_type>(-1);
  };

  // identify which native boundary loop each vertex belongs to (loop id = index of first-seen rep)
  std::vector<int> loop_id_of(num_vertices(mesh), -1);
  {
    int lid = 0;
    std::vector<bool> visited(num_vertices(mesh), false);
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (!CGAL::is_border(h, mesh)) continue;
      vertex_descriptor vt = target(h, mesh);
      if (visited[static_cast<std::uint32_t>(vt)]) continue;
      halfedge_descriptor start = h, cur = h;
      do {
        vertex_descriptor tv = target(cur, mesh);
        visited[static_cast<std::uint32_t>(tv)] = true;
        loop_id_of[static_cast<std::uint32_t>(tv)] = lid;
        cur = next(cur, mesh);
      } while (cur != start);
      ++lid;
    }
  }

  vertex_descriptor v1(seed_v1), v2(seed_v2);
  std::cout << "SEED: v1=" << v1 << " (loop=" << loop_id_of[seed_v1] << ") v2=" << v2 << " (loop=" << loop_id_of[seed_v2] << ")" << std::endl;

  std::ofstream csv(csv_path);
  csv << "step,direction,v1,v2,loop1,loop2,comp1,comp2,mult1,mult2,edge_len1,edge_len2,"
         "normal_dot1,normal_dot2,stop_reason\n";

  // direction A: predecessor(side1) vs successor(side2), uncapped
  auto walk = [&](vertex_descriptor start1, vertex_descriptor start2, const std::string& dirlabel, int max_steps) {
    auto [h1_0, m1_0] = incoming_border_info(mesh, start1);
    auto [h2_0, m2_0] = incoming_border_info(mesh, start2);
    if (h1_0 == Mesh::null_halfedge() || h2_0 == Mesh::null_halfedge()) {
      csv << "0," << dirlabel << "," << start1 << "," << start2 << ","
          << loop_id_of[static_cast<std::uint32_t>(start1)] << "," << loop_id_of[static_cast<std::uint32_t>(start2)] << ","
          << vcomp(start1) << "," << vcomp(start2) << "," << m1_0 << "," << m2_0 << ",0,0,0,0,SEED_NOT_UNIQUE_BORDER\n";
      return 0;
    }
    halfedge_descriptor cur1 = h1_0, cur2h = next(h2_0, mesh);
    int step = 0;
    for (; step < max_steps; ++step) {
      vertex_descriptor a = source(cur1, mesh);
      vertex_descriptor b = target(cur2h, mesh);
      double d = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(a), mesh.point(b))));
      std::string stop_reason = "-";
      bool stop = false;
      if (d != 0.0) { stop_reason = "POSITION_MISMATCH(dist=" + std::to_string(d) + ")"; stop = true; }
      auto [ha, ma] = incoming_border_info(mesh, a);
      auto [hb, mb] = incoming_border_info(mesh, b);
      double el1 = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(a), mesh.point(target(cur1,mesh)))));
      double el2 = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(source(cur2h,mesh)), mesh.point(b))));
      Vec3 na = avg_normal(mesh, a), nb = avg_normal(mesh, b);
      // "normal_dot" here = dot of this vertex's normal against the OTHER side's corresponding vertex normal
      Vec3 na_partner = avg_normal(mesh, b), nb_partner = avg_normal(mesh, a);
      double ndot1 = dot(na, na_partner), ndot2 = dot(nb, nb_partner);
      if (!stop) {
        if (ma != 1) { stop_reason = "MULTIPLICITY_AT_A(m=" + std::to_string(ma) + ")"; stop = true; }
        else if (mb != 1) { stop_reason = "MULTIPLICITY_AT_B(m=" + std::to_string(mb) + ")"; stop = true; }
        else if (a == start2 || b == start1) { stop_reason = "WRAPPED_AROUND"; stop = true; }
      }
      csv << step << "," << dirlabel << "," << a << "," << b << ","
          << loop_id_of[static_cast<std::uint32_t>(a)] << "," << loop_id_of[static_cast<std::uint32_t>(b)] << ","
          << vcomp(a) << "," << vcomp(b) << "," << ma << "," << mb << "," << el1 << "," << el2 << ","
          << ndot1 << "," << ndot2 << "," << (stop ? stop_reason : "OK") << "\n";
      if (stop) break;
      if (!CGAL::is_border(prev(cur1, mesh), mesh) || !CGAL::is_border(next(cur2h, mesh), mesh)) {
        csv << (step+1) << "," << dirlabel << "," << a << "," << b << ",,,,,,,,,,,NEXT_STEP_NOT_BORDER\n";
        break;
      }
      cur1 = prev(cur1, mesh); cur2h = next(cur2h, mesh);
    }
    return step;
  };

  // same-direction hypothesis: successor(v1) vs successor(v2), and predecessor(v1) vs predecessor(v2)
  // - distinct from the "reversed" hypothesis above (pred vs succ). Established earlier this
  // session that the giant min_dist=0 chains are "same_direction" matches, not "reversed" ones.
  auto walk_same_dir = [&](vertex_descriptor start1, vertex_descriptor start2, bool forward, const std::string& dirlabel, int max_steps) {
    auto [h1_0, m1_0] = incoming_border_info(mesh, start1);
    auto [h2_0, m2_0] = incoming_border_info(mesh, start2);
    if (h1_0 == Mesh::null_halfedge() || h2_0 == Mesh::null_halfedge()) return 0;
    halfedge_descriptor cur1 = forward ? next(h1_0, mesh) : h1_0;
    halfedge_descriptor cur2 = forward ? next(h2_0, mesh) : h2_0;
    int step = 0;
    for (; step < max_steps; ++step) {
      vertex_descriptor a = forward ? target(cur1, mesh) : source(cur1, mesh);
      vertex_descriptor b = forward ? target(cur2, mesh) : source(cur2, mesh);
      double d = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(a), mesh.point(b))));
      std::string stop_reason = "-"; bool stop = false;
      if (d != 0.0) { stop_reason = "POSITION_MISMATCH(dist=" + std::to_string(d) + ")"; stop = true; }
      auto [ha, ma] = incoming_border_info(mesh, a);
      auto [hb, mb] = incoming_border_info(mesh, b);
      double el1 = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(a), mesh.point(forward?source(cur1,mesh):target(cur1,mesh)))));
      double el2 = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(b), mesh.point(forward?source(cur2,mesh):target(cur2,mesh)))));
      Vec3 na = avg_normal(mesh, a), nb = avg_normal(mesh, b);
      double ndot1 = dot(na, nb), ndot2 = ndot1;
      if (!stop) {
        if (ma != 1) { stop_reason = "MULTIPLICITY_AT_A(m=" + std::to_string(ma) + ")"; stop = true; }
        else if (mb != 1) { stop_reason = "MULTIPLICITY_AT_B(m=" + std::to_string(mb) + ")"; stop = true; }
        else if (a == start2 || b == start1) { stop_reason = "WRAPPED_AROUND_CROSS"; stop = true; }
        else if (a == start1 || b == start2) { stop_reason = "WRAPPED_AROUND_SELF(loop_shorter_than_walk)"; stop = true; }
      }
      csv << step << "," << dirlabel << "," << a << "," << b << ","
          << loop_id_of[static_cast<std::uint32_t>(a)] << "," << loop_id_of[static_cast<std::uint32_t>(b)] << ","
          << vcomp(a) << "," << vcomp(b) << "," << ma << "," << mb << "," << el1 << "," << el2 << ","
          << ndot1 << "," << ndot2 << "," << (stop ? stop_reason : "OK") << "\n";
      if (stop) break;
      halfedge_descriptor n1 = forward ? next(cur1, mesh) : prev(cur1, mesh);
      halfedge_descriptor n2 = forward ? next(cur2, mesh) : prev(cur2, mesh);
      if (!CGAL::is_border(n1, mesh) || !CGAL::is_border(n2, mesh)) {
        csv << (step+1) << "," << dirlabel << "," << a << "," << b << ",,,,,,,,,,,NEXT_STEP_NOT_BORDER\n";
        break;
      }
      cur1 = n1; cur2 = n2;
    }
    return step;
  };

  int lenA = walk(v1, v2, "A_pred1_succ2", 5000);
  int lenB = walk(v2, v1, "B_pred2_succ1", 5000); // symmetric other direction from the seed itself (reuses same seed, opposite role assignment)
  int lenC = walk_same_dir(v1, v2, true, "C_samedir_forward", 300);
  int lenD = walk_same_dir(v1, v2, false, "D_samedir_backward", 300);
  csv.close();
  std::cout << "DIRECTION_A_STEPS: " << lenA << "  DIRECTION_B_STEPS: " << lenB
            << "  DIRECTION_C_STEPS: " << lenC << "  DIRECTION_D_STEPS: " << lenD << std::endl;
  std::cout << "Wrote " << csv_path << std::endl;
  return EXIT_SUCCESS;
}
