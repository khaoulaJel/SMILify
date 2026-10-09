// Priority 2 Step 4 (2026-08-17): sheet-guided boundary split diagnosis for a single tangled loop.
// Assigns each boundary vertex to sheet A/B via its OWN incident home-face normals (2-means on the
// loop's home faces, then per-vertex majority/nearest-centroid assignment), finds transition points
// along the loop, and reports whether the resulting split is topologically coherent (exactly 2
// transitions = clean bipartition into 2 arcs) BEFORE proposing any edge to add. Diagnostic only -
// does not modify the mesh. A companion tool (sheet_split_experiment.cpp) performs the actual
// mesh-copy split once this diagnostic confirms coherence.
//
// Usage: sheet_split_diagnosis <input.obj> <loop_id> <out_csv>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <cmath>
#include <random>

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
  Vec3 operator+(const Vec3& o) const { return {x+o.x,y+o.y,z+o.z}; }
  Vec3 operator*(double s) const { return {x*s,y*s,z*s}; }
  double dot(const Vec3& o) const { return x*o.x+y*o.y+z*o.z; }
  double norm() const { return std::sqrt(dot(*this)); }
};
static Vec3 P(const Point_3& p) { return {CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())}; }
static Vec3 cross(const Vec3&a,const Vec3&b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <loop_id> <out_csv>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::size_t target_loop_id = std::stoul(argv[2]);
  const std::string csv_path = argv[3];

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

  std::vector<halfedge_descriptor> loop_reps;
  { std::unordered_set<std::size_t> visited;
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (!CGAL::is_border(h, mesh)) continue;
      std::size_t hid = static_cast<std::size_t>(h);
      if (visited.count(hid)) continue;
      loop_reps.push_back(h);
      halfedge_descriptor start = h, cur = h;
      do { visited.insert(static_cast<std::size_t>(cur)); cur = next(cur, mesh); } while (cur != start);
    }
  }
  if (target_loop_id >= loop_reps.size()) { std::cerr << "loop_id out of range" << std::endl; return EXIT_FAILURE; }

  std::vector<vertex_descriptor> loop_verts;
  { halfedge_descriptor start = loop_reps[target_loop_id], cur = start;
    do { loop_verts.push_back(target(cur, mesh)); cur = next(cur, mesh); } while (cur != start);
  }
  const std::size_t n = loop_verts.size();
  std::cout << "LOOP_SIZE: " << n << std::endl;
  std::set<vertex_descriptor> loop_set(loop_verts.begin(), loop_verts.end());

  // collect home faces + normals
  std::vector<face_descriptor> home_faces;
  { std::set<face_descriptor> s;
    for (vertex_descriptor v : loop_verts)
      for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh))
        if (f != Mesh::null_face()) s.insert(f);
    home_faces.assign(s.begin(), s.end());
  }
  auto face_normal = [&](face_descriptor f) -> Vec3 {
    std::vector<Vec3> pts;
    for (vertex_descriptor v : vertices_around_face(halfedge(f, mesh), mesh)) pts.push_back(P(mesh.point(v)));
    Vec3 nrm = cross(pts[1]-pts[0], pts[2]-pts[0]);
    double l = nrm.norm();
    return l > 1e-12 ? nrm * (1.0/l) : Vec3{0,0,0};
  };
  std::vector<Vec3> home_normals;
  for (face_descriptor f : home_faces) home_normals.push_back(face_normal(f));

  // 2-means on home_normals (deterministic seed, few iterations - small N, converges fast)
  Vec3 c0 = home_normals[0], c1 = home_normals[home_normals.size()/2];
  for (int iter = 0; iter < 50; ++iter) {
    Vec3 s0{0,0,0}, s1{0,0,0}; int n0=0, n1=0;
    for (auto& nrm : home_normals) {
      if (nrm.dot(c0) >= nrm.dot(c1)) { s0 = s0 + nrm; ++n0; } else { s1 = s1 + nrm; ++n1; }
    }
    if (n0>0) c0 = s0 * (1.0/n0); if (n1>0) c1 = s1 * (1.0/n1);
    double l0=c0.norm(), l1=c1.norm();
    if (l0>1e-9) c0 = c0*(1.0/l0); if (l1>1e-9) c1 = c1*(1.0/l1);
  }
  double cluster_angle = std::acos(std::max(-1.0,std::min(1.0,c0.dot(c1)))) * 180.0/M_PI;
  std::cout << "NORMAL_CLUSTER_ANGLE: " << cluster_angle << " deg" << std::endl;

  // per-vertex sheet assignment: average normal of ITS OWN incident home faces, nearest-centroid
  std::vector<int> sheet(n, -1);
  std::vector<double> confidence(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    vertex_descriptor v = loop_verts[i];
    Vec3 avg{0,0,0}; int cnt=0;
    for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh))
      if (f != Mesh::null_face()) { avg = avg + face_normal(f); ++cnt; }
    if (cnt) avg = avg * (1.0/cnt);
    double d0 = avg.dot(c0), d1 = avg.dot(c1);
    sheet[i] = (d0 >= d1) ? 0 : 1;
    confidence[i] = std::fabs(d0 - d1);
  }

  // find transitions
  std::vector<std::size_t> transitions;
  for (std::size_t i = 0; i < n; ++i)
    if (sheet[i] != sheet[(i+1)%n]) transitions.push_back(i); // transition occurs between i and i+1
  std::cout << "N_TRANSITIONS: " << transitions.size() << std::endl;

  std::ofstream csv(csv_path);
  csv << "idx,v,x,y,z,sheet,confidence,is_transition_start\n";
  std::set<std::size_t> trans_set(transitions.begin(), transitions.end());
  for (std::size_t i = 0; i < n; ++i) {
    const Point_3& p = mesh.point(loop_verts[i]);
    csv << i << "," << loop_verts[i] << "," << CGAL::to_double(p.x()) << "," << CGAL::to_double(p.y()) << "," << CGAL::to_double(p.z()) << ","
        << sheet[i] << "," << confidence[i] << "," << (trans_set.count(i)?"True":"False") << "\n";
  }
  csv.close();

  // summarize runs
  std::cout << "TRANSITIONS_AT_IDX:";
  for (auto t : transitions) std::cout << " " << t << "(v" << loop_verts[t] << "->v" << loop_verts[(t+1)%n] << ")";
  std::cout << std::endl;
  std::size_t n_sheet0 = 0; for (int s : sheet) if (s==0) ++n_sheet0;
  std::cout << "SHEET_COUNTS: sheet0=" << n_sheet0 << " sheet1=" << (n-n_sheet0) << std::endl;
  std::cout << "Wrote " << csv_path << std::endl;
  return EXIT_SUCCESS;
}
