// Priority 2, Step 1 (2026-08-17): for a given tangled boundary loop (identified by loop_id, same
// deterministic enumeration as boundary_loop_analysis.cpp), export the geometry needed to assess
// its sheet/topology mechanism: the loop's own vertices, its HOME faces (existing faces adjacent
// to the hole boundary - one ring in from the hole), and a wider local-context ring (faces within
// N hops of the boundary), with face normals and vertex positions, for normal/spatial clustering
// in Python. Diagnostic only.
//
// Usage: tangled_sheet_analysis <input.obj> <loop_id> <out_csv_prefix>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/measure.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <queue>
#include <cmath>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;
typedef boost::graph_traits<Mesh>::faces_size_type faces_size_type;

static double to_d(const K::FT& x) { return CGAL::to_double(x); }

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <loop_id> <out_csv_prefix>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::size_t target_loop_id = std::stoul(argv[2]);
  const std::string out_prefix = argv[3];

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

  // enumerate loops identically to boundary_loop_analysis.cpp
  std::vector<halfedge_descriptor> loop_reps;
  {
    std::unordered_set<std::size_t> visited;
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (!CGAL::is_border(h, mesh)) continue;
      std::size_t hid = static_cast<std::size_t>(h);
      if (visited.count(hid)) continue;
      loop_reps.push_back(h);
      halfedge_descriptor start = h, cur = h;
      do { visited.insert(static_cast<std::size_t>(cur)); cur = next(cur, mesh); } while (cur != start);
    }
  }
  std::cout << "N_LOOPS: " << loop_reps.size() << std::endl;
  if (target_loop_id >= loop_reps.size()) { std::cerr << "loop_id out of range" << std::endl; return EXIT_FAILURE; }

  std::vector<vertex_descriptor> loop_verts;
  { halfedge_descriptor start = loop_reps[target_loop_id], cur = start;
    do { loop_verts.push_back(target(cur, mesh)); cur = next(cur, mesh); } while (cur != start);
  }
  std::cout << "LOOP_SIZE: " << loop_verts.size() << std::endl;
  std::set<vertex_descriptor> loop_set(loop_verts.begin(), loop_verts.end());

  // home faces: all faces incident to any loop vertex
  std::set<face_descriptor> home_faces;
  for (vertex_descriptor v : loop_verts)
    for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh))
      if (f != Mesh::null_face()) home_faces.insert(f);
  std::cout << "N_HOME_FACES: " << home_faces.size() << std::endl;

  // BFS out 2 more hops for a slightly wider local-context ring
  std::set<face_descriptor> context_faces = home_faces;
  {
    std::set<face_descriptor> frontier = home_faces;
    for (int hop = 0; hop < 2; ++hop) {
      std::set<face_descriptor> next_frontier;
      for (face_descriptor f : frontier) {
        for (vertex_descriptor v : vertices_around_face(halfedge(f, mesh), mesh)) {
          for (face_descriptor f2 : faces_around_target(halfedge(v, mesh), mesh)) {
            if (f2 != Mesh::null_face() && !context_faces.count(f2)) { next_frontier.insert(f2); context_faces.insert(f2); }
          }
        }
      }
      frontier = next_frontier;
    }
  }
  std::cout << "N_CONTEXT_FACES(2-ring): " << context_faces.size() << std::endl;

  // export loop vertices
  { std::ofstream f(out_prefix + "_loop_verts.csv");
    f.precision(17);
    f << "idx,v,x,y,z\n";
    for (std::size_t i = 0; i < loop_verts.size(); ++i) {
      const Point_3& p = mesh.point(loop_verts[i]);
      f << i << "," << loop_verts[i] << "," << to_d(p.x()) << "," << to_d(p.y()) << "," << to_d(p.z()) << "\n";
    }
  }

  // export home faces: normal + centroid + component + whether each vertex is a loop vertex
  { std::ofstream f(out_prefix + "_home_faces.csv");
    f.precision(17);
    f << "face,is_home,cx,cy,cz,nx,ny,nz,area,component,n_loop_verts_in_face\n";
    for (face_descriptor fd : context_faces) {
      bool is_home = home_faces.count(fd) > 0;
      std::vector<Point_3> pts;
      int n_loop_v = 0;
      for (vertex_descriptor v : vertices_around_face(halfedge(fd, mesh), mesh)) {
        pts.push_back(mesh.point(v));
        if (loop_set.count(v)) ++n_loop_v;
      }
      if (pts.size() != 3) continue;
      K::Vector_3 e1 = pts[1]-pts[0], e2 = pts[2]-pts[0];
      K::Vector_3 n = CGAL::cross_product(e1, e2);
      double nl = std::sqrt(to_d(n.squared_length()));
      double nx=0,ny=0,nz=0;
      if (nl > 1e-12) { nx = to_d(n.x())/nl; ny = to_d(n.y())/nl; nz = to_d(n.z())/nl; }
      double cx=0,cy=0,cz=0;
      for (auto& p : pts) { cx += to_d(p.x()); cy += to_d(p.y()); cz += to_d(p.z()); }
      cx/=3; cy/=3; cz/=3;
      double area = to_d(PMP::face_area(fd, mesh));
      faces_size_type comp = get(orig_component, fd);
      f << fd << "," << (is_home?"True":"False") << "," << cx << "," << cy << "," << cz << ","
        << nx << "," << ny << "," << nz << "," << area << "," << comp << "," << n_loop_v << "\n";
    }
  }
  std::cout << "Wrote " << out_prefix << "_loop_verts.csv, " << out_prefix << "_home_faces.csv" << std::endl;
  return EXIT_SUCCESS;
}
