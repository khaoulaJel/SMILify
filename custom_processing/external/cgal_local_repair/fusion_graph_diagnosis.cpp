// Priority 2 pivot (2026-08-17): test whether a tangled boundary is explained by a small number of
// existing vertex/edge incidences that incorrectly fuse two sheets of surface, rather than by
// inventing a reconnection. Exports the home-face dual graph (face adjacency via shared mesh
// edges) with each face's sheet-cluster label and its 3 vertex ids, so Python can find the
// "fusion" cut: face-adjacency edges that cross between sheet clusters, and the mesh
// vertices/edges those crossings pass through. Diagnostic only - does not modify the mesh.
//
// Usage: fusion_graph_diagnosis <input.obj> <loop_id> <out_prefix>

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
#include <map>
#include <unordered_set>
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
    std::cerr << "Usage: " << argv[0] << " <input.obj> <loop_id> <out_prefix>" << std::endl;
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
  std::cout << "LOOP_SIZE: " << loop_verts.size() << std::endl;
  std::set<vertex_descriptor> loop_set(loop_verts.begin(), loop_verts.end());

  // home faces (1-ring from boundary) - the region whose sheet structure we already validated
  std::set<face_descriptor> home_faces;
  for (vertex_descriptor v : loop_verts)
    for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh))
      if (f != Mesh::null_face()) home_faces.insert(f);
  std::cout << "N_HOME_FACES: " << home_faces.size() << std::endl;

  auto face_normal = [&](face_descriptor f) -> Vec3 {
    std::vector<Vec3> pts;
    for (vertex_descriptor v : vertices_around_face(halfedge(f, mesh), mesh)) pts.push_back(P(mesh.point(v)));
    Vec3 nrm = cross(pts[1]-pts[0], pts[2]-pts[0]);
    double l = nrm.norm();
    return l > 1e-12 ? nrm * (1.0/l) : Vec3{0,0,0};
  };

  std::vector<face_descriptor> home_vec(home_faces.begin(), home_faces.end());
  std::vector<Vec3> normals(home_vec.size());
  for (std::size_t i = 0; i < home_vec.size(); ++i) normals[i] = face_normal(home_vec[i]);

  // 2-means clustering (same deterministic method as sheet_split_diagnosis.cpp)
  Vec3 c0 = normals[0], c1 = normals[normals.size()/2];
  for (int iter = 0; iter < 50; ++iter) {
    Vec3 s0{0,0,0}, s1{0,0,0}; int n0=0, n1=0;
    for (auto& nrm : normals) { if (nrm.dot(c0) >= nrm.dot(c1)) { s0=s0+nrm; ++n0; } else { s1=s1+nrm; ++n1; } }
    if (n0>0) c0 = s0*(1.0/n0); if (n1>0) c1 = s1*(1.0/n1);
    double l0=c0.norm(), l1=c1.norm(); if (l0>1e-9) c0=c0*(1.0/l0); if (l1>1e-9) c1=c1*(1.0/l1);
  }
  std::map<face_descriptor,int> label;
  for (std::size_t i = 0; i < home_vec.size(); ++i)
    label[home_vec[i]] = (normals[i].dot(c0) >= normals[i].dot(c1)) ? 0 : 1;

  // export faces (id, label, 3 vertex ids)
  { std::ofstream f(out_prefix + "_faces.csv");
    f << "face,label,v0,v1,v2\n";
    for (face_descriptor fd : home_vec) {
      std::vector<vertex_descriptor> vs;
      for (vertex_descriptor v : vertices_around_face(halfedge(fd, mesh), mesh)) vs.push_back(v);
      f << fd << "," << label[fd] << "," << vs[0] << "," << vs[1] << "," << vs[2] << "\n";
    }
  }

  // dual-graph adjacency: for each home face, each of its 3 edges, find the opposite face (if
  // it's also in home_faces) - record cross-label adjacencies with their shared edge endpoints.
  std::size_t n_same = 0, n_cross = 0;
  std::set<vertex_descriptor> fusion_verts;
  std::set<std::pair<std::uint32_t,std::uint32_t>> fusion_edges;
  { std::ofstream f(out_prefix + "_adjacency.csv");
    f << "face1,face2,label1,label2,shared_v1,shared_v2,cross_label\n";
    std::set<std::pair<face_descriptor,face_descriptor>> seen_pairs;
    for (face_descriptor fd : home_vec) {
      for (halfedge_descriptor h : halfedges_around_face(halfedge(fd, mesh), mesh)) {
        halfedge_descriptor opp = opposite(h, mesh);
        face_descriptor f2 = face(opp, mesh);
        if (f2 == Mesh::null_face() || !home_faces.count(f2)) continue;
        face_descriptor fa = fd < f2 ? fd : f2;
        face_descriptor fb = fd < f2 ? f2 : fd;
        if (seen_pairs.count({fa,fb})) continue;
        seen_pairs.insert({fa,fb});
        vertex_descriptor sv1 = source(h, mesh), sv2 = target(h, mesh);
        bool cross = (label[fd] != label[f2]);
        if (cross) {
          ++n_cross;
          fusion_verts.insert(sv1); fusion_verts.insert(sv2);
          std::uint32_t a = static_cast<std::uint32_t>(sv1), b = static_cast<std::uint32_t>(sv2);
          fusion_edges.insert(a<b ? std::make_pair(a,b) : std::make_pair(b,a));
        } else ++n_same;
        f << fd << "," << f2 << "," << label[fd] << "," << label[f2] << "," << sv1 << "," << sv2 << "," << (cross?"True":"False") << "\n";
      }
    }
  }
  std::cout << "DUAL_ADJACENCY: same_label=" << n_same << " cross_label=" << n_cross << std::endl;
  std::cout << "N_FUSION_VERTICES(touched by a cross-label adjacency): " << fusion_verts.size() << std::endl;
  std::cout << "N_FUSION_EDGES: " << fusion_edges.size() << std::endl;

  // how many fusion vertices are ALSO on the tangled boundary loop itself?
  std::size_t n_fusion_on_loop = 0;
  for (vertex_descriptor v : fusion_verts) if (loop_set.count(v)) ++n_fusion_on_loop;
  std::cout << "N_FUSION_VERTICES_ON_BOUNDARY_LOOP: " << n_fusion_on_loop << " / " << fusion_verts.size() << std::endl;

  std::ofstream fv(out_prefix + "_fusion_verts.csv");
  fv << "v,x,y,z,on_loop\n";
  for (vertex_descriptor v : fusion_verts) {
    const Point_3& p = mesh.point(v);
    fv << v << "," << CGAL::to_double(p.x()) << "," << CGAL::to_double(p.y()) << "," << CGAL::to_double(p.z()) << ","
       << (loop_set.count(v)?"True":"False") << "\n";
  }

  std::cout << "Wrote " << out_prefix << "_faces.csv, _adjacency.csv, _fusion_verts.csv" << std::endl;
  return EXIT_SUCCESS;
}
