// Priority 2, intersection-graph pivot (2026-08-17): compute the ACTUAL geometric self-
// intersections among EXISTING triangles (not hypothetical fill patches) in and around a tangled
// loop's local context, on the pristine mesh. For each intersecting pair local to the defect,
// compute the real intersection segment (CGAL exact triangle-triangle intersection), then group
// segments into continuous curves by shared/near endpoints. Diagnostic only - no mesh modification.
//
// Usage: intersection_graph_diagnosis <input.obj> <loop_id> <out_prefix>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/intersections.h>

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
using Triangle_3 = K::Triangle_3;
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

  // loop + home + 2-ring context (same construction as tangled_sheet_analysis.cpp)
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

  std::set<face_descriptor> home_faces;
  for (vertex_descriptor v : loop_verts)
    for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh))
      if (f != Mesh::null_face()) home_faces.insert(f);

  std::set<face_descriptor> context_faces = home_faces;
  { std::set<face_descriptor> frontier = home_faces;
    for (int hop = 0; hop < 3; ++hop) { // slightly wider than before (3-ring) to be safe
      std::set<face_descriptor> next_frontier;
      for (face_descriptor f : frontier)
        for (vertex_descriptor v : vertices_around_face(halfedge(f, mesh), mesh))
          for (face_descriptor f2 : faces_around_target(halfedge(v, mesh), mesh))
            if (f2 != Mesh::null_face() && !context_faces.count(f2)) { next_frontier.insert(f2); context_faces.insert(f2); }
      frontier = next_frontier;
    }
  }
  std::cout << "N_CONTEXT_FACES(3-ring): " << context_faces.size() << std::endl;

  Mesh::Property_map<face_descriptor, faces_size_type> orig_component =
      mesh.add_property_map<face_descriptor, faces_size_type>("f:orig_cc", 0).first;
  std::size_t n_components = PMP::connected_components(mesh, orig_component);
  std::cout << "N_COMPONENTS: " << n_components << std::endl;

  // global self-intersections on the PRISTINE mesh (existing triangles only - no fill patches exist)
  std::vector<std::pair<face_descriptor,face_descriptor>> all_si;
  PMP::self_intersections(mesh, std::back_inserter(all_si));
  std::cout << "TOTAL_SELF_INTERSECTING_PAIRS(whole mesh): " << all_si.size() << std::endl;

  auto get_triangle = [&](face_descriptor f) -> Triangle_3 {
    std::vector<Point_3> pts;
    for (vertex_descriptor v : vertices_around_face(halfedge(f, mesh), mesh)) pts.push_back(mesh.point(v));
    return Triangle_3(pts[0], pts[1], pts[2]);
  };

  std::vector<std::pair<face_descriptor,face_descriptor>> local_si;
  for (auto& pr : all_si)
    if (context_faces.count(pr.first) && context_faces.count(pr.second))
      local_si.push_back(pr);
  std::cout << "LOCAL_SELF_INTERSECTING_PAIRS(both faces in 3-ring context): " << local_si.size() << std::endl;
  std::size_t local_si_either = 0;
  for (auto& pr : all_si) if (context_faces.count(pr.first) || context_faces.count(pr.second)) ++local_si_either;
  std::cout << "LOCAL_SELF_INTERSECTING_PAIRS(either face in context): " << local_si_either << std::endl;

  // compute actual intersection segments for local pairs
  std::ofstream segf(out_prefix + "_segments.csv");
  segf.precision(17);
  segf << "face1,face2,comp1,comp2,cross_component,type,ax,ay,az,bx,by,bz\n";
  std::size_t n_segment = 0, n_point = 0, n_other = 0, n_cross_comp = 0;
  std::vector<std::pair<Point_3,Point_3>> segments;
  for (auto& pr : local_si) {
    faces_size_type c1 = get(orig_component, pr.first), c2 = get(orig_component, pr.second);
    bool cross_comp = (c1 != c2);
    if (cross_comp) ++n_cross_comp;
    Triangle_3 t1 = get_triangle(pr.first), t2 = get_triangle(pr.second);
    auto result = CGAL::intersection(t1, t2);
    if (!result) { ++n_other; continue; }
    if (const Point_3* p = std::get_if<Point_3>(&*result)) {
      ++n_point;
      segf << pr.first << "," << pr.second << "," << c1 << "," << c2 << "," << (cross_comp?"True":"False") << ",POINT," << to_d(p->x()) << "," << to_d(p->y()) << "," << to_d(p->z()) << ",,,\n";
    } else if (const K::Segment_3* s = std::get_if<K::Segment_3>(&*result)) {
      ++n_segment;
      Point_3 a = s->source(), b = s->target();
      segments.emplace_back(a,b);
      segf << pr.first << "," << pr.second << "," << c1 << "," << c2 << "," << (cross_comp?"True":"False") << ",SEGMENT," << to_d(a.x()) << "," << to_d(a.y()) << "," << to_d(a.z()) << ","
           << to_d(b.x()) << "," << to_d(b.y()) << "," << to_d(b.z()) << "\n";
    } else {
      ++n_other; // triangle/polygon overlap - coplanar or degenerate case
      segf << pr.first << "," << pr.second << "," << c1 << "," << c2 << "," << (cross_comp?"True":"False") << ",OTHER(coplanar_or_complex),,,,,,\n";
    }
  }
  std::cout << "N_CROSS_COMPONENT_INTERSECTING_PAIRS: " << n_cross_comp << " / " << local_si.size() << std::endl;
  segf.close();
  std::cout << "INTERSECTION_TYPES: point=" << n_point << " segment=" << n_segment << " other=" << n_other << std::endl;

  // group segments into curves by endpoint proximity (union-find on near-identical endpoints)
  std::vector<int> parent(segments.size());
  for (std::size_t i = 0; i < segments.size(); ++i) parent[i] = i;
  std::function<int(int)> find = [&](int x) { while (parent[x]!=x) { parent[x]=parent[parent[x]]; x=parent[x]; } return x; };
  auto unite = [&](int a, int b) { a=find(a); b=find(b); if (a!=b) parent[a]=b; };
  const double link_eps = 1e-6;
  for (std::size_t i = 0; i < segments.size(); ++i)
    for (std::size_t j = i+1; j < segments.size(); ++j) {
      double d1 = std::sqrt(to_d(CGAL::squared_distance(segments[i].first, segments[j].first)));
      double d2 = std::sqrt(to_d(CGAL::squared_distance(segments[i].first, segments[j].second)));
      double d3 = std::sqrt(to_d(CGAL::squared_distance(segments[i].second, segments[j].first)));
      double d4 = std::sqrt(to_d(CGAL::squared_distance(segments[i].second, segments[j].second)));
      if (std::min({d1,d2,d3,d4}) < link_eps) unite(i,j);
    }
  std::map<int, std::vector<std::size_t>> curves;
  for (std::size_t i = 0; i < segments.size(); ++i) curves[find(i)].push_back(i);
  std::cout << "N_INTERSECTION_CURVES(connected segment chains): " << curves.size() << std::endl;
  std::vector<std::size_t> curve_sizes;
  for (auto& kv : curves) curve_sizes.push_back(kv.second.size());
  std::sort(curve_sizes.rbegin(), curve_sizes.rend());
  std::cout << "CURVE_SIZES(n_segments):";
  for (auto s : curve_sizes) std::cout << " " << s;
  std::cout << std::endl;

  // total length + spatial extent of the largest curve
  if (!curve_sizes.empty()) {
    int biggest_root = -1; std::size_t biggest_size = 0;
    for (auto& kv : curves) if (kv.second.size() > biggest_size) { biggest_size = kv.second.size(); biggest_root = kv.first; }
    double total_len = 0; std::vector<Point_3> pts;
    for (auto idx : curves[biggest_root]) {
      total_len += std::sqrt(to_d(CGAL::squared_distance(segments[idx].first, segments[idx].second)));
      pts.push_back(segments[idx].first); pts.push_back(segments[idx].second);
    }
    double minx=1e300,miny=1e300,minz=1e300,maxx=-1e300,maxy=-1e300,maxz=-1e300;
    for (auto& p : pts) { minx=std::min(minx,to_d(p.x())); miny=std::min(miny,to_d(p.y())); minz=std::min(minz,to_d(p.z()));
                          maxx=std::max(maxx,to_d(p.x())); maxy=std::max(maxy,to_d(p.y())); maxz=std::max(maxz,to_d(p.z())); }
    double diag = std::sqrt((maxx-minx)*(maxx-minx)+(maxy-miny)*(maxy-miny)+(maxz-minz)*(maxz-minz));
    std::cout << "LARGEST_CURVE: n_segments=" << biggest_size << " total_length=" << total_len << " bbox_diag=" << diag << std::endl;
  }

  std::cout << "Wrote " << out_prefix << "_segments.csv" << std::endl;
  return EXIT_SUCCESS;
}
