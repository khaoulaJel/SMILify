// Experiment A (2026-08-17): for SIMPLE + near-planar boundary loops (identified in Python from
// boundary_loop_analysis.cpp output via a data-driven planarity split, NOT from success labels),
// compare CGAL::Polygon_mesh_processing::triangulate_hole default mode (use_delaunay_triangulation,
// 3D/cubic search space) against the explicit 2D constrained-Delaunay mode
// (use_2d_constrained_delaunay_triangulation(true)) - CGAL documents the latter as intended for
// near-planar holes. Each requested loop is filled TWICE, independently, on a FRESH copy of the
// pristine (unfilled) mesh each time - isolates "does CDT mode produce a better patch for THIS
// hole" from patch-vs-other-patch interactions/fill order, which is not what this experiment
// measures. Does NOT use triangulate_refine_and_fair_hole. Does NOT modify original vertices.
//
// Usage: experimentA_planar_hole_comparison <input.obj> <loop_ids_file> <results_csv_out>
//   loop_ids_file: one integer loop_id (0-based, matching boundary_loop_analysis.cpp's
//   enumeration order) per line.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/measure.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;

struct LoopInfo { halfedge_descriptor rep; std::size_t length; };

static std::vector<LoopInfo> enumerate_loops(const Mesh& mesh)
{
  std::vector<LoopInfo> loops;
  std::unordered_set<std::size_t> visited;
  for (halfedge_descriptor h : halfedges(mesh)) {
    if (!CGAL::is_border(h, mesh)) continue;
    std::size_t hid = static_cast<std::size_t>(h);
    if (visited.count(hid)) continue;
    std::size_t length = 0;
    halfedge_descriptor start = h, cur = h;
    do { visited.insert(static_cast<std::size_t>(cur)); ++length; cur = next(cur, mesh); } while (cur != start);
    loops.push_back({h, length});
  }
  return loops;
}

static std::size_t count_non_manifold_vertices(const Mesh& mesh)
{
  std::vector<halfedge_descriptor> nm_halfedges;
  PMP::non_manifold_vertices(mesh, std::back_inserter(nm_halfedges));
  std::set<vertex_descriptor> out;
  for (halfedge_descriptor h : nm_halfedges) out.insert(target(h, mesh));
  return out.size();
}

struct FillResult {
  bool success = false;
  std::size_t n_new_faces = 0;
  double patch_area = 0.0;
  std::size_t n_patch_vs_existing = 0;
  std::size_t n_patch_self_intersect = 0; // patch-vs-patch (incl. self, since only one patch exists in this isolated test)
  std::size_t n_new_non_manifold_vertices = 0;
  std::size_t n_degenerate_faces = 0;
  std::size_t n_original_vertices_moved = 0;
  double max_original_vertex_displacement = 0.0;
};

static FillResult run_variant(const Mesh& base_mesh, halfedge_descriptor rep, bool use_cdt2d)
{
  FillResult r;
  Mesh mesh = base_mesh; // deep copy; CGAL::Surface_mesh preserves descriptor indices across copy
  const std::size_t f_before = num_faces(mesh);
  const std::size_t nmv_before = count_non_manifold_vertices(mesh);

  std::vector<Point_3> original_points;
  original_points.reserve(num_vertices(mesh));
  for (vertex_descriptor v : vertices(mesh)) original_points.push_back(mesh.point(v));

  std::vector<face_descriptor> patch;
  try {
    if (use_cdt2d) {
      PMP::triangulate_hole(mesh, rep,
          CGAL::parameters::face_output_iterator(std::back_inserter(patch))
                            .use_2d_constrained_delaunay_triangulation(true));
    } else {
      PMP::triangulate_hole(mesh, rep,
          CGAL::parameters::face_output_iterator(std::back_inserter(patch)));
    }
  } catch (...) {
    return r; // success=false
  }
  if (patch.empty()) return r;

  r.success = true;
  r.n_new_faces = patch.size();
  for (face_descriptor f : patch) {
    r.patch_area += CGAL::to_double(PMP::face_area(f, mesh));
    if (PMP::is_degenerate_triangle_face(f, mesh)) ++r.n_degenerate_faces;
  }

  // patch faces are those with index >= f_before (CGAL::Surface_mesh appends, never recycles, here)
  auto is_patch_face = [&](face_descriptor f) { return static_cast<std::size_t>(f) >= f_before; };

  std::vector<std::pair<face_descriptor, face_descriptor>> pairs;
  PMP::self_intersections(mesh, std::back_inserter(pairs));
  for (const auto& pr : pairs) {
    bool p1 = is_patch_face(pr.first), p2 = is_patch_face(pr.second);
    if (p1 && p2) ++r.n_patch_self_intersect;
    else if (p1 || p2) ++r.n_patch_vs_existing;
    // both-original pairs (pre-existing, unrelated to this loop) are ignored
  }

  const std::size_t nmv_after = count_non_manifold_vertices(mesh);
  r.n_new_non_manifold_vertices = (nmv_after > nmv_before) ? (nmv_after - nmv_before) : 0;

  for (std::size_t i = 0; i < original_points.size(); ++i) {
    vertex_descriptor v(static_cast<std::uint32_t>(i));
    double d = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(v), original_points[i])));
    if (d > 0.0) { ++r.n_original_vertices_moved; if (d > r.max_original_vertex_displacement) r.max_original_vertex_displacement = d; }
  }
  return r;
}

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <loop_ids_file> <results_csv_out>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string loop_ids_path = argv[2];
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

  std::vector<LoopInfo> loops = enumerate_loops(mesh);
  std::cout << "N_BOUNDARY_LOOPS_FOUND: " << loops.size() << std::endl;

  std::vector<std::size_t> requested_ids;
  { std::ifstream f(loop_ids_path); std::size_t id; while (f >> id) requested_ids.push_back(id); }
  std::cout << "N_REQUESTED_LOOPS: " << requested_ids.size() << std::endl;

  std::ofstream csv(csv_path);
  csv << "loop_id,length,variant,success,n_new_faces,patch_area,n_patch_vs_existing,"
         "n_patch_self_intersect,n_new_non_manifold_vertices,n_degenerate_faces,"
         "n_original_vertices_moved,max_original_vertex_displacement\n";

  for (std::size_t lid : requested_ids) {
    if (lid >= loops.size()) { std::cerr << "SKIP invalid loop_id=" << lid << std::endl; continue; }
    const LoopInfo& li = loops[lid];
    for (int variant = 0; variant < 2; ++variant) {
      bool use_cdt2d = (variant == 1);
      FillResult r = run_variant(mesh, li.rep, use_cdt2d);
      std::string vname = use_cdt2d ? "B_2D_CDT" : "A_default";
      csv << lid << "," << li.length << "," << vname << "," << (r.success ? "True" : "False") << ","
          << r.n_new_faces << "," << r.patch_area << "," << r.n_patch_vs_existing << ","
          << r.n_patch_self_intersect << "," << r.n_new_non_manifold_vertices << ","
          << r.n_degenerate_faces << "," << r.n_original_vertices_moved << ","
          << r.max_original_vertex_displacement << "\n";
      std::cout << "LOOP_RESULT: loop_id=" << lid << " length=" << li.length << " variant=" << vname
                << " success=" << r.success << " n_new_faces=" << r.n_new_faces
                << " patch_area=" << r.patch_area
                << " n_patch_vs_existing=" << r.n_patch_vs_existing
                << " n_patch_self_intersect=" << r.n_patch_self_intersect
                << " n_new_non_manifold_vertices=" << r.n_new_non_manifold_vertices
                << " n_degenerate_faces=" << r.n_degenerate_faces
                << " vertices_moved=" << r.n_original_vertices_moved << std::endl;
    }
  }
  csv.close();
  std::cout << "Wrote " << csv_path << std::endl;
  return EXIT_SUCCESS;
}
