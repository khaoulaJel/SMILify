// Controlled experiment (2026-08-17): isolate CGAL::Polygon_mesh_processing::triangulate_hole
// as the ONLY repair operation, to measure what it solves by itself before introducing component
// bridging, self-intersection repair, or any cleanup pass. Deliberately does NOT call
// triangulate_refine_and_fair_hole (that adds new vertices / smooths geometry, which would
// confound this measurement), does NOT weld/duplicate/collapse vertices, does NOT delete faces,
// does NOT attempt bridging across separate connected components.
//
// Same build pattern as Alpha_wrap_3/examples/Alpha_wrap_3/triangle_soup_wrap.cpp (this repo's
// existing CGAL tool) - CGAL::Surface_mesh<Point_3>, CGAL::IO polygon-soup read/write.
//
// Usage: triangulate_hole_experiment <input.obj> <output.off>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Real_timer.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <set>
#include <unordered_set>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;

struct MeshStats {
  std::size_t n_vertices = 0;
  std::size_t n_faces = 0;
  std::size_t n_components = 0;
  std::size_t n_boundary_edges = 0;
  std::size_t n_boundary_loops = 0;
  std::size_t n_non_manifold_vertices = 0;
  std::size_t n_self_intersecting_pairs = 0;
  std::size_t n_degenerate_faces = 0;
  double surface_area = 0.0;
  bool is_closed = false;
  double volume = 0.0; // only meaningful if is_closed
};

// counts boundary edges/loops without modifying the mesh
static void count_boundary(const Mesh& mesh, std::size_t& n_boundary_edges, std::size_t& n_boundary_loops)
{
  n_boundary_edges = 0;
  std::unordered_set<std::size_t> visited;
  n_boundary_loops = 0;
  for (halfedge_descriptor h : halfedges(mesh)) {
    if (CGAL::is_border(h, mesh)) {
      ++n_boundary_edges;
      std::size_t hid = static_cast<std::size_t>(h);
      if (visited.count(hid)) continue;
      ++n_boundary_loops;
      halfedge_descriptor start = h;
      halfedge_descriptor cur = h;
      do {
        visited.insert(static_cast<std::size_t>(cur));
        cur = next(cur, mesh);
      } while (cur != start);
    }
  }
}

static std::size_t count_degenerate_faces(const Mesh& mesh)
{
  std::size_t n = 0;
  for (face_descriptor f : faces(mesh)) {
    if (PMP::is_degenerate_triangle_face(f, mesh))
      ++n;
  }
  return n;
}

static std::size_t count_components(const Mesh& mesh)
{
  typedef boost::graph_traits<Mesh>::faces_size_type faces_size_type;
  Mesh::Property_map<face_descriptor, faces_size_type> fccmap =
      const_cast<Mesh&>(mesh).add_property_map<face_descriptor, faces_size_type>("f:cc_tmp", 0).first;
  std::size_t n = PMP::connected_components(mesh, fccmap);
  const_cast<Mesh&>(mesh).remove_property_map(fccmap);
  return n;
}

// non_manifold_vertices() collects one HALFEDGE per non-manifold "umbrella" seen at a vertex
// (its target() is the non-manifold vertex) - not vertex_descriptor directly, per the actual API.
static std::size_t count_non_manifold_vertices(const Mesh& mesh, std::set<vertex_descriptor>& out_verts)
{
  std::vector<halfedge_descriptor> nm_halfedges;
  PMP::non_manifold_vertices(mesh, std::back_inserter(nm_halfedges));
  for (halfedge_descriptor h : nm_halfedges)
    out_verts.insert(target(h, mesh));
  return out_verts.size();
}

static MeshStats compute_stats(const Mesh& mesh, bool do_self_intersections)
{
  MeshStats s;
  s.n_vertices = num_vertices(mesh);
  s.n_faces = num_faces(mesh);
  s.n_components = count_components(mesh);
  count_boundary(mesh, s.n_boundary_edges, s.n_boundary_loops);

  std::set<vertex_descriptor> nm_verts_set;
  s.n_non_manifold_vertices = count_non_manifold_vertices(mesh, nm_verts_set);

  s.n_degenerate_faces = count_degenerate_faces(mesh);

  if (do_self_intersections) {
    std::vector<std::pair<face_descriptor, face_descriptor>> si;
    try {
      PMP::self_intersections(mesh, std::back_inserter(si));
      s.n_self_intersecting_pairs = si.size();
    } catch (const std::exception& e) {
      std::cerr << "WARNING: self_intersections() threw: " << e.what() << std::endl;
      s.n_self_intersecting_pairs = static_cast<std::size_t>(-1); // sentinel: not measurable
    }
  } else {
    s.n_self_intersecting_pairs = static_cast<std::size_t>(-1); // skipped (too expensive at this size)
  }

  try {
    s.surface_area = PMP::area(mesh);
  } catch (const std::exception& e) {
    std::cerr << "WARNING: area() threw: " << e.what() << std::endl;
  }

  s.is_closed = CGAL::is_closed(mesh);
  if (s.is_closed) {
    try {
      s.volume = PMP::volume(mesh);
    } catch (const std::exception& e) {
      std::cerr << "WARNING: volume() threw despite is_closed==true: " << e.what() << std::endl;
      s.is_closed = false;
    }
  }
  return s;
}

static void print_stats(const std::string& label, const MeshStats& s)
{
  std::cout << "STATS_" << label << ": n_vertices=" << s.n_vertices
            << " n_faces=" << s.n_faces
            << " n_components=" << s.n_components
            << " n_boundary_edges=" << s.n_boundary_edges
            << " n_boundary_loops=" << s.n_boundary_loops
            << " n_non_manifold_vertices=" << s.n_non_manifold_vertices
            << " n_self_intersecting_pairs=" << (s.n_self_intersecting_pairs == static_cast<std::size_t>(-1) ? std::string("SKIPPED") : std::to_string(s.n_self_intersecting_pairs))
            << " n_degenerate_faces=" << s.n_degenerate_faces
            << " surface_area=" << s.surface_area
            << " is_closed=" << (s.is_closed ? "True" : "False")
            << " volume=" << (s.is_closed ? std::to_string(s.volume) : std::string("NaN"))
            << std::endl;
}

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <output.off>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string output_path = argv[2];

  std::cout << "Reading " << input_path << "..." << std::endl;
  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  if (!CGAL::IO::read_polygon_soup(input_path, points, polygons) || polygons.empty()) {
    std::cerr << "Invalid input: " << input_path << std::endl;
    return EXIT_FAILURE;
  }
  const std::size_t soup_n_points_before = points.size();
  const std::size_t soup_n_polygons_before = polygons.size();
  std::cout << "SOUP_READ: n_points=" << soup_n_points_before << " n_polygons=" << soup_n_polygons_before << std::endl;

  const bool was_already_mesh = PMP::is_polygon_soup_a_polygon_mesh(polygons);
  std::cout << "SOUP_IS_ALREADY_POLYGON_MESH: " << (was_already_mesh ? "True" : "False") << std::endl;

  // orient_polygon_soup may duplicate vertices to resolve non-manifoldness/orientation issues so
  // the soup becomes representable as a halfedge mesh - this is a PRECONDITION-SATISFYING step,
  // distinct from the hole-filling repair itself. Reported separately and transparently, not
  // hidden inside the "repair" numbers below. It does NOT delete any polygon/face.
  PMP::orient_polygon_soup(points, polygons);
  const std::size_t soup_n_points_after_orient = points.size();
  std::cout << "SOUP_AFTER_ORIENT: n_points=" << soup_n_points_after_orient
            << " n_polygons=" << polygons.size()
            << " n_points_added_by_orient=" << (soup_n_points_after_orient - soup_n_points_before)
            << " n_polygons_changed=" << (polygons.size() != soup_n_polygons_before ? "True" : "False")
            << std::endl;

  Mesh mesh;
  PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);
  std::cout << "MESH_AFTER_CONVERSION: n_vertices=" << num_vertices(mesh)
            << " n_faces=" << num_faces(mesh) << std::endl;
  if (num_faces(mesh) != soup_n_polygons_before) {
    std::cout << "WARNING_FACE_COUNT_MISMATCH: soup had " << soup_n_polygons_before
              << " polygons, mesh has " << num_faces(mesh) << " faces after conversion "
              << "(some polygons were not representable as mesh faces - see CGAL "
              << "polygon_soup_to_polygon_mesh docs, typically duplicate/degenerate polygons)."
              << std::endl;
  }

  // Cache original vertex points, indexed by vertex_descriptor, to verify immutability at the end.
  std::vector<Point_3> original_points;
  original_points.reserve(num_vertices(mesh));
  for (vertex_descriptor v : vertices(mesh))
    original_points.push_back(mesh.point(v));
  const std::size_t n_original_vertices = original_points.size();

  const bool do_full_self_intersections = (num_faces(mesh) < 200000); // cost guard, both test specimens are well under this
  MeshStats before = compute_stats(mesh, do_full_self_intersections);
  print_stats("BEFORE", before);

  // Non-manifold vertex set, for per-loop annotation below.
  std::set<vertex_descriptor> nm_verts;
  count_non_manifold_vertices(mesh, nm_verts);

  // Enumerate ALL border-halfedge loop representatives FIRST (before filling any of them) -
  // once a loop is filled its halfedges stop being border halfedges, so this must be a single
  // upfront pass.
  struct LoopInfo {
    halfedge_descriptor rep;
    std::size_t length;
    bool has_non_manifold_vertex;
  };
  std::vector<LoopInfo> loops;
  {
    std::unordered_set<std::size_t> visited;
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (!CGAL::is_border(h, mesh)) continue;
      std::size_t hid = static_cast<std::size_t>(h);
      if (visited.count(hid)) continue;

      std::size_t length = 0;
      bool has_nm = false;
      halfedge_descriptor start = h;
      halfedge_descriptor cur = h;
      do {
        visited.insert(static_cast<std::size_t>(cur));
        if (nm_verts.count(target(cur, mesh))) has_nm = true;
        ++length;
        cur = next(cur, mesh);
      } while (cur != start);

      loops.push_back({h, length, has_nm});
    }
  }
  std::cout << "N_BOUNDARY_LOOPS_FOUND: " << loops.size() << std::endl;

  std::size_t n_success = 0, n_rejected = 0, n_exception = 0;
  std::size_t total_new_faces = 0;
  std::size_t total_new_vertices = 0;

  for (std::size_t i = 0; i < loops.size(); ++i) {
    const LoopInfo& li = loops[i];
    const std::size_t v_before = num_vertices(mesh);
    const std::size_t f_before = num_faces(mesh);

    std::vector<face_descriptor> patch;
    std::string status;
    std::string reason;

    if (li.length < 3) {
      status = "REJECTED";
      reason = "loop_length<3 (degenerate, not attempted - triangulate_hole requires >=3)";
    } else {
      try {
        // Re-verify the representative halfedge is still a valid border halfedge (defensive -
        // should always hold since we never delete anything, but the API precondition is that
        // face(border_halfedge, mesh) == null_face()).
        if (!CGAL::is_border(li.rep, mesh)) {
          status = "REJECTED";
          reason = "representative halfedge no longer border at time of attempt (unexpected - "
                    "investigate if seen)";
        } else {
          PMP::triangulate_hole(mesh, li.rep, CGAL::parameters::face_output_iterator(std::back_inserter(patch)));
          if (patch.empty()) {
            status = "REJECTED";
            reason = "triangulate_hole returned zero faces (algorithm found no valid triangulation "
                      "in its search space - see CGAL Hole_filling docs: can happen for "
                      "self-intersecting, highly non-planar, or otherwise pathological boundaries)";
          } else {
            status = "SUCCESS";
            reason = "-";
          }
        }
      } catch (const std::exception& e) {
        status = "EXCEPTION";
        reason = e.what();
      } catch (...) {
        status = "EXCEPTION";
        reason = "unknown C++ exception (non-std::exception)";
      }
    }

    const std::size_t v_after = num_vertices(mesh);
    const std::size_t f_after = num_faces(mesh);
    const std::size_t new_v = v_after - v_before;
    const std::size_t new_f = f_after - f_before;

    if (status == "SUCCESS") ++n_success;
    else if (status == "REJECTED") ++n_rejected;
    else ++n_exception;
    total_new_faces += new_f;
    total_new_vertices += new_v;

    std::cout << "LOOP_RESULT: idx=" << i << " loop_length=" << li.length
              << " has_non_manifold_vertex=" << (li.has_non_manifold_vertex ? "True" : "False")
              << " status=" << status
              << " new_vertices=" << new_v << " new_faces=" << new_f
              << " reason=\"" << reason << "\"" << std::endl;
  }

  std::cout << "LOOP_SUMMARY: n_loops=" << loops.size()
            << " n_success=" << n_success << " n_rejected=" << n_rejected
            << " n_exception=" << n_exception
            << " total_new_vertices=" << total_new_vertices
            << " total_new_faces=" << total_new_faces << std::endl;

  // Verify original vertex immutability - the core geometry-preservation constraint.
  std::size_t n_moved = 0;
  double max_displacement = 0.0;
  for (std::size_t i = 0; i < n_original_vertices; ++i) {
    vertex_descriptor v(static_cast<std::uint32_t>(i));
    const Point_3& p_now = mesh.point(v);
    const Point_3& p_orig = original_points[i];
    const double d = std::sqrt(CGAL::to_double(CGAL::squared_distance(p_now, p_orig)));
    if (d > 0.0) {
      ++n_moved;
      if (d > max_displacement) max_displacement = d;
    }
  }
  std::cout << "VERTEX_IMMUTABILITY: n_original_vertices=" << n_original_vertices
            << " n_moved=" << n_moved << " max_displacement=" << max_displacement << std::endl;

  const bool do_full_self_intersections_after = (num_faces(mesh) < 200000);
  MeshStats after = compute_stats(mesh, do_full_self_intersections_after);
  print_stats("AFTER", after);

  std::cout << "Writing result to " << output_path << std::endl;
  CGAL::IO::write_polygon_mesh(output_path, mesh, CGAL::parameters::stream_precision(17));

  return EXIT_SUCCESS;
}
