// Controlled mini-experiment (2026-08-17), NOT a production change. Tests
// CGAL::Polygon_mesh_processing::stitch_borders on a COPY of the mesh, to determine whether the
// duplicate-position vertex pairs characterized as likely-genuine-seams by
// duplicate_seam_characterization.cpp actually resolve safely when stitched.
//
// Deliberately uses stitch_borders' own auto-detecting matching (not a hand-built halfedge-pair
// list) because correctly deriving the oriented pair (which halfedge stitches with which,
// accounting for same-direction vs reversed chain walks) by hand was judged too error-prone to
// trust for a topology-changing operation; CGAL's own position-based matching is the robust,
// correct implementation of exactly this logic. This run therefore ALSO serves as an independent
// check on the duplicate_seam_characterization.cpp classifier: if stitch_borders only ever merges
// vertices we classified A, that corroborates the classifier; if it touches B or C pairs too,
// that is reported as a red flag, not silently accepted.
//
// Usage: duplicate_seam_stitch_experiment <input.obj> <specimen_tag> <report_txt_out>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <cmath>
#include <sstream>
#include <cstdint>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;

// 8 genuine near-touch candidates (Strumigenys_alberti), same coordinates used throughout this
// session, for the pre/post preservation check.
struct Candidate { std::string label; Point_3 a, b; double pre_gap; };
static const std::vector<Candidate> STRUMIGENYS_CANDIDATES = {
  {"cand_leftfarcluster_A",  Point_3(-553.329,373.323,162.388), Point_3(-550.746,376.307,163.258), 4.041},
  {"cand_rightfarcluster_A", Point_3(500.403,-248.654,17.331),  Point_3(501.690,-248.217,21.930),  4.796},
  {"cand_rightfarcluster_B", Point_3(556.372,15.116,-344.669),  Point_3(559.884,18.150,-346.722),  5.075},
  {"cand_midlower",          Point_3(90.885,-464.780,-268.590), Point_3(93.982,-461.545,-265.883), 5.234},
  {"cand_rightfarcluster_C", Point_3(358.671,-281.712,-277.352),Point_3(358.083,-276.261,-277.261),5.483},
  {"cand_nearcore",          Point_3(-89.012,8.958,121.246),    Point_3(-87.956,13.613,124.310),   5.672},
  {"cand_leftfarcluster_B",  Point_3(-552.949,376.465,168.640), Point_3(-550.746,376.307,163.258), 5.818},
  {"cand_leftfarcluster_C",  Point_3(-528.255,426.411,142.535), Point_3(-526.154,421.787,145.641), 5.953},
};

static std::size_t count_non_manifold_vertices(const Mesh& mesh)
{
  std::vector<halfedge_descriptor> nm;
  PMP::non_manifold_vertices(mesh, std::back_inserter(nm));
  std::set<vertex_descriptor> s;
  for (halfedge_descriptor h : nm) s.insert(target(h, mesh));
  return s.size();
}

static void count_boundary(const Mesh& mesh, std::size_t& n_edges, std::size_t& n_loops)
{
  n_edges = 0; n_loops = 0;
  std::unordered_set<std::size_t> visited;
  for (halfedge_descriptor h : halfedges(mesh)) {
    if (!CGAL::is_border(h, mesh)) continue;
    ++n_edges;
    std::size_t hid = static_cast<std::size_t>(h);
    if (visited.count(hid)) continue;
    ++n_loops;
    halfedge_descriptor start = h, cur = h;
    do { visited.insert(static_cast<std::size_t>(cur)); cur = next(cur, mesh); } while (cur != start);
  }
}

static std::size_t count_components(Mesh& mesh)
{
  typedef boost::graph_traits<Mesh>::faces_size_type fst;
  auto fcc = mesh.add_property_map<face_descriptor, fst>("f:cc_tmp", 0).first;
  std::size_t n = PMP::connected_components(mesh, fcc);
  mesh.remove_property_map(fcc);
  return n;
}

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <specimen_tag> <report_txt_out> <classified_pairs_csv> [merge_report_csv_out]" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string specimen_tag = argv[2];
  const std::string report_path = argv[3];
  const std::string classified_pairs_csv = argv[4];
  const std::string merge_report_csv = (argc >= 6) ? argv[5] : "";

  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  if (!CGAL::IO::read_polygon_soup(input_path, points, polygons) || polygons.empty()) {
    std::cerr << "Invalid input: " << input_path << std::endl;
    return EXIT_FAILURE;
  }
  PMP::orient_polygon_soup(points, polygons);
  Mesh mesh;
  PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);

  std::ofstream rep(report_path);
  rep.precision(17);

  const std::size_t n_verts_before = num_vertices(mesh);
  const std::size_t n_faces_before = num_faces(mesh);
  std::size_t be_before, bl_before;
  count_boundary(mesh, be_before, bl_before);
  const std::size_t nmv_before = count_non_manifold_vertices(mesh);
  const std::size_t nc_before = count_components(mesh);
  std::vector<std::pair<face_descriptor,face_descriptor>> si_before;
  PMP::self_intersections(mesh, std::back_inserter(si_before));

  std::vector<Point_3> original_points(mesh.points().begin(), mesh.points().end());

  // near-touch gaps before (Strumigenys only)
  auto nearest_vertex = [&](const Point_3& q) {
    vertex_descriptor best; double best_d = std::numeric_limits<double>::infinity();
    for (vertex_descriptor v : vertices(mesh)) {
      double d = CGAL::to_double(CGAL::squared_distance(mesh.point(v), q));
      if (d < best_d) { best_d = d; best = v; }
    }
    return best;
  };
  std::vector<double> gaps_before;
  if (specimen_tag == "strumigenys") {
    for (const auto& c : STRUMIGENYS_CANDIDATES) {
      vertex_descriptor va = nearest_vertex(c.a), vb = nearest_vertex(c.b);
      gaps_before.push_back(std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(va), mesh.point(vb)))));
    }
  }

  rep << "=== BEFORE STITCH ===\n";
  rep << "n_vertices=" << n_verts_before << " n_faces=" << n_faces_before << "\n";
  rep << "n_boundary_edges=" << be_before << " n_boundary_loops=" << bl_before << "\n";
  rep << "n_non_manifold_vertices=" << nmv_before << " n_components=" << nc_before << "\n";
  rep << "n_self_intersecting_pairs=" << si_before.size() << "\n";
  if (specimen_tag == "strumigenys") {
    rep << "near_touch_gaps_before:";
    for (std::size_t i = 0; i < STRUMIGENYS_CANDIDATES.size(); ++i)
      rep << " " << STRUMIGENYS_CANDIDATES[i].label << "=" << gaps_before[i];
    rep << "\n";
  }

  // --- the actual (controlled, mesh-copy-only) operation under test ---
  std::size_t n_stitched = PMP::stitch_borders(mesh);
  rep << "\n=== STITCH_BORDERS CALLED: n_pairs_stitched=" << n_stitched << " ===\n";

  const std::size_t n_verts_after = num_vertices(mesh);
  const std::size_t n_faces_after = num_faces(mesh);
  std::size_t be_after, bl_after;
  count_boundary(mesh, be_after, bl_after);
  const std::size_t nmv_after = count_non_manifold_vertices(mesh);
  const std::size_t nc_after = count_components(mesh);
  std::vector<std::pair<face_descriptor,face_descriptor>> si_after;
  PMP::self_intersections(mesh, std::back_inserter(si_after));

  // Verify immutability of SURVIVING vertices only (stitching removes vertices; it must not move
  // the ones that remain). vertex_descriptor indices may be reused after removal in CGAL::Surface_mesh's
  // internal storage - so we can only safely check: for each ORIGINAL point, does at least one
  // vertex in the (possibly reindexed) surviving mesh still hold that exact point, and is the
  // total count of matched points consistent with (n_verts_before - n_stitched*2 removed... )
  // Simpler and robust: original_points is index-keyed by the PRE-stitch vertex_descriptor ids;
  // after stitching, CGAL::Surface_mesh does NOT renumber surviving vertices (only marks removed
  // ones), so mesh.point(v) for any v still valid (not mesh.is_removed(v)) must be unchanged.
  std::size_t n_moved = 0, n_checked = 0;
  double max_disp = 0.0;
  for (std::size_t i = 0; i < original_points.size(); ++i) {
    vertex_descriptor v(static_cast<std::uint32_t>(i));
    if (mesh.is_removed(v)) continue;
    ++n_checked;
    double d = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(v), original_points[i])));
    if (d > 0.0) { ++n_moved; if (d > max_disp) max_disp = d; }
  }

  std::vector<double> gaps_after;
  if (specimen_tag == "strumigenys") {
    for (const auto& c : STRUMIGENYS_CANDIDATES) {
      vertex_descriptor va = nearest_vertex(c.a), vb = nearest_vertex(c.b);
      gaps_after.push_back(std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(va), mesh.point(vb)))));
    }
  }

  rep << "\n=== AFTER STITCH ===\n";
  rep << "n_vertices=" << n_verts_after << " (removed=" << (n_verts_before - n_verts_after) << ") n_faces=" << n_faces_after << "\n";
  rep << "n_boundary_edges=" << be_after << " (was " << be_before << ") n_boundary_loops=" << bl_after << " (was " << bl_before << ")\n";
  rep << "n_non_manifold_vertices=" << nmv_after << " (was " << nmv_before << ")\n";
  rep << "n_components=" << nc_after << " (was " << nc_before << ")\n";
  rep << "n_self_intersecting_pairs=" << si_after.size() << " (was " << si_before.size() << ")\n";
  rep << "surviving_vertex_immutability: n_checked=" << n_checked << " n_moved=" << n_moved << " max_displacement=" << max_disp << "\n";
  if (specimen_tag == "strumigenys") {
    rep << "near_touch_gaps_after:";
    for (std::size_t i = 0; i < STRUMIGENYS_CANDIDATES.size(); ++i)
      rep << " " << STRUMIGENYS_CANDIDATES[i].label << "=" << gaps_after[i]
          << "(pre_gap=" << STRUMIGENYS_CANDIDATES[i].pre_gap << ")";
    rep << "\n";
    rep << "near_touch_preserved_count=";
    int n_preserved = 0;
    for (std::size_t i = 0; i < STRUMIGENYS_CANDIDATES.size(); ++i)
      if (gaps_after[i] > STRUMIGENYS_CANDIDATES[i].pre_gap * 0.3) ++n_preserved;
    rep << n_preserved << "/8\n";
  }
  rep.close();

  // --- forensic check: for every pair we classified (A/B/C), did stitch_borders actually merge
  // it? Cross-reference against the classification to validate (or refute) the classifier. ---
  if (!classified_pairs_csv.empty() && !merge_report_csv.empty()) {
    std::ifstream in(classified_pairs_csv);
    std::string header; std::getline(in, header);
    std::ofstream out(merge_report_csv);
    out << "v1,v2,classification,got_merged\n";
    std::map<std::string, std::size_t> cls_total, cls_merged;
    std::string line;
    while (std::getline(in, line)) {
      std::stringstream ss(line);
      std::vector<std::string> f;
      std::string tok;
      while (std::getline(ss, tok, ',')) f.push_back(tok);
      if (f.size() < 19) continue;
      auto parse_v = [](std::string s) { return static_cast<std::uint32_t>(std::stoul(s.substr(1))); }; // strip leading 'v'
      vertex_descriptor v1(parse_v(f[0])), v2(parse_v(f[1]));
      std::string cls = f[18];
      bool r1 = mesh.is_removed(v1), r2 = mesh.is_removed(v2);
      bool got_merged = r1 != r2; // exactly one side removed = this pair got merged into the other
      out << f[0] << "," << f[1] << "," << cls << "," << (got_merged ? "True" : "False") << "\n";
      cls_total[cls]++;
      if (got_merged) cls_merged[cls]++;
    }
    out.close();
    rep.open(report_path, std::ios::app);
    rep << "\n=== FORENSIC CROSS-CHECK vs classification (" << classified_pairs_csv << ") ===\n";
    for (const auto& kv : cls_total)
      rep << "class " << kv.first << ": " << cls_merged[kv.first] << "/" << kv.second << " pairs got merged by stitch_borders\n";
    rep.close();
    std::cout << "Wrote " << merge_report_csv << std::endl;
  }

  std::cout << "n_stitched=" << n_stitched
            << " boundary_edges_before=" << be_before << " after=" << be_after
            << " boundary_loops_before=" << bl_before << " after=" << bl_after
            << " nonmanifold_before=" << nmv_before << " after=" << nmv_after
            << " selfint_before=" << si_before.size() << " after=" << si_after.size()
            << " n_moved_surviving_verts=" << n_moved << std::endl;
  std::cout << "Wrote " << report_path << std::endl;
  return EXIT_SUCCESS;
}
