// Stage 2/3 (2026-08-17): explicit, classifier-constrained duplicate-seam stitching. Uses the
// Stage-1-validated primitive: PMP::internal::collect_duplicated_stitchable_boundary_edges()
// restricted to an EXPLICIT candidate halfedge list (only border halfedges of vertices that (a)
// are classified 'A' by duplicate_seam_characterization.cpp, (b) sit on a validated coherent
// border chain (chain_match_length >= 3, not an isolated one-ring-only match), (c) are same-
// component with their duplicate partner, (d) are not within the near-touch exclusion radius),
// then the CONFIRMED WORKING explicit-pairs PMP::stitch_borders(pmesh, hedge_pairs_to_stitch).
// Every B/C pair, cross-component pair, and near-touch-adjacent vertex is structurally absent
// from the candidate set, not merely hoped to be skipped.
//
// Does NOT run triangulate_hole, self-intersection repair, component bridging, or Alpha Wrap.
//
// Usage: explicit_A_only_stitch <input.obj> <classified_pairs_csv> <specimen_tag> <report_txt_out>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/measure.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <cmath>
#include <cstdint>
#include <limits>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;
typedef boost::graph_traits<Mesh>::faces_size_type faces_size_type;

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
static const double NEAR_TOUCH_EXCLUSION_RADIUS = 20.0; // same generous radius used in classification

static std::size_t count_non_manifold_vertices(const Mesh& mesh)
{
  std::vector<halfedge_descriptor> nm;
  PMP::non_manifold_vertices(mesh, std::back_inserter(nm));
  std::set<vertex_descriptor> s;
  for (halfedge_descriptor h : nm) s.insert(target(h, mesh));
  return s.size();
}
static std::size_t count_non_manifold_edges(const Mesh& mesh)
{
  // an edge is non-manifold in a Surface_mesh (which only stores 2 halfedges per edge) if either
  // of its vertices is non-manifold AND the edge itself is incident to that umbrella issue; as a
  // practical proxy consistent with the rest of this session's tooling, we report vertices only
  // (Surface_mesh's own representation cannot directly encode a >2-face edge in the first place).
  return 0;
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
  auto fcc = mesh.add_property_map<face_descriptor, faces_size_type>("f:cc_tmp", 0).first;
  std::size_t n = PMP::connected_components(mesh, fcc);
  mesh.remove_property_map(fcc);
  return n;
}
static std::size_t count_degenerate_faces(const Mesh& mesh)
{
  std::size_t n = 0;
  for (face_descriptor f : faces(mesh)) if (PMP::is_degenerate_triangle_face(f, mesh)) ++n;
  return n;
}

struct PairRow {
  std::uint32_t v1, v2;
  std::string classification;
  int chain_match_length;
  bool same_component;
  bool near_genuine;
};

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <classified_pairs_csv> <specimen_tag> <report_txt_out>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string classified_csv = argv[2];
  const std::string specimen_tag = argv[3];
  const std::string report_path = argv[4];

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

  // --- load classification CSV ---
  std::vector<PairRow> all_pairs;
  {
    std::ifstream in(classified_csv);
    std::string header; std::getline(in, header);
    std::string line;
    auto parse_v = [](const std::string& s) { return static_cast<std::uint32_t>(std::stoul(s.substr(1))); };
    while (std::getline(in, line)) {
      std::stringstream ss(line);
      std::vector<std::string> f; std::string tok;
      while (std::getline(ss, tok, ',')) f.push_back(tok);
      if (f.size() < 19) continue;
      PairRow r;
      r.v1 = parse_v(f[0]); r.v2 = parse_v(f[1]);
      r.same_component = (f[9] == "True");
      r.chain_match_length = std::stoi(f[11]);
      r.near_genuine = (f[14] == "True");
      r.classification = f[18];
      all_pairs.push_back(r);
    }
  }
  std::cout << "N_CLASSIFIED_PAIRS_LOADED: " << all_pairs.size() << std::endl;

  // --- near-touch exclusion zone (Strumigenys only) ---
  auto within_near_touch_zone = [&](const Point_3& p) {
    if (specimen_tag != "strumigenys") return false;
    for (const auto& c : STRUMIGENYS_CANDIDATES) {
      double da = std::sqrt(CGAL::to_double(CGAL::squared_distance(p, c.a)));
      double db = std::sqrt(CGAL::to_double(CGAL::squared_distance(p, c.b)));
      if (da < NEAR_TOUCH_EXCLUSION_RADIUS || db < NEAR_TOUCH_EXCLUSION_RADIUS) return true;
    }
    return false;
  };

  // --- select the SAFE candidate set: A-classified, coherent chain (>=3), same component, and
  // BOTH vertices geometrically clear of the near-touch exclusion zone ---
  std::set<std::uint32_t> safe_vertex_ids;
  std::size_t n_rejected_not_A = 0, n_rejected_short_chain = 0, n_rejected_diff_component = 0, n_rejected_near_touch = 0;
  std::size_t n_accepted_pairs = 0;
  for (const auto& r : all_pairs) {
    if (r.classification != "A") { ++n_rejected_not_A; continue; }
    int chain_cap_max = getenv("STITCH_NO_CHAIN_CAP") ? 100000 : 150;
    if (r.chain_match_length < 3 || r.chain_match_length > chain_cap_max) { ++n_rejected_short_chain; continue; } // "do not stitch isolated pairs" -> require a validated coherent chain
    if (!r.same_component) { ++n_rejected_diff_component; continue; }
    vertex_descriptor v1(r.v1), v2(r.v2);
    if (within_near_touch_zone(mesh.point(v1)) || within_near_touch_zone(mesh.point(v2))) { ++n_rejected_near_touch; continue; }
    safe_vertex_ids.insert(r.v1);
    safe_vertex_ids.insert(r.v2);
    ++n_accepted_pairs;
  }
  // A vertex can appear in MULTIPLE csv rows if it's part of a >2-way duplicate cluster (e.g. the
  // "middle" vertex of a 3-way coincident-position group): it may be safely A-classified via one
  // relationship yet still geometrically coincide with a B/C partner from a DIFFERENT relationship.
  // collect_duplicated_stitchable_boundary_edges matches by geometry, not by which csv row a
  // vertex came from, so any such vertex must be excluded globally, not just per-row.
  std::size_t n_excluded_multiway = 0;
  for (const auto& r : all_pairs) {
    if (r.classification == "A") continue;
    if (safe_vertex_ids.erase(r.v1)) ++n_excluded_multiway;
    if (safe_vertex_ids.erase(r.v2)) ++n_excluded_multiway;
  }
  std::cout << "MULTIWAY_CLUSTER_EXCLUSION: removed " << n_excluded_multiway
            << " vertices that also appear in a non-A row" << std::endl;
  std::cout << "SELECTION: accepted_pairs=" << n_accepted_pairs
            << " rejected_not_A=" << n_rejected_not_A
            << " rejected_short_chain(<3)=" << n_rejected_short_chain
            << " rejected_diff_component=" << n_rejected_diff_component
            << " rejected_near_touch_zone=" << n_rejected_near_touch
            << " n_safe_vertices=" << safe_vertex_ids.size() << std::endl;

  // --- build the candidate halfedge range: border halfedges whose BOTH endpoints (source and
  // target) are safe vertices - collect_duplicated_stitchable_boundary_edges matches whole EDGES
  // (keyed by both endpoint positions), so a halfedge with only its target vertex "safe" but its
  // source vertex excluded cannot possibly find its reversed counterpart within the candidate set. ---
  std::vector<halfedge_descriptor> candidate_halfedges;
  bool debug_all = (getenv("STITCH_DEBUG_ALL_BORDER") != nullptr);
  for (halfedge_descriptor h : halfedges(mesh)) {
    if (!CGAL::is_border(h, mesh)) continue;
    if (debug_all) { candidate_halfedges.push_back(h); continue; }
    if (safe_vertex_ids.count(static_cast<std::uint32_t>(source(h, mesh))) &&
        safe_vertex_ids.count(static_cast<std::uint32_t>(target(h, mesh))))
      candidate_halfedges.push_back(h);
  }
  std::cout << "N_CANDIDATE_BORDER_HALFEDGES: " << candidate_halfedges.size() << std::endl;
  if (getenv("STITCH_DEBUG")) {
    for (halfedge_descriptor h : candidate_halfedges) {
      const Point_3& ps = mesh.point(source(h, mesh));
      const Point_3& pt = mesh.point(target(h, mesh));
      std::cout << "  cand_h: src=v" << source(h,mesh) << "(" << ps << ") tgt=v" << target(h,mesh) << "(" << pt << ")\n";
    }
  }
  if (getenv("STITCH_DEBUG_MULT")) {
    // undirected-position-key multiplicity histogram among candidate_halfedges
    std::map<std::pair<Point_3,Point_3>, int> key_count;
    for (halfedge_descriptor h : candidate_halfedges) {
      Point_3 a = mesh.point(source(h,mesh)), b = mesh.point(target(h,mesh));
      auto key = (a < b) ? std::make_pair(a,b) : std::make_pair(b,a);
      key_count[key]++;
    }
    std::map<int,int> hist;
    for (auto& kv : key_count) hist[kv.second]++;
    std::cout << "MULTIPLICITY_HISTOGRAM (undirected key -> count of keys with that many halfedges):\n";
    for (auto& kv : hist) std::cout << "  multiplicity=" << kv.first << " : " << kv.second << " keys\n";
  }

  // --- BEFORE stats ---
  std::size_t be_before, bl_before;
  count_boundary(mesh, be_before, bl_before);
  std::size_t nmv_before = count_non_manifold_vertices(mesh);
  std::size_t nc_before = count_components(mesh);
  std::size_t deg_before = count_degenerate_faces(mesh);
  std::vector<std::pair<face_descriptor,face_descriptor>> si0;
  PMP::self_intersections(mesh, std::back_inserter(si0));
  std::size_t si_before = si0.size();
  std::vector<Point_3> original_points(mesh.points().begin(), mesh.points().end());

  std::vector<double> gaps_before;
  auto nearest_vertex = [&](const Point_3& q) {
    vertex_descriptor best; double best_d = std::numeric_limits<double>::infinity();
    for (vertex_descriptor v : vertices(mesh)) {
      double d = CGAL::to_double(CGAL::squared_distance(mesh.point(v), q));
      if (d < best_d) { best_d = d; best = v; }
    }
    return best;
  };
  if (specimen_tag == "strumigenys")
    for (const auto& c : STRUMIGENYS_CANDIDATES) {
      vertex_descriptor va = nearest_vertex(c.a), vb = nearest_vertex(c.b);
      gaps_before.push_back(std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(va), mesh.point(vb)))));
    }

  // --- collect pairs (Stage-1-validated primitive) and stitch (confirmed-working explicit-pairs overload) ---
  std::size_t n_stitched = 0;
  if (getenv("STITCH_TWO_STAGE")) {
    // Replicate stitch_borders(pmesh)'s internal two-stage pipeline manually, restricted to the
    // candidate cycle representatives, using the PUBLIC (non-hanging, per earlier Stage-1 finding
    // about the buggy stitch_borders(reps,pmesh) wrapper) stitch_boundary_cycles() function first,
    // then collect_duplicated_stitchable_boundary_edges on the remainder - this mirrors exactly
    // what the confirmed-working global stitch_borders(pmesh) does under the hood.
    // stitch_boundary_cycles walks the FULL native boundary loop from a representative (via
    // next()), not just the halfedges present in candidate_halfedges - so a representative may
    // belong to a loop that also contains B/C vertices absent from our candidate set, "leaking"
    // stitching into unsafe territory even though only safe halfedges were ever listed as
    // candidates. Guard: only accept a loop's representative if EVERY vertex on its full native
    // loop is in safe_vertex_ids (whole-loop purity), not merely the representative itself.
    std::vector<halfedge_descriptor> cand_reps;
    { std::unordered_set<std::size_t> visited;
      for (halfedge_descriptor h : candidate_halfedges) {
        std::size_t hid = static_cast<std::size_t>(h);
        if (visited.count(hid)) continue;
        bool pure = true;
        std::vector<std::size_t> loop_halfedges;
        halfedge_descriptor cur = h;
        do {
          loop_halfedges.push_back(static_cast<std::size_t>(cur));
          if (!safe_vertex_ids.count(static_cast<std::uint32_t>(target(cur, mesh)))) pure = false;
          cur = next(cur, mesh);
        } while (cur != h);
        for (std::size_t hid2 : loop_halfedges) visited.insert(hid2);
        if (pure) cand_reps.push_back(h);
      }
    }
    std::cout << "TWO_STAGE: n_cand_reps(whole-loop-pure)=" << cand_reps.size() << std::endl << std::flush;
    n_stitched = PMP::stitch_boundary_cycles(cand_reps, mesh);
    std::cout << "TWO_STAGE: after stitch_boundary_cycles, n_stitched_stage1=" << n_stitched << std::endl << std::flush;

    std::vector<halfedge_descriptor> remaining;
    for (halfedge_descriptor h : candidate_halfedges) if (CGAL::is_border(h, mesh)) remaining.push_back(h);
    std::cout << "TWO_STAGE: n_remaining_border=" << remaining.size() << std::endl << std::flush;

    std::vector<std::pair<halfedge_descriptor, halfedge_descriptor>> pairs_to_stitch;
    PMP::internal::Default_halfedges_keeper<Mesh> hd_kpr;
    PMP::internal::collect_duplicated_stitchable_boundary_edges(
        remaining, mesh, hd_kpr, false, std::back_inserter(pairs_to_stitch),
        CGAL::parameters::default_values());
    std::cout << "TWO_STAGE: n_matched_pairs_stage2=" << pairs_to_stitch.size() << std::endl << std::flush;
    n_stitched += PMP::stitch_borders(mesh, pairs_to_stitch);
  } else {
    std::vector<std::pair<halfedge_descriptor, halfedge_descriptor>> pairs_to_stitch;
    PMP::internal::Default_halfedges_keeper<Mesh> hd_kpr;
    PMP::internal::collect_duplicated_stitchable_boundary_edges(
        candidate_halfedges, mesh, hd_kpr, false, std::back_inserter(pairs_to_stitch),
        CGAL::parameters::default_values());
    std::cout << "N_MATCHED_PAIRS_WITHIN_CANDIDATE_SET: " << pairs_to_stitch.size() << std::endl;
    n_stitched = PMP::stitch_borders(mesh, pairs_to_stitch);
  }

  // --- AFTER stats ---
  std::size_t be_after, bl_after;
  count_boundary(mesh, be_after, bl_after);
  std::size_t nmv_after = count_non_manifold_vertices(mesh);
  std::size_t nc_after = count_components(mesh);
  std::size_t deg_after = count_degenerate_faces(mesh);
  std::vector<std::pair<face_descriptor,face_descriptor>> si1;
  PMP::self_intersections(mesh, std::back_inserter(si1));
  std::size_t si_after = si1.size();

  std::size_t n_moved = 0, n_checked = 0; double max_disp = 0.0;
  for (std::size_t i = 0; i < original_points.size(); ++i) {
    vertex_descriptor v(static_cast<std::uint32_t>(i));
    if (mesh.is_removed(v)) continue;
    ++n_checked;
    double d = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(v), original_points[i])));
    if (d > 0.0) { ++n_moved; if (d > max_disp) max_disp = d; }
  }

  std::vector<double> gaps_after;
  if (specimen_tag == "strumigenys")
    for (const auto& c : STRUMIGENYS_CANDIDATES) {
      vertex_descriptor va = nearest_vertex(c.a), vb = nearest_vertex(c.b);
      gaps_after.push_back(std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(va), mesh.point(vb)))));
    }

  // --- forensic check: did the stitch touch ANYTHING outside the safe candidate set? ---
  std::size_t n_bc_touched = 0, n_diffcomp_touched = 0, n_neartouch_touched = 0;
  for (const auto& r : all_pairs) {
    vertex_descriptor v1(r.v1), v2(r.v2);
    bool r1 = mesh.is_removed(v1), r2 = mesh.is_removed(v2);
    bool got_merged = (r1 != r2);
    if (!got_merged) continue;
    if (r.classification != "A") { ++n_bc_touched; std::cout << "  BC_TOUCHED: v" << r.v1 << " v" << r.v2 << " cls=" << r.classification << " chain=" << r.chain_match_length << std::endl; }
    if (!r.same_component) ++n_diffcomp_touched;
    if (r.near_genuine) ++n_neartouch_touched;
  }

  std::ofstream rep(report_path);
  rep.precision(17);
  rep << "=== SELECTION ===\n";
  rep << "accepted_pairs=" << n_accepted_pairs << " rejected_not_A=" << n_rejected_not_A
      << " rejected_short_chain=" << n_rejected_short_chain
      << " rejected_diff_component=" << n_rejected_diff_component
      << " rejected_near_touch_zone=" << n_rejected_near_touch << "\n";
  rep << "n_candidate_border_halfedges=" << candidate_halfedges.size()
      << " n_stitched=" << n_stitched << "\n";
  rep << "\n=== BEFORE ===\n";
  rep << "n_boundary_edges=" << be_before << " n_boundary_loops=" << bl_before << "\n";
  rep << "n_non_manifold_vertices=" << nmv_before << " n_components=" << nc_before << "\n";
  rep << "n_degenerate_faces=" << deg_before << " n_self_intersecting_pairs=" << si_before << "\n";
  rep << "\n=== AFTER ===\n";
  rep << "n_boundary_edges=" << be_after << " (delta=" << (long)(be_after)-(long)(be_before) << ")\n";
  rep << "n_boundary_loops=" << bl_after << " (delta=" << (long)(bl_after)-(long)(bl_before) << ")\n";
  rep << "n_non_manifold_vertices=" << nmv_after << " (delta=" << (long)(nmv_after)-(long)(nmv_before) << ")\n";
  rep << "n_components=" << nc_after << " (delta=" << (long)(nc_after)-(long)(nc_before) << ")\n";
  rep << "n_degenerate_faces=" << deg_after << " (delta=" << (long)(deg_after)-(long)(deg_before) << ")\n";
  rep << "n_self_intersecting_pairs=" << si_after << " (delta=" << (long)(si_after)-(long)(si_before) << ")\n";
  rep << "vertex_displacement: n_checked=" << n_checked << " n_moved=" << n_moved << " max_disp=" << max_disp << "\n";
  rep << "\n=== FORENSIC SAFETY CHECK ===\n";
  rep << "n_B_or_C_pairs_touched=" << n_bc_touched << " (MUST be 0)\n";
  rep << "n_diffcomponent_pairs_touched=" << n_diffcomp_touched << " (MUST be 0)\n";
  rep << "n_near_genuine_touch_pairs_touched=" << n_neartouch_touched << " (MUST be 0)\n";
  if (specimen_tag == "strumigenys") {
    rep << "\n=== NEAR-TOUCH GAPS ===\n";
    int n_preserved = 0;
    for (std::size_t i = 0; i < STRUMIGENYS_CANDIDATES.size(); ++i) {
      rep << STRUMIGENYS_CANDIDATES[i].label << ": before=" << gaps_before[i] << " after=" << gaps_after[i]
          << " pre_gap=" << STRUMIGENYS_CANDIDATES[i].pre_gap << "\n";
      if (gaps_after[i] > STRUMIGENYS_CANDIDATES[i].pre_gap * 0.3) ++n_preserved;
    }
    rep << "near_touch_preserved_count=" << n_preserved << "/8\n";
  }
  rep.close();

  std::cout << "n_stitched=" << n_stitched
            << " be=" << be_before << "->" << be_after
            << " bl=" << bl_before << "->" << bl_after
            << " nc=" << nc_before << "->" << nc_after
            << " nmv=" << nmv_before << "->" << nmv_after
            << " si=" << si_before << "->" << si_after
            << " n_moved=" << n_moved
            << " n_B_or_C_touched=" << n_bc_touched
            << " n_diffcomp_touched=" << n_diffcomp_touched
            << " n_neartouch_touched=" << n_neartouch_touched << std::endl;
  std::cout << "Wrote " << report_path << std::endl;
  return EXIT_SUCCESS;
}
