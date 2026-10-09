// Stage 1 (2026-08-17): mechanically validate the halfedge-pairing/stitching convention on
// synthetic toy meshes before touching the insect meshes.
//
// First attempted CGAL::Polygon_mesh_processing::stitch_borders(boundary_cycle_representatives,
// pmesh) - the documented public "restricted" overload - but it HANGS even on the simplest
// possible 2-triangle case (see tiny_stitch_test.cpp), a bug in this vendored CGAL version's
// internal cycle bookkeeping, not something to build production logic on top of. Caught here in
// Stage 1 exactly as intended, before it could hang on the real meshes.
//
// Settled approach instead: PMP::internal::collect_duplicated_stitchable_boundary_edges() - the
// lower-level matching primitive that the (confirmed-working) global stitch_borders(pmesh) also
// uses internally - called directly on an explicit candidate halfedge list, feeding its output
// pairs to the CONFIRMED WORKING explicit-pairs stitch_borders(pmesh, hedge_pairs). This avoids
// (a) hand-deriving individual halfedge pairs (judged too error-prone previously), (b) the
// global/unrestricted auto-stitch (found unsafe: touched B/C-classified pairs on the real meshes),
// and (c) the buggy restricted-cycle overload found above.
//
// Cases:
//   1. two identical duplicated boundary chains, same traversal direction
//   2. two identical duplicated boundary chains, opposite traversal direction
//   3. a closed duplicated seam (whole hexagonal disk boundary duplicated)
//   4. intentionally incorrect pairing (a representative loop with no real geometric match)
//
// Usage: synthetic_stitch_validation

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>

#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <cmath>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;

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

// NOTE: the public stitch_borders(boundary_cycle_representatives, pmesh) overload was found to
// HANG even on the simplest possible 2-triangle case (verified separately in tiny_stitch_test.cpp)
// - likely a bug in this vendored CGAL version's stitch_boundary_cycles() bookkeeping when
// representatives don't already form a closed self-matching cycle. Per Stage 1's mandate ("do not
// proceed until the pairing convention is demonstrated mechanically"), that overload is NOT used.
// Instead: PMP::internal::collect_duplicated_stitchable_boundary_edges() - the lower-level building
// block that the (working) global stitch_borders(pmesh) also uses internally - is called directly,
// restricted to an explicit candidate halfedge range, to get correctly-paired (h1,h2), and the
// result is fed to the CONFIRMED WORKING explicit-pairs stitch_borders(pmesh, hedge_pairs).
static std::vector<halfedge_descriptor> all_border_halfedges(const Mesh& mesh)
{
  std::vector<halfedge_descriptor> hs;
  for (halfedge_descriptor h : halfedges(mesh))
    if (CGAL::is_border(h, mesh)) hs.push_back(h);
  return hs;
}

static std::vector<std::pair<halfedge_descriptor, halfedge_descriptor>>
collect_pairs_restricted(Mesh& mesh, const std::vector<halfedge_descriptor>& candidate_halfedges)
{
  std::vector<std::pair<halfedge_descriptor, halfedge_descriptor>> pairs;
  PMP::internal::Default_halfedges_keeper<Mesh> hd_kpr;
  PMP::internal::collect_duplicated_stitchable_boundary_edges(
      candidate_halfedges, mesh, hd_kpr, false /*per_cc*/, std::back_inserter(pairs),
      CGAL::parameters::default_values());
  return pairs;
}

struct Result {
  std::size_t v_before, v_after, f_before, f_after;
  std::size_t be_before, bl_before, be_after, bl_after;
  std::size_t nmv_before, nmv_after;
  std::size_t nc_before, nc_after;
  std::size_t si_before, si_after;
  std::size_t n_stitched;
  std::size_t n_moved;
  double max_disp;
};

static Result run_case(Mesh mesh, const std::vector<halfedge_descriptor>& candidate_halfedges, bool unused_flag)
{
  (void)unused_flag;
  Result r{};
  r.v_before = num_vertices(mesh); r.f_before = num_faces(mesh);
  count_boundary(mesh, r.be_before, r.bl_before);
  r.nmv_before = count_non_manifold_vertices(mesh);
  r.nc_before = count_components(mesh);
  std::vector<std::pair<face_descriptor,face_descriptor>> si0;
  PMP::self_intersections(mesh, std::back_inserter(si0));
  r.si_before = si0.size();

  std::vector<Point_3> original(mesh.points().begin(), mesh.points().end());

  std::cout << "  [run_case] before pair collection, be_before=" << r.be_before << std::endl << std::flush;
  auto pairs = collect_pairs_restricted(mesh, candidate_halfedges);
  std::cout << "  [run_case] collected " << pairs.size() << " candidate pairs, calling explicit-pairs stitch_borders" << std::endl << std::flush;
  r.n_stitched = PMP::stitch_borders(mesh, pairs);
  std::cout << "  [run_case] after stitch_borders call, n_stitched=" << r.n_stitched << std::endl << std::flush;

  r.v_after = num_vertices(mesh); r.f_after = num_faces(mesh);
  count_boundary(mesh, r.be_after, r.bl_after);
  r.nmv_after = count_non_manifold_vertices(mesh);
  r.nc_after = count_components(mesh);
  std::vector<std::pair<face_descriptor,face_descriptor>> si1;
  PMP::self_intersections(mesh, std::back_inserter(si1));
  r.si_after = si1.size();

  r.n_moved = 0; r.max_disp = 0.0;
  for (std::size_t i = 0; i < original.size(); ++i) {
    vertex_descriptor v(static_cast<std::uint32_t>(i));
    if (mesh.is_removed(v)) continue;
    double d = std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(v), original[i])));
    if (d > 0.0) { ++r.n_moved; if (d > r.max_disp) r.max_disp = d; }
  }
  return r;
}

static void print_result(const std::string& name, const Result& r, int expected_incident_faces_on_seam)
{
  std::cout << "\n--- " << name << " ---\n";
  std::cout << "n_stitched=" << r.n_stitched << "\n";
  std::cout << "vertices: " << r.v_before << " -> " << r.v_after << " (removed=" << (r.v_before - r.v_after) << ")\n";
  std::cout << "faces: " << r.f_before << " -> " << r.f_after << "\n";
  std::cout << "boundary_edges: " << r.be_before << " -> " << r.be_after << "\n";
  std::cout << "boundary_loops: " << r.bl_before << " -> " << r.bl_after << "\n";
  std::cout << "non_manifold_vertices: " << r.nmv_before << " -> " << r.nmv_after << "\n";
  std::cout << "components: " << r.nc_before << " -> " << r.nc_after << "\n";
  std::cout << "self_intersecting_pairs: " << r.si_before << " -> " << r.si_after << "\n";
  std::cout << "vertex_displacement: n_moved=" << r.n_moved << " max_disp=" << r.max_disp << "\n";
  std::cout << "PASS_CRITERIA_CHECK: nmv_after==0 -> " << (r.nmv_after == 0 ? "PASS" : "FAIL")
            << " ; n_moved==0 -> " << (r.n_moved == 0 ? "PASS" : "FAIL") << "\n";
}

int main()
{
  std::cout.precision(17);

  // ===== CASE 1: two duplicated boundary chains, SAME traversal direction =====
  // Two 1x4 triangle strips (top half y in [0,1], bottom half y in [-1,0]) sharing a 5-vertex
  // seam at y=0, each strip triangulated with the SAME winding convention. Seam vertices are
  // DUPLICATED (distinct indices, identical positions) between the two strips.
  {
    Mesh mesh;
    std::vector<vertex_descriptor> seamA, seamB, top, bot;
    for (int i = 0; i <= 4; ++i) seamA.push_back(mesh.add_vertex(Point_3(i, 0, 0)));
    for (int i = 0; i <= 4; ++i) top.push_back(mesh.add_vertex(Point_3(i, 1, 0)));
    for (int i = 0; i <= 4; ++i) seamB.push_back(mesh.add_vertex(Point_3(i, 0, 0))); // duplicate positions
    for (int i = 0; i <= 4; ++i) bot.push_back(mesh.add_vertex(Point_3(i, -1, 0)));
    // top strip: CCW when viewed from +Z, quad(seamA[i],seamA[i+1],top[i+1],top[i])
    for (int i = 0; i < 4; ++i) {
      mesh.add_face(seamA[i], seamA[i+1], top[i+1]);
      mesh.add_face(seamA[i], top[i+1], top[i]);
    }
    // bottom strip: SAME winding convention (also CCW from +Z), quad(bot[i],bot[i+1],seamB[i+1],seamB[i])
    for (int i = 0; i < 4; ++i) {
      mesh.add_face(bot[i], bot[i+1], seamB[i+1]);
      mesh.add_face(bot[i], seamB[i+1], seamB[i]);
    }
    std::cout << "CASE 1: mesh built, n_verts=" << num_vertices(mesh) << " n_faces=" << num_faces(mesh) << std::endl;
    std::vector<halfedge_descriptor> reps = all_border_halfedges(mesh);
    std::cout << "CASE 1: n_border_reps=" << reps.size() << std::endl;
    Result r = run_case(mesh, reps, true);
    print_result("CASE 1: same-direction duplicated chain (5 seam verts, 2 disconnected strips)", r, 2);
    std::cout << "EXPECT: n_stitched close to 5 boundary edges of the seam, 2 components -> 1, "
                 "boundary loops reduced, 0 new non-manifold, 0 displacement\n" << std::flush;
  }

  // ===== CASE 2: two duplicated boundary chains, OPPOSITE traversal direction =====
  // Same geometry and SAME winding convention as case 1 (both patches consistently +Z-facing, so
  // this is a physically valid glue-able pair), but seamB's vertex ARRAY INDEX order is reversed
  // relative to seamA's (seamB[i] sits at the position seamA[4-i] occupies) - i.e. the two
  // duplicated chains visit the shared physical positions in the OPPOSITE order when walked by
  // index. This is the genuine "reversed traversal direction" case, as distinct from case 1's
  // "same traversal direction" case, while keeping orientation consistent (unlike a first attempt
  // that flipped the whole patch's winding and broke consistency - that produced 0 matches, which
  // is a separate, correct "inconsistent orientation is correctly never stitched" finding, not
  // what this case is meant to test).
  {
    Mesh mesh;
    std::vector<vertex_descriptor> seamA, seamB, top, bot;
    for (int i = 0; i <= 4; ++i) seamA.push_back(mesh.add_vertex(Point_3(i, 0, 0)));
    for (int i = 0; i <= 4; ++i) top.push_back(mesh.add_vertex(Point_3(i, 1, 0)));
    for (int i = 0; i <= 4; ++i) seamB.push_back(mesh.add_vertex(Point_3(4 - i, 0, 0))); // reversed index order
    for (int i = 0; i <= 4; ++i) bot.push_back(mesh.add_vertex(Point_3(4 - i, -1, 0))); // mirrored too, so the strip stays a valid non-crossing quad strip
    for (int i = 0; i < 4; ++i) {
      mesh.add_face(seamA[i], seamA[i+1], top[i+1]);
      mesh.add_face(seamA[i], top[i+1], top[i]);
    }
    // SAME winding convention as case 1's bottom strip - only seamB's index-to-position mapping differs
    for (int i = 0; i < 4; ++i) {
      mesh.add_face(bot[i], bot[i+1], seamB[i+1]);
      mesh.add_face(bot[i], seamB[i+1], seamB[i]);
    }
    std::vector<halfedge_descriptor> reps = all_border_halfedges(mesh);
    Result r = run_case(mesh, reps, true);
    print_result("CASE 2: opposite-direction (reversed index order) duplicated chain", r, 2);
    std::cout << "EXPECT: same qualitative outcome as case 1 - stitching succeeds regardless of "
                 "which index order the duplicate chain happens to use, since the matcher works on "
                 "geometric position, not array index\n";
  }

  // ===== CASE 3: closed duplicated seam (whole hexagonal disk boundary duplicated) =====
  // Two independent hexagonal fans (center + 6 rim triangles) sharing the SAME 6 rim positions
  // (duplicated vertices) - stitching the closed rim loop should fully close the boundary.
  {
    Mesh mesh;
    std::vector<Point_3> rim_pos;
    for (int i = 0; i < 6; ++i) {
      double a = i * M_PI / 3.0;
      rim_pos.push_back(Point_3(std::cos(a), std::sin(a), 0));
    }
    vertex_descriptor c1 = mesh.add_vertex(Point_3(0, 0, 0.3));
    std::vector<vertex_descriptor> rim1;
    for (auto& p : rim_pos) rim1.push_back(mesh.add_vertex(p));
    for (int i = 0; i < 6; ++i) mesh.add_face(c1, rim1[i], rim1[(i+1)%6]);

    vertex_descriptor c2 = mesh.add_vertex(Point_3(0, 0, -0.3));
    std::vector<vertex_descriptor> rim2;
    for (auto& p : rim_pos) rim2.push_back(mesh.add_vertex(p)); // duplicate positions
    for (int i = 0; i < 6; ++i) mesh.add_face(c2, rim2[(i+1)%6], rim2[i]); // opposite winding (faces -Z cap's normal outward)

    std::vector<halfedge_descriptor> reps = all_border_halfedges(mesh);
    Result r = run_case(mesh, reps, true);
    print_result("CASE 3: closed duplicated seam (two hex fans forming a bipyramid)", r, 2);
    std::cout << "EXPECT: n_stitched==6 (full rim), boundary_loops after == 0 (fully closed), "
                 "2 components -> 1, resulting solid is closed (bipyramid)\n";
  }

  // ===== CASE 4: intentionally incorrect / non-matching pairing =====
  // A lone triangle with no geometric duplicate anywhere in the mesh, combined with case 1's
  // setup - passing ITS representative alongside should result in NO spurious stitch for that
  // loop (safety property: the restricted-range API must not force a bad match).
  {
    Mesh mesh;
    // case-1-like valid pair
    std::vector<vertex_descriptor> seamA, seamB, top, bot;
    for (int i = 0; i <= 2; ++i) seamA.push_back(mesh.add_vertex(Point_3(i, 0, 0)));
    for (int i = 0; i <= 2; ++i) top.push_back(mesh.add_vertex(Point_3(i, 1, 0)));
    for (int i = 0; i <= 2; ++i) seamB.push_back(mesh.add_vertex(Point_3(i, 0, 0)));
    for (int i = 0; i <= 2; ++i) bot.push_back(mesh.add_vertex(Point_3(i, -1, 0)));
    for (int i = 0; i < 2; ++i) {
      mesh.add_face(seamA[i], seamA[i+1], top[i+1]);
      mesh.add_face(seamA[i], top[i+1], top[i]);
      mesh.add_face(bot[i], bot[i+1], seamB[i+1]);
      mesh.add_face(bot[i], seamB[i+1], seamB[i]);
    }
    // an UNRELATED, non-duplicated lone triangle far away, with its own 3 border edges
    vertex_descriptor x1 = mesh.add_vertex(Point_3(100, 100, 100));
    vertex_descriptor x2 = mesh.add_vertex(Point_3(101, 100, 100));
    vertex_descriptor x3 = mesh.add_vertex(Point_3(100, 101, 100));
    mesh.add_face(x1, x2, x3);

    std::vector<halfedge_descriptor> reps = all_border_halfedges(mesh); // includes the lone triangle's loop too
    Result r = run_case(mesh, reps, true);
    print_result("CASE 4: valid pair + an intentionally non-matching lone triangle in the candidate set", r, 2);
    std::cout << "EXPECT: the valid seam still stitches; the lone triangle's 3 border edges remain "
                 "UNTOUCHED (no match exists for them) - demonstrates the API safely no-ops on "
                 "non-matching candidates rather than forcing an incorrect merge\n";
  }

  return 0;
}
