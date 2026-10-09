// Diagnostic-only (2026-08-17): characterizes every exact duplicate-position vertex pair left in
// the mesh after orient_polygon_soup + polygon_soup_to_polygon_mesh, to determine whether each is
// a genuine same-surface duplicate seam (safe to stitch) or two anatomically distinct surfaces
// that merely coincide/touch at a point (dangerous to stitch). Does NOT stitch or modify the mesh
// in this diagnostic pass (see duplicate_seam_stitch_experiment.cpp for the controlled mini-test
// on the explicitly-classified subset).
//
// Usage: duplicate_seam_characterization <input.obj> <pairs_csv_out> <specimen_tag>
//   specimen_tag: "strumigenys" (enables the 8-near-touch-candidate proximity check) or "mayriella"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
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
static Vec3 cross(const Vec3&a,const Vec3&b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static Vec3 P(const Point_3& p) { return {CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())}; }

// 8 genuine anatomical near-touch candidates (Strumigenys_alberti), reused verbatim from earlier
// experiments in this session - midpoint of the two flagged points, used only for a proximity check.
static const std::vector<Vec3> NEAR_TOUCH_MIDPOINTS = {
  {(-553.329-550.746)/2, (373.323+376.307)/2, (162.388+163.258)/2},
  {(500.403+501.690)/2, (-248.654-248.217)/2, (17.331+21.930)/2},
  {(556.372+559.884)/2, (15.116+18.150)/2, (-344.669-346.722)/2},
  {(90.885+93.982)/2, (-464.780-461.545)/2, (-268.590-265.883)/2},
  {(358.671+358.083)/2, (-281.712-276.261)/2, (-277.352-277.261)/2},
  {(-89.012-87.956)/2, (8.958+13.613)/2, (121.246+124.310)/2},
  {(-552.949-550.746)/2, (376.465+376.307)/2, (168.640+163.258)/2},
  {(-528.255-526.154)/2, (426.411+421.787)/2, (142.535+145.641)/2},
};

// SIMPLE + min_dist_to_nonadjacent_surface==0 loop_ids identified in the prior turn's
// boundary_loop_analysis.cpp output (enumeration order is deterministic/identical given the same
// input file - verified in that turn via 0 join mismatches against patches.csv).
static const std::vector<std::size_t> STRUMIGENYS_MINDIST0_LOOPS = {34, 37, 42};
static const std::vector<std::size_t> MAYRIELLA_MINDIST0_LOOPS = {17, 26, 41, 164, 181, 184};

static Vec3 vertex_avg_normal(const Mesh& mesh, vertex_descriptor v)
{
  Vec3 n{0,0,0};
  for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh)) {
    if (f == Mesh::null_face()) continue;
    std::vector<Vec3> pts;
    for (vertex_descriptor fv : vertices_around_face(halfedge(f, mesh), mesh)) pts.push_back(P(mesh.point(fv)));
    if (pts.size() == 3) {
      Vec3 fn = cross(pts[1]-pts[0], pts[2]-pts[0]);
      double fl = fn.norm();
      if (fl > 1e-12) n = n + fn * (1.0/fl);
    }
  }
  double nl = n.norm();
  return nl > 1e-12 ? n * (1.0/nl) : n;
}

static int vertex_degree(const Mesh& mesh, vertex_descriptor v)
{
  int d = 0;
  for (halfedge_descriptor h : halfedges_around_target(halfedge(v, mesh), mesh)) ++d;
  return d;
}

static bool vertex_is_border(const Mesh& mesh, vertex_descriptor v)
{
  for (halfedge_descriptor h : halfedges_around_target(halfedge(v, mesh), mesh))
    if (CGAL::is_border(h, mesh)) return true;
  return false;
}

// returns the unique incoming border halfedge (target==v), or null_halfedge if v is not a
// (manifold) border vertex with exactly one such halfedge.
static halfedge_descriptor unique_incoming_border_halfedge(const Mesh& mesh, vertex_descriptor v)
{
  halfedge_descriptor found = Mesh::null_halfedge();
  int count = 0;
  for (halfedge_descriptor h : halfedges_around_target(halfedge(v, mesh), mesh)) {
    if (CGAL::is_border(h, mesh)) { found = h; ++count; }
  }
  return (count == 1) ? found : Mesh::null_halfedge();
}

// Walk outward from v1 (predecessor direction along its border loop) and from v2 (successor
// direction along its border loop) simultaneously, counting how many consecutive steps the
// visited vertex POSITIONS coincide - the "reversed matching chain" hypothesis (the natural
// pattern when a manifold strip is split into two boundary copies with opposite local winding).
// Returns the match length (0 if the pair itself is the only match, i.e. no chain continuation).
static int reversed_chain_match_length(const Mesh& mesh, vertex_descriptor v1, vertex_descriptor v2, int max_steps)
{
  halfedge_descriptor h1 = unique_incoming_border_halfedge(mesh, v1); // target==v1
  halfedge_descriptor h2 = unique_incoming_border_halfedge(mesh, v2); // target==v2
  if (h1 == Mesh::null_halfedge() || h2 == Mesh::null_halfedge()) return -1; // not simple border vertices

  int matches = 0;
  halfedge_descriptor cur1 = h1; // step backward (predecessor) via prev()
  halfedge_descriptor cur2h = next(h2, mesh); // step forward (successor) starting from v2's outgoing
  for (int step = 0; step < max_steps; ++step) {
    vertex_descriptor pred1 = source(cur1, mesh); // predecessor of v1's current chain position
    vertex_descriptor succ2 = target(cur2h, mesh); // successor of v2's current chain position
    Point_3 p1 = mesh.point(pred1), p2 = mesh.point(succ2);
    if (CGAL::squared_distance(p1, p2) != 0) break;
    ++matches;
    if (!CGAL::is_border(prev(cur1, mesh), mesh)) break;
    if (!CGAL::is_border(next(cur2h, mesh), mesh)) break;
    cur1 = prev(cur1, mesh);
    cur2h = next(cur2h, mesh);
    if (source(cur1, mesh) == v2 || target(cur2h, mesh) == v1) break; // wrapped around
  }
  return matches;
}

// Same idea, but the "same-direction" hypothesis: v1 and v2 are two copies of the same oriented
// boundary curve walked in the SAME direction (predecessor-vs-predecessor, successor-vs-successor).
// Missing this direction undercounts matches for splits that preserve local winding rather than
// reversing it - both directions must be tried before concluding "no chain continuation".
static int same_direction_chain_match_length(const Mesh& mesh, vertex_descriptor v1, vertex_descriptor v2, int max_steps)
{
  halfedge_descriptor h1 = unique_incoming_border_halfedge(mesh, v1);
  halfedge_descriptor h2 = unique_incoming_border_halfedge(mesh, v2);
  if (h1 == Mesh::null_halfedge() || h2 == Mesh::null_halfedge()) return -1;

  int fwd_matches = 0;
  halfedge_descriptor c1 = next(h1, mesh), c2 = next(h2, mesh); // successors
  for (int step = 0; step < max_steps; ++step) {
    if (CGAL::squared_distance(mesh.point(target(c1, mesh)), mesh.point(target(c2, mesh))) != 0) break;
    ++fwd_matches;
    if (!CGAL::is_border(next(c1, mesh), mesh) || !CGAL::is_border(next(c2, mesh), mesh)) break;
    c1 = next(c1, mesh); c2 = next(c2, mesh);
    if (target(c1, mesh) == v2 || target(c2, mesh) == v1) break;
  }
  int bwd_matches = 0;
  halfedge_descriptor d1 = h1, d2 = h2; // predecessors, via source()
  for (int step = 0; step < max_steps; ++step) {
    if (CGAL::squared_distance(mesh.point(source(d1, mesh)), mesh.point(source(d2, mesh))) != 0) break;
    ++bwd_matches;
    if (!CGAL::is_border(prev(d1, mesh), mesh) || !CGAL::is_border(prev(d2, mesh), mesh)) break;
    d1 = prev(d1, mesh); d2 = prev(d2, mesh);
    if (source(d1, mesh) == v2 || source(d2, mesh) == v1) break;
  }
  return fwd_matches + bwd_matches;
}

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <pairs_csv_out> <specimen_tag>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string csv_path = argv[2];
  const std::string specimen_tag = argv[3];

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

  // Loop enumeration IDENTICAL to boundary_loop_analysis.cpp, to recover the vertex-index sets of
  // the 9 known SIMPLE+min_dist=0 loops (their loop_id ordering is deterministic on this input).
  std::vector<std::vector<vertex_descriptor>> loops;
  {
    std::unordered_set<std::size_t> visited;
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (!CGAL::is_border(h, mesh)) continue;
      std::size_t hid = static_cast<std::size_t>(h);
      if (visited.count(hid)) continue;
      std::vector<vertex_descriptor> lv;
      halfedge_descriptor start = h, cur = h;
      do { visited.insert(static_cast<std::size_t>(cur)); lv.push_back(target(cur, mesh)); cur = next(cur, mesh); } while (cur != start);
      loops.push_back(lv);
    }
  }
  const std::vector<std::size_t>& target_loop_ids =
      (specimen_tag == "strumigenys") ? STRUMIGENYS_MINDIST0_LOOPS : MAYRIELLA_MINDIST0_LOOPS;
  std::set<vertex_descriptor> mindist0_loop_verts;
  for (std::size_t lid : target_loop_ids)
    if (lid < loops.size())
      for (vertex_descriptor v : loops[lid]) mindist0_loop_verts.insert(v);
  std::cout << "N_MINDIST0_LOOP_VERTS: " << mindist0_loop_verts.size() << " (from " << target_loop_ids.size() << " loops)" << std::endl;

  // Find exact duplicate-position vertex pairs via sort-by-position (exact, not KD-tree radius).
  std::vector<vertex_descriptor> verts(vertices(mesh).begin(), vertices(mesh).end());
  std::sort(verts.begin(), verts.end(), [&](vertex_descriptor a, vertex_descriptor b) {
    const Point_3& pa = mesh.point(a); const Point_3& pb = mesh.point(b);
    if (pa.x() != pb.x()) return pa.x() < pb.x();
    if (pa.y() != pb.y()) return pa.y() < pb.y();
    return pa.z() < pb.z();
  });

  std::ofstream csv(csv_path);
  csv << "v1,v2,distance,deg_v1,deg_v2,is_border_v1,is_border_v2,cc_v1,cc_v2,same_component,"
         "normal_dot,chain_match_length,chain_direction,one_ring_position_match_fraction,"
         "near_genuine_near_touch,near_touch_min_dist,in_mindist0_loop_v1,in_mindist0_loop_v2,"
         "classification,classification_reason\n";

  std::size_t n_pairs = 0;
  for (std::size_t i = 0; i + 1 < verts.size(); ++i) {
    vertex_descriptor v1 = verts[i], v2 = verts[i+1];
    if (mesh.point(v1) != mesh.point(v2)) continue; // only strictly consecutive exact duplicates paired once
    ++n_pairs;

    int deg1 = vertex_degree(mesh, v1), deg2 = vertex_degree(mesh, v2);
    bool border1 = vertex_is_border(mesh, v1), border2 = vertex_is_border(mesh, v2);
    faces_size_type cc1 = faces_size_type(-1), cc2 = faces_size_type(-1);
    for (face_descriptor f : faces_around_target(halfedge(v1, mesh), mesh)) if (f != Mesh::null_face()) { cc1 = get(orig_component, f); break; }
    for (face_descriptor f : faces_around_target(halfedge(v2, mesh), mesh)) if (f != Mesh::null_face()) { cc2 = get(orig_component, f); break; }
    bool same_component = (cc1 == cc2);

    Vec3 n1 = vertex_avg_normal(mesh, v1), n2 = vertex_avg_normal(mesh, v2);
    double normal_dot = n1.dot(n2);

    int chain_match = -1;
    std::string chain_direction = "n/a";
    if (border1 && border2) {
      int rev = reversed_chain_match_length(mesh, v1, v2, 200);
      int same = same_direction_chain_match_length(mesh, v1, v2, 200);
      if (rev >= same) { chain_match = rev; chain_direction = "reversed"; }
      else { chain_match = same; chain_direction = "same_direction"; }
    }

    // one-ring position-match fraction: for each neighbor of v1, is there a position-identical
    // neighbor of v2? (evidence the whole local patch, not just this one vertex, is duplicated)
    std::vector<Point_3> ring1, ring2;
    for (vertex_descriptor nb : vertices_around_target(halfedge(v1, mesh), mesh)) ring1.push_back(mesh.point(nb));
    for (vertex_descriptor nb : vertices_around_target(halfedge(v2, mesh), mesh)) ring2.push_back(mesh.point(nb));
    int n_matched = 0;
    for (const Point_3& p1 : ring1)
      for (const Point_3& p2 : ring2)
        if (p1 == p2) { ++n_matched; break; }
    double one_ring_frac = ring1.empty() ? 0.0 : double(n_matched) / ring1.size();

    Vec3 pos = P(mesh.point(v1));
    double near_touch_min_dist = std::numeric_limits<double>::infinity();
    if (specimen_tag == "strumigenys")
      for (const Vec3& c : NEAR_TOUCH_MIDPOINTS) near_touch_min_dist = std::min(near_touch_min_dist, (pos - c).norm());
    bool near_genuine = near_touch_min_dist < 20.0; // generous radius given near-touch gaps are ~4-6 units

    bool in_loop1 = mindist0_loop_verts.count(v1) > 0, in_loop2 = mindist0_loop_verts.count(v2) > 0;

    // --- classification (heuristic, documented) ---
    std::string cls, reason;
    if (!same_component) {
      cls = "C"; reason = "different_connected_components_never_bridge";
    } else if (near_genuine) {
      cls = "C"; reason = "coincides_with_genuine_anatomical_near_touch_candidate";
    } else if (border1 && border2 && chain_match >= 3) {
      cls = "A"; reason = "reversed_border_chain_matches_for_" + std::to_string(chain_match) + "_consecutive_steps";
    } else if (one_ring_frac >= 0.5) {
      cls = "A"; reason = "majority_of_one_ring_neighbors_position_duplicated";
    } else if (border1 && border2 && chain_match <= 1) {
      cls = "C"; reason = "border_chains_do_not_continue_matching_beyond_the_touch_point";
    } else {
      cls = "B"; reason = "mixed_or_insufficient_evidence";
    }

    csv << v1 << "," << v2 << "," << 0.0 << "," << deg1 << "," << deg2 << ","
        << (border1?"True":"False") << "," << (border2?"True":"False") << ","
        << cc1 << "," << cc2 << "," << (same_component?"True":"False") << ","
        << normal_dot << "," << chain_match << "," << chain_direction << "," << one_ring_frac << ","
        << (near_genuine?"True":"False") << "," << near_touch_min_dist << ","
        << (in_loop1?"True":"False") << "," << (in_loop2?"True":"False") << ","
        << cls << "," << reason << "\n";
  }
  csv.close();
  std::cout << "N_DUPLICATE_PAIRS: " << n_pairs << std::endl;
  std::cout << "Wrote " << csv_path << std::endl;
  return EXIT_SUCCESS;
}
