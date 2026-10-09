// Priority 1 step 5 (2026-08-17): given seed pairs (v1,v2) that were identified as the CORRECT
// disambiguated pairing within a >2-way duplicate-position cluster (via full one-ring neighbor-
// identity matching in Python, from multiplicity_cluster_analysis.cpp output), extend each seed
// outward along its matching border chain (same algorithm as duplicate_seam_characterization.cpp's
// reversed/same-direction chain walk) to recover the FULL set of vertex-pair correspondences along
// that seam - a lone (v1,v2) vertex pair is not itself border-adjacent to anything, so the actual
// stitchable edges live between v1's chain neighbors and v2's chain neighbors.
//
// Diagnostic/export only - does not stitch.
//
// Usage: multiway_chain_extension <input.obj> <seed_pairs_csv> <extended_pairs_csv_out>
//   seed_pairs_csv: v1,v2 columns (vN format), one seed per line, header required.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <cstdint>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;
typedef boost::graph_traits<Mesh>::faces_size_type faces_size_type;

static halfedge_descriptor unique_incoming_border_halfedge(const Mesh& mesh, vertex_descriptor v)
{
  halfedge_descriptor found = Mesh::null_halfedge(); int count = 0;
  for (halfedge_descriptor h : halfedges_around_target(halfedge(v, mesh), mesh))
    if (CGAL::is_border(h, mesh)) { found = h; ++count; }
  return (count == 1) ? found : Mesh::null_halfedge();
}

// Walk both the "reversed" and "same-direction" hypotheses outward from (v1,v2), extending as far
// as consecutive positions keep matching, collecting every (v1_k, v2_k) pair encountered along the
// way (excluding the seed itself, which is not border-adjacent to its partner). Stops the moment a
// step's position stops matching, a wrap-around occurs, or a non-manifold/ambiguous vertex (more
// than one incoming border halfedge) is met - extension must not silently cross into unrelated
// territory.
static std::vector<std::pair<vertex_descriptor,vertex_descriptor>>
extend_chain(const Mesh& mesh, vertex_descriptor v1, vertex_descriptor v2, int max_steps)
{
  std::vector<std::pair<vertex_descriptor,vertex_descriptor>> out;
  halfedge_descriptor h1 = unique_incoming_border_halfedge(mesh, v1);
  halfedge_descriptor h2 = unique_incoming_border_halfedge(mesh, v2);
  if (h1 == Mesh::null_halfedge() || h2 == Mesh::null_halfedge()) return out;

  // direction A: predecessor(v1) vs successor(v2) - extends "backward" from v1 / "forward" from v2
  {
    halfedge_descriptor cur1 = h1;
    halfedge_descriptor cur2h = next(h2, mesh);
    for (int step = 0; step < max_steps; ++step) {
      vertex_descriptor pred1 = source(cur1, mesh);
      vertex_descriptor succ2 = target(cur2h, mesh);
      if (mesh.point(pred1) != mesh.point(succ2)) break;
      out.emplace_back(pred1, succ2);
      halfedge_descriptor p1 = prev(cur1, mesh), n2 = next(cur2h, mesh);
      if (!CGAL::is_border(p1, mesh) || !CGAL::is_border(n2, mesh)) break;
      if (unique_incoming_border_halfedge(mesh, pred1) == Mesh::null_halfedge()) break;
      if (unique_incoming_border_halfedge(mesh, succ2) == Mesh::null_halfedge()) break;
      cur1 = p1; cur2h = n2;
      if (source(cur1, mesh) == v2 || target(cur2h, mesh) == v1) break;
    }
  }
  // direction B: successor(v1) vs predecessor(v2) - the other rotational direction, needed to
  // recover BOTH of a seed's matching neighbors, not just the one direction A happens to walk.
  {
    halfedge_descriptor cur1 = next(h1, mesh);
    halfedge_descriptor cur2h = h2;
    for (int step = 0; step < max_steps; ++step) {
      vertex_descriptor succ1 = target(cur1, mesh);
      vertex_descriptor pred2 = source(cur2h, mesh);
      if (mesh.point(succ1) != mesh.point(pred2)) break;
      out.emplace_back(succ1, pred2);
      halfedge_descriptor n1 = next(cur1, mesh), p2 = prev(cur2h, mesh);
      if (!CGAL::is_border(n1, mesh) || !CGAL::is_border(p2, mesh)) break;
      if (unique_incoming_border_halfedge(mesh, succ1) == Mesh::null_halfedge()) break;
      if (unique_incoming_border_halfedge(mesh, pred2) == Mesh::null_halfedge()) break;
      cur1 = n1; cur2h = p2;
      if (target(cur1, mesh) == v2 || source(cur2h, mesh) == v1) break;
    }
  }
  return out;
}

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <seed_pairs_csv> <extended_pairs_csv_out>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string seeds_path = argv[2];
  const std::string out_path = argv[3];

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
  auto vcomp = [&](vertex_descriptor v) -> faces_size_type {
    for (face_descriptor f : faces_around_target(halfedge(v, mesh), mesh))
      if (f != Mesh::null_face()) return get(orig_component, f);
    return static_cast<faces_size_type>(-1);
  };

  std::vector<std::pair<vertex_descriptor,vertex_descriptor>> seeds;
  { std::ifstream in(seeds_path); std::string header; std::getline(in, header);
    std::string line;
    auto parse_v = [](const std::string& s) { return static_cast<std::uint32_t>(std::stoul(s.substr(1))); };
    while (std::getline(in, line)) {
      std::stringstream ss(line); std::vector<std::string> f; std::string tok;
      while (std::getline(ss, tok, ',')) f.push_back(tok);
      if (f.size() < 2) continue;
      seeds.emplace_back(vertex_descriptor(parse_v(f[0])), vertex_descriptor(parse_v(f[1])));
    }
  }
  std::cout << "N_SEEDS: " << seeds.size() << std::endl;

  std::ofstream csv(out_path);
  csv << "v1,v2,distance,deg_v1,deg_v2,is_border_v1,is_border_v2,cc_v1,cc_v2,same_component,"
         "normal_dot,chain_match_length,chain_direction,one_ring_position_match_fraction,"
         "near_genuine_near_touch,near_touch_min_dist,in_mindist0_loop_v1,in_mindist0_loop_v2,"
         "classification,classification_reason\n";
  std::size_t total_extended = 0;
  for (auto& seed : seeds) {
    auto chain = extend_chain(mesh, seed.first, seed.second, 500);
    std::cout << "  seed v" << seed.first << ",v" << seed.second << " -> extended chain length=" << chain.size() << std::endl;
    for (auto& pr : chain) {
      vertex_descriptor a = pr.first, b = pr.second;
      csv << a << "," << b << ",0,0,0,True,True," << vcomp(a) << "," << vcomp(b) << ","
          << (vcomp(a)==vcomp(b) ? "True" : "False") << ",0," << (int)chain.size() << ",multiway_extended,1,False,999,False,False,"
          << "A,multiway_chain_extension\n";
      ++total_extended;
    }
  }
  csv.close();
  std::cout << "TOTAL_EXTENDED_PAIRS: " << total_extended << std::endl;
  std::cout << "Wrote " << out_path << std::endl;
  return EXIT_SUCCESS;
}
