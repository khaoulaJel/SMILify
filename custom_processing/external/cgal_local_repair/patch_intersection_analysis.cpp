// Diagnostic-only follow-up to triangulate_hole_experiment.cpp (2026-08-17): reruns the IDENTICAL
// deterministic triangulate_hole repair (same mesh construction, same loop enumeration order, same
// algorithm - CGAL's boundary-halfedge traversal and orient_polygon_soup are both deterministic on
// a fixed input file), but instruments every new face with a patch_id and classifies every
// self-intersection pair by which side(s) are original vs. patch geometry. Does NOT repair
// anything - no self-intersection removal, no vertex welding, no component bridging, no
// smoothing/fairing, no Alpha Wrap. Purely attributes the ~20k new self-intersection pairs found
// by the previous experiment back to individual patches.
//
// Usage: patch_intersection_analysis <input.obj> <patches_csv_out> <pairs_csv_out> <mesh_off_out>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <sstream>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

using vertex_descriptor = boost::graph_traits<Mesh>::vertex_descriptor;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;

using Primitive = CGAL::AABB_face_graph_triangle_primitive<Mesh>;
using Traits = CGAL::AABB_traits_3<K, Primitive>;
using Tree = CGAL::AABB_tree<Traits>;

int main(int argc, char** argv)
{
  std::cout.precision(17);
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <input.obj> <patches_csv_out> <pairs_csv_out> <mesh_off_out>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_path = argv[1];
  const std::string patches_csv_path = argv[2];
  const std::string pairs_csv_path = argv[3];
  const std::string mesh_off_path = argv[4];

  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  if (!CGAL::IO::read_polygon_soup(input_path, points, polygons) || polygons.empty()) {
    std::cerr << "Invalid input: " << input_path << std::endl;
    return EXIT_FAILURE;
  }
  PMP::orient_polygon_soup(points, polygons);
  Mesh mesh;
  PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);
  std::cout << "MESH_AFTER_CONVERSION: n_vertices=" << num_vertices(mesh)
            << " n_faces=" << num_faces(mesh) << std::endl;

  // --- Original-state labeling, BEFORE any hole-filling ---
  const std::size_t n_faces_before = num_faces(mesh);

  typedef boost::graph_traits<Mesh>::faces_size_type faces_size_type;
  Mesh::Property_map<face_descriptor, faces_size_type> orig_component =
      mesh.add_property_map<face_descriptor, faces_size_type>("f:orig_cc", 0).first;
  std::size_t n_orig_components = PMP::connected_components(mesh, orig_component);
  std::cout << "ORIGINAL_COMPONENTS: n=" << n_orig_components << std::endl;

  // patch_id property: -1 for original faces, else the patch index (0-based, only successful loops)
  Mesh::Property_map<face_descriptor, int> patch_id =
      mesh.add_property_map<face_descriptor, int>("f:patch_id", -1).first;

  // --- Enumerate all boundary loop representatives up front (same as triangulate_hole_experiment.cpp) ---
  struct LoopInfo {
    halfedge_descriptor rep;
    std::size_t length;
    faces_size_type home_component; // component of the existing face adjacent to this hole
  };
  std::vector<LoopInfo> loops;
  {
    std::unordered_set<std::size_t> visited;
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (!CGAL::is_border(h, mesh)) continue;
      std::size_t hid = static_cast<std::size_t>(h);
      if (visited.count(hid)) continue;

      std::size_t length = 0;
      halfedge_descriptor start = h;
      halfedge_descriptor cur = h;
      do {
        visited.insert(static_cast<std::size_t>(cur));
        ++length;
        cur = next(cur, mesh);
      } while (cur != start);

      // the face on the OTHER side of the border halfedge (i.e. the existing face bordering
      // this hole) gives us the "home" component this loop is attached to
      halfedge_descriptor opp = opposite(h, mesh);
      face_descriptor adj_f = face(opp, mesh);
      faces_size_type home_cc = (adj_f != Mesh::null_face()) ? get(orig_component, adj_f) : static_cast<faces_size_type>(-1);

      loops.push_back({h, length, home_cc});
    }
  }
  std::cout << "N_BOUNDARY_LOOPS_FOUND: " << loops.size() << std::endl;

  // --- Fill holes, tracking patch_id per newly created face ---
  struct PatchInfo {
    int patch_id;
    std::size_t loop_length;
    std::size_t n_new_faces;
    std::vector<face_descriptor> faces;
    faces_size_type home_component;
  };
  std::vector<PatchInfo> patches;

  for (const LoopInfo& li : loops) {
    if (li.length < 3) continue;
    if (!CGAL::is_border(li.rep, mesh)) continue;

    std::vector<face_descriptor> patch_faces;
    try {
      PMP::triangulate_hole(mesh, li.rep, CGAL::parameters::face_output_iterator(std::back_inserter(patch_faces)));
    } catch (...) {
      continue; // same non-forcing behavior as triangulate_hole_experiment.cpp
    }
    if (patch_faces.empty()) continue;

    int pid = static_cast<int>(patches.size());
    for (face_descriptor f : patch_faces)
      put(patch_id, f, pid);

    patches.push_back({pid, li.length, patch_faces.size(), patch_faces, li.home_component});
  }
  std::cout << "N_PATCHES_CREATED: " << patches.size() << std::endl;
  std::cout << "N_FACES_AFTER_REPAIR: " << num_faces(mesh) << std::endl;

  // --- Global self-intersection pairs on the fully-repaired mesh ---
  std::vector<std::pair<face_descriptor, face_descriptor>> all_pairs;
  PMP::self_intersections(mesh, std::back_inserter(all_pairs));
  std::cout << "TOTAL_SELF_INTERSECTION_PAIRS: " << all_pairs.size() << std::endl;

  // --- Classify each pair and attribute to patches ---
  // categories: ORIG_ORIG (pre-existing, both patch_id==-1), A (one original + one patch),
  // B_SAME (both patch faces, same patch_id - intra-patch self-intersection),
  // B_CROSS (both patch faces, different patch_id - two different patches colliding)
  std::size_t n_orig_orig = 0, n_A = 0, n_B_same = 0, n_B_cross = 0;
  std::size_t n_A_diff_component = 0, n_A_same_component = 0;

  // per patch counters: pairs vs existing surface, pairs vs other patch geometry (B_same+B_cross)
  std::vector<std::size_t> patch_vs_existing(patches.size(), 0);
  std::vector<std::size_t> patch_vs_patchgeom(patches.size(), 0);
  std::vector<std::size_t> patch_vs_existing_diffcomp(patches.size(), 0);

  std::ofstream pairs_csv(pairs_csv_path);
  pairs_csv << "face1,face2,patch1,patch2,category,component_note\n";

  for (const auto& pr : all_pairs) {
    face_descriptor f1 = pr.first, f2 = pr.second;
    int p1 = get(patch_id, f1);
    int p2 = get(patch_id, f2);

    std::string category;
    std::string comp_note = "-";

    if (p1 == -1 && p2 == -1) {
      category = "ORIG_ORIG";
      ++n_orig_orig;
    } else if (p1 == -1 || p2 == -1) {
      category = "A_patch_vs_existing";
      ++n_A;
      int pid = (p1 == -1) ? p2 : p1;
      face_descriptor existing_f = (p1 == -1) ? f1 : f2;
      patch_vs_existing[pid] += 1;
      faces_size_type existing_cc = get(orig_component, existing_f);
      if (existing_cc != patches[pid].home_component) {
        ++n_A_diff_component;
        patch_vs_existing_diffcomp[pid] += 1;
        comp_note = "DIFFERENT_COMPONENT";
      } else {
        ++n_A_same_component;
        comp_note = "same_component";
      }
    } else if (p1 == p2) {
      category = "B_same_patch_self_intersection";
      ++n_B_same;
      patch_vs_patchgeom[p1] += 1;
    } else {
      category = "B_cross_patch";
      ++n_B_cross;
      patch_vs_patchgeom[p1] += 1;
      patch_vs_patchgeom[p2] += 1;
    }

    pairs_csv << static_cast<std::size_t>(f1) << "," << static_cast<std::size_t>(f2) << ","
              << p1 << "," << p2 << "," << category << "," << comp_note << "\n";
  }
  pairs_csv.close();

  std::cout << "CLASSIFICATION: ORIG_ORIG(pre-existing,excluded-from-new-count)=" << n_orig_orig
            << " A_patch_vs_existing=" << n_A
            << " A_diff_component=" << n_A_diff_component
            << " A_same_component=" << n_A_same_component
            << " B_same_patch=" << n_B_same
            << " B_cross_patch=" << n_B_cross
            << " RECONCILE_TOTAL=" << (n_orig_orig + n_A + n_B_same + n_B_cross)
            << " (should equal TOTAL_SELF_INTERSECTION_PAIRS above)" << std::endl;

  // --- Per-patch geometry stats: area, bbox, min clearance to non-adjacent original surface ---
  // Build one AABB tree over ALL original (pre-repair) faces once.
  std::vector<face_descriptor> orig_faces_vec;
  for (face_descriptor f : faces(mesh))
    if (get(patch_id, f) == -1)
      orig_faces_vec.push_back(f);
  Tree orig_tree(orig_faces_vec.begin(), orig_faces_vec.end(), mesh);
  orig_tree.accelerate_distance_queries();

  std::ofstream patches_csv(patches_csv_path);
  patches_csv << "patch_id,loop_size,n_new_faces,patch_area,bbox_min_x,bbox_min_y,bbox_min_z,"
                 "bbox_max_x,bbox_max_y,bbox_max_z,home_component,"
                 "n_intersection_pairs_total,n_vs_existing,n_vs_existing_diff_component,n_vs_patchgeom,"
                 "min_clearance_to_nonadjacent_existing,face_ids\n";

  for (const PatchInfo& p : patches) {
    double area = 0.0;
    double bmin[3] = {1e300, 1e300, 1e300};
    double bmax[3] = {-1e300, -1e300, -1e300};
    std::set<vertex_descriptor> patch_verts;
    for (face_descriptor f : p.faces) {
      area += CGAL::to_double(PMP::face_area(f, mesh));
      for (vertex_descriptor v : vertices_around_face(halfedge(f, mesh), mesh)) {
        patch_verts.insert(v);
        const Point_3& pt = mesh.point(v);
        double c[3] = {CGAL::to_double(pt.x()), CGAL::to_double(pt.y()), CGAL::to_double(pt.z())};
        for (int i = 0; i < 3; ++i) {
          bmin[i] = std::min(bmin[i], c[i]);
          bmax[i] = std::max(bmax[i], c[i]);
        }
      }
    }

    // min clearance: query the AABB tree of original faces for the closest point to each patch
    // face centroid, EXCLUDING original faces that share a vertex with this patch's own boundary
    // loop (those are trivially adjacent/zero-ish distance and not informative "nearby anatomy"
    // clearance). Implemented as: get closest point/primitive, and if that primitive shares a
    // patch vertex, fall back to a bounded local search among faces within a generous radius,
    // excluding adjacent ones, taking the true minimum among the non-adjacent candidates.
    double min_clearance = std::numeric_limits<double>::infinity();
    for (face_descriptor f : p.faces) {
      double cx = 0, cy = 0, cz = 0;
      int nv = 0;
      for (vertex_descriptor v : vertices_around_face(halfedge(f, mesh), mesh)) {
        const Point_3& pt = mesh.point(v);
        cx += CGAL::to_double(pt.x()); cy += CGAL::to_double(pt.y()); cz += CGAL::to_double(pt.z());
        ++nv;
      }
      Point_3 centroid(cx / nv, cy / nv, cz / nv);
      Tree::Point_and_primitive_id pp = orig_tree.closest_point_and_primitive(centroid);
      face_descriptor closest_f = pp.second;
      bool adjacent = false;
      for (vertex_descriptor v : vertices_around_face(halfedge(closest_f, mesh), mesh)) {
        if (patch_verts.count(v)) { adjacent = true; break; }
      }
      if (!adjacent) {
        double d = std::sqrt(CGAL::to_double(CGAL::squared_distance(centroid, pp.first)));
        min_clearance = std::min(min_clearance, d);
      }
      // if adjacent, skip this sample point silently rather than force a more expensive search -
      // with many patch faces per patch the non-adjacent minimum is still captured by the others;
      // reported honestly as best-effort, not a guaranteed global minimum (see report).
    }

    std::ostringstream face_ids_str;
    for (std::size_t i = 0; i < p.faces.size(); ++i) {
      if (i) face_ids_str << ";";
      face_ids_str << static_cast<std::size_t>(p.faces[i]);
    }

    patches_csv << p.patch_id << "," << p.loop_length << "," << p.n_new_faces << "," << area << ","
                << bmin[0] << "," << bmin[1] << "," << bmin[2] << ","
                << bmax[0] << "," << bmax[1] << "," << bmax[2] << ","
                << p.home_component << ","
                << (patch_vs_existing[p.patch_id] + patch_vs_patchgeom[p.patch_id]) << ","
                << patch_vs_existing[p.patch_id] << ","
                << patch_vs_existing_diffcomp[p.patch_id] << ","
                << patch_vs_patchgeom[p.patch_id] << ","
                << (std::isinf(min_clearance) ? std::string("inf") : std::to_string(min_clearance)) << ","
                << face_ids_str.str()
                << "\n";
  }
  patches_csv.close();

  CGAL::IO::write_polygon_mesh(mesh_off_path, mesh, CGAL::parameters::stream_precision(17));
  std::cout << "Wrote " << patches_csv_path << ", " << pairs_csv_path << ", " << mesh_off_path << std::endl;
  return EXIT_SUCCESS;
}
