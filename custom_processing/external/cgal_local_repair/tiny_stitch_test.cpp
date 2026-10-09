#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <iostream>
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;
using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;

int main(int argc, char** argv) {
  Mesh mesh;
  auto A = mesh.add_vertex(Point_3(0,0,0));
  auto B = mesh.add_vertex(Point_3(1,0,0));
  auto C = mesh.add_vertex(Point_3(0,1,0));
  auto A2 = mesh.add_vertex(Point_3(0,0,0));
  auto B2 = mesh.add_vertex(Point_3(1,0,0));
  auto D = mesh.add_vertex(Point_3(1,-1,0));
  mesh.add_face(A,B,C);
  mesh.add_face(A2,D,B2); // opposite winding along AB edge
  std::cout << "built: verts=" << num_vertices(mesh) << " faces=" << num_faces(mesh) << std::endl << std::flush;

  if (argc > 1 && std::string(argv[1]) == "restricted") {
    std::vector<halfedge_descriptor> reps;
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (CGAL::is_border(h, mesh)) { reps.push_back(h); break; } // just grab first border halfedge per test simplicity - not correct but let's see if it hangs
    }
    std::cout << "calling restricted stitch with reps.size()=" << reps.size() << std::endl << std::flush;
    // properly collect one rep per cycle instead:
    reps.clear();
    std::vector<bool> visited(2*num_edges(mesh)+10, false);
    for (halfedge_descriptor h : halfedges(mesh)) {
      if (!CGAL::is_border(h,mesh)) continue;
      std::size_t hid = (std::size_t)h;
      if (hid < visited.size() && visited[hid]) continue;
      reps.push_back(h);
      halfedge_descriptor start=h, cur=h;
      do { std::size_t cid=(std::size_t)cur; if(cid<visited.size()) visited[cid]=true; cur=next(cur,mesh);} while(cur!=start);
    }
    std::cout << "n_cycle_reps=" << reps.size() << std::endl << std::flush;
    std::size_t n = PMP::stitch_borders(reps, mesh);
    std::cout << "restricted stitch done, n_stitched=" << n << std::endl << std::flush;
  } else {
    std::size_t n = PMP::stitch_borders(mesh);
    std::cout << "global stitch done, n_stitched=" << n << std::endl << std::flush;
  }
  std::cout << "after: verts=" << num_vertices(mesh) << " faces=" << num_faces(mesh) << std::endl;
  return 0;
}
