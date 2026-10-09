# Same build pattern as cgal_alpha_wrap/CGAL/Alpha_wrap_3/examples/Alpha_wrap_3/CMakeLists.txt -
# find_package(CGAL) + create_single_source_cgal_program per .cpp. Points at the ALREADY-CLONED
# CGAL checkout under cgal_alpha_wrap/CGAL (via -DCGAL_DIR=... at configure time) - no re-clone.
cmake_minimum_required(VERSION 3.12...3.31)
project(CGAL_Local_Repair_Experiment)

find_package(CGAL REQUIRED)

create_single_source_cgal_program("triangulate_hole_experiment.cpp")
create_single_source_cgal_program("patch_intersection_analysis.cpp")
create_single_source_cgal_program("boundary_loop_analysis.cpp")
create_single_source_cgal_program("experimentA_planar_hole_comparison.cpp")
create_single_source_cgal_program("export_tangled_loop_geometry.cpp")
create_single_source_cgal_program("duplicate_seam_characterization.cpp")
create_single_source_cgal_program("duplicate_seam_stitch_experiment.cpp")
create_single_source_cgal_program("synthetic_stitch_validation.cpp")
create_single_source_cgal_program("tiny_stitch_test.cpp")
create_single_source_cgal_program("explicit_A_only_stitch.cpp")
create_single_source_cgal_program("multiplicity_cluster_analysis.cpp")
create_single_source_cgal_program("multiway_chain_extension.cpp")
create_single_source_cgal_program("giant_chain_diagnosis.cpp")
create_single_source_cgal_program("tangled_sheet_analysis.cpp")
create_single_source_cgal_program("sheet_split_diagnosis.cpp")
create_single_source_cgal_program("fusion_graph_diagnosis.cpp")
create_single_source_cgal_program("intersection_graph_diagnosis.cpp")
