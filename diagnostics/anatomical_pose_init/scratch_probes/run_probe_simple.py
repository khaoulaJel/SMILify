import sys, os
sys.path.insert(0, os.path.abspath("diagnostics/morphometrics"))
sys.path.insert(0, os.path.abspath("diagnostics/registration_interventions"))
sys.path.insert(0, os.path.abspath("diagnostics/anatomical_pose_init"))
import probe_B0_geom_init_validation as p
from simple_leg_heuristic import estimate_leg_rotations
p.init_joint_rot_for_specimen = estimate_leg_rotations
p.CONDITIONS = [("pose25", "synth_clean")]
p.OUT = "diagnostics/anatomical_pose_init/scratch_probes/out_simple"
p.main()
