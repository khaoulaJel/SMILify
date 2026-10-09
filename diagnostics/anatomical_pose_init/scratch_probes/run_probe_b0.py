import sys, os
sys.path.insert(0, os.path.abspath("diagnostics/morphometrics"))
sys.path.insert(0, os.path.abspath("diagnostics/registration_interventions"))
import probe_B0_geom_init_validation as p
p.CONDITIONS = [("pose25", "synth_clean")]
p.main()
