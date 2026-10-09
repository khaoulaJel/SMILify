"""Arm H input: the project's existing learned leg-pose initialiser (bench50_G1d protocol,
`learned_init_lowpose05` checkpoint) applied, unchanged, to the benchmark meshes. Only the input
directory and output path differ from `generate_bench50_learned_init.py`."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics/anatomical_pose_init"))
import generate_bench50_learned_init as g  # noqa: E402

g.BENCH50_DIR = sys.argv[1]
g.OUT = sys.argv[2]
g.main()
