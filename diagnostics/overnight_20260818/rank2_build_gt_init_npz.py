"""Rank 2 (master prompt Section 9.5.2): model-capacity ceiling.

Builds a synthetic "init" npz usable by `fitter_3d.optimise_moonshot --init_from`, whose ONLY
key is joint_rot = the KNOWN ground truth pose (--init_from only overwrites keys present in the
npz, fitter_3d/optimise_moonshot.py:87-93, so omitting every other key deliberately leaves
SMAL3DFitter's own constructor defaults -- e.g. betas starts at mean_betas, not zero -- exactly
as every non-GT-init run in this project already does). Combined with a `scheme: shape` stage (trainer.py's SMALParamGroup "shape"
scheme, which contains global_rot/trans/betas/log_beta_scales/betas_trans and explicitly
EXCLUDES joint_rot and deform_verts -- fitter_3d/trainer.py:373-383), this isolates the
question BARC's precedent motivates (Section 7F/H6): with pose given for free and NO free-form
offsets, how much distal-leg morphometric error remains once only shape/scale is allowed to
explain the target? That residual is the capacity floor no correspondence/initialization fix
can be expected to close.

This script only WRITES the init npz (cheap, CPU-only, no fitting). The actual ceiling fit is
launched by rank2_ceiling.sbatch via `optimise_moonshot --init_from <this file>
--yaml_src cfg/ceiling_shape_only.yaml`, which never touches joint_rot's gradient (scheme
excludes it) -- so joint_rot is pinned at the ground-truth value used here for the ENTIRE run,
not just at initialisation.
"""

import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mesh_files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in mesh_files]
    assert stems, f"no .obj files found in {args.mesh_dir}"

    gt = np.load(os.path.join(args.mesh_dir, "ground_truth.npz"))
    gt_names = [str(x) for x in gt["names"]]
    idx = [gt_names.index(s) for s in stems]  # raises loudly if a specimen is missing
    joint_rot = gt["joint_rot"][idx].astype(np.float32)  # (N, 54, 3), the ONLY key written

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, joint_rot=joint_rot)
    print(f"wrote {args.out}: {len(stems)} specimens, joint_rot only, from GT "
          f"({args.mesh_dir}/ground_truth.npz). All other params left at SMAL3DFitter's own "
          f"constructor default when loaded via --init_from.")


if __name__ == "__main__":
    main()
