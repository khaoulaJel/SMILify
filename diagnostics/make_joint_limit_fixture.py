"""Issue #97 probe: build a SMIL_OmniAnt.pkl copy with a synthetic, deliberately
narrow joint_limits entry that conflicts with where the unconstrained fit lands.

Target joint chosen from fit3d_results_nolimit/Stage_3_deform_fine.npz: 'l_3_pt_l'
had the largest unconstrained rotation magnitude (axis 1 / y: 0.4045 rad). We clamp
that axis to [-0.05, 0.05] so an unconstrained fit is expected to violate it and a
w_limit>0 fit is expected to be pulled back toward it.
"""
import pickle as pkl
import numpy as np

SRC = "3D_model_prep/SMIL_OmniAnt.pkl"
DST = "diagnostics/SMIL_OmniAnt_test_limits.pkl"
TARGET_JOINT = "l_3_pt_l"
TARGET_AXIS = 1
TARGET_RANGE = (-0.05, 0.05)

with open(SRC, "rb") as f:
    u = pkl._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()

n_joints = len(dd["J_names"])
joint_limits = np.empty((n_joints, 3, 2))
joint_limits[..., 0] = -np.pi
joint_limits[..., 1] = np.pi

target_idx = dd["J_names"].index(TARGET_JOINT)
joint_limits[target_idx, TARGET_AXIS] = TARGET_RANGE

dd["joint_limits"] = joint_limits

with open(DST, "wb") as f:
    pkl.dump(dd, f)

print(f"wrote {DST}")
print(f"target joint {TARGET_JOINT} (idx {target_idx}), axis {TARGET_AXIS} -> {TARGET_RANGE}")
