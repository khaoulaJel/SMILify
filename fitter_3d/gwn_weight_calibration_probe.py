"""
gwn_weight_calibration_probe.py

Cheap calibration probe for w_penetration_gwn, BEFORE any full training run --
the post-mortem on the first GWN penetration-loss attempt found penetration_gwn
at 70.5% of the total weighted loss by the end of Stage_3 (vs. chamfer's 14.1%),
because w_penetration_gwn=0.05 was copied by analogy from the existing
proximity loss's weight without accounting for the two losses living on
different natural scales (Euclidean-distance-fraction vs. winding-number-based).

Method: warm up ONCE via Stage_0/Stage_1 (same as the real experiment), snapshot
that converged parameter state, then for each candidate weight run a SHORT
(20-iteration) scheme:'all' probe from the SAME snapshot (ramp disabled --
calibrating steady-state magnitude, ramp is a separate safety measure layered
on afterward) and track the weighted-contribution fraction of penetration_gwn
over those 20 iterations. Target: comparable to edge/normal (single-digit %),
not dominating chamfer.
"""
import os
import numpy as np
import pandas as pd
import torch

import config
config.PLOT_RESULTS = False  # probe only, no mesh renders needed

from fitter_3d.utils import load_meshes
from fitter_3d.trainer import SMAL3DFitter, Stage

device = "cuda" if torch.cuda.is_available() else "cpu"

MESH_DIR = "diagnostics/moonshot/bench50_clean"
SPECIMENS = [
    "Acanthostichus_aff.brevicornis_CASENT0744328_processed.obj",
    "Acromyrmex_coronatus_CASENT0744365_processed.obj",
    "Solenopsis_invicta_CASENT0744384_processed.obj",
]
GWN_PAIRS = [("gaster", "legs")]
CANDIDATE_WEIGHTS = [0.0005, 0.001, 0.005]
PROBE_NITS = 20
OUT_ROOT = os.environ.get("GWN_CALIB_OUT_ROOT", "gwn_calibration_probe_scratch")  # scratch, not a real results dir

STAGE1_LOSS_WEIGHTS = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.02, w_laplacian=0.01, w_limit=100.0)

# Both real stages' non-GWN base weights, from run_gwn_penetration_experiment.py's
# STAGE2_LOSS_WEIGHTS / STAGE3_LOSS_WEIGHTS (minus w_penetration_gwn itself). Swept
# separately because Stage_3 deliberately drops w_chamfer/w_edge/w_normal well below
# Stage_2's values -- the SAME w_penetration_gwn therefore lands as a proportionally
# larger fraction of total loss under Stage_3's weights even with identical raw GWN
# loss values, so a weight calibrated only against Stage_2 cannot be assumed safe for
# Stage_3 (and definitely should not simply be scaled up further, as the original
# 0.02->0.05 choice did).
BASE_WEIGHT_PROFILES = {
    "stage2": dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.005, w_laplacian=0.01, w_limit=100.0),
    "stage3": dict(w_chamfer=0.5, w_edge=0.2, w_normal=0.002, w_laplacian=0.001, w_limit=100.0),
}

PARAM_NAMES = ["global_rot", "joint_rot", "betas", "trans", "log_beta_scales", "betas_trans", "deform_verts"]


def snapshot(fitter):
    return {name: getattr(fitter, name).data.clone() for name in PARAM_NAMES}


def restore(fitter, snap):
    for name, val in snap.items():
        getattr(fitter, name).data.copy_(val)


def main():
    mesh_names, target_meshes = load_meshes(mesh_files=[os.path.join(MESH_DIR, s) for s in SPECIMENS], device=device)
    print(f"Loaded {len(mesh_names)} specimens: {mesh_names}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    fitter = SMAL3DFitter(batch_size=len(mesh_names), device=device, shape_family=-1)
    stage0 = Stage(nits=100, scheme="init_rot_lock", smal_3d_fitter=fitter, target_meshes=target_meshes,
                   mesh_names=mesh_names, name="Stage_0_init", lr=0.05, out_dir=OUT_ROOT, device=device)
    stage0.run()
    stage1 = Stage(nits=300, scheme="default", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_1_default", lr=0.02, out_dir=OUT_ROOT, device=device,
                    loss_weights=STAGE1_LOSS_WEIGHTS, custom_lrs={"joint_rot": 0.002})
    stage1.run()
    warm_snapshot = snapshot(fitter)
    print("Warm-up (Stage_0/1) complete, snapshot taken.\n")

    all_rows = []
    for profile_name, base_weights in BASE_WEIGHT_PROFILES.items():
        for wgt in CANDIDATE_WEIGHTS:
            restore(fitter, warm_snapshot)  # fresh start from the SAME warmed-up state for every candidate
            loss_weights = dict(base_weights)
            loss_weights["w_penetration_gwn"] = wgt
            loss_weights["penetration_gwn_ramp_iters"] = 0  # calibrating steady-state magnitude, not ramp behavior

            probe_stage = Stage(nits=PROBE_NITS, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
                                 mesh_names=mesh_names, name=f"probe_{profile_name}_w{wgt}", lr=0.002,
                                 out_dir=OUT_ROOT, device=device, loss_weights=loss_weights,
                                 gwn_penetration_pairs=GWN_PAIRS)

            print(f"=== profile={profile_name} candidate w_penetration_gwn = {wgt} ===")
            for it in range(PROBE_NITS):
                probe_stage.optimizer.zero_grad()
                loss, loss_components = probe_stage.step(it)  # Stage's own step(): forward + backward + optimizer.step()

                raw = {k: v.item() for k, v in loss_components.items()}
                weighted = {k: loss_weights.get(f"w_{k}", 0.0) * v for k, v in raw.items()}
                total = sum(weighted.values())
                gwn_frac = weighted.get("penetration_gwn", 0.0) / total if total > 0 else float("nan")

                row = {"profile": profile_name, "candidate_weight": wgt, "iteration": it,
                       "gwn_frac_of_total": gwn_frac, "raw_penetration_gwn": raw.get("penetration_gwn", 0.0),
                       "raw_chamfer": raw.get("chamfer", 0.0)}
                all_rows.append(row)
                if it in (0, 4, 9, 19):
                    print(f"  it {it:2d}: gwn_frac_of_total={gwn_frac:.4f}  raw_gwn={raw.get('penetration_gwn', 0):.5f}  "
                          f"raw_chamfer={raw.get('chamfer', 0):.5f}  weighted_gwn={weighted.get('penetration_gwn', 0):.6f}  "
                          f"weighted_chamfer={weighted.get('chamfer', 0):.6f}")
            print()

    df = pd.DataFrame(all_rows)
    df.to_csv("diagnostics/gwn_weight_calibration_probe_results.csv", index=False)

    print("=== SUMMARY: mean gwn_frac_of_total over the probe, per (profile, candidate) ===")
    summary = df.groupby(["profile", "candidate_weight"])["gwn_frac_of_total"].agg(["mean", "min", "max"])
    print(summary.to_string())
    print("\nTarget: single-digit % (comparable to edge/normal), not dominating chamfer (70%+ was the failure mode).")


if __name__ == "__main__":
    main()
