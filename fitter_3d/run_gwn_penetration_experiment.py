"""
run_gwn_penetration_experiment.py

Tests fitter_3d/gwn_penetration_loss.py's differentiable, winding-number-based
penetration test as a replacement SIGNAL for penetration_loss.py's nearest-
triangle-centroid proximity test, on the (gaster, legs) pair Step 6 already
validated. Single new arm: "gentle-GWN" (w_penetration_gwn active,
w_penetration=0 -- an isolated, one-variable-at-a-time test, not combined
with the existing proximity loss).

Same acceptance criterion as every prior test in this investigation: does
penetration COUNT actually move, not just depth -- measured with the exact
same proximity-based penetration_num_penetrating metric Stage.compute_eval_metrics()
already reports unconditionally (regardless of which loss trained the
geometry), so the new arm is scored on the identical yardstick as the
existing baseline / gentle-proximity data already sitting in
fit3d_results_all_baseline/ and fit3d_results_all_gentle/ for these same 3
specimens -- no need to rerun those.

3 specimens (same as the neighbor-drag experiment): Acanthostichus_aff.brevicornis,
Acromyrmex_coronatus (strongest asymmetry signal), Solenopsis_invicta (no
signal -- contrast case). Full Stage_0-3 schedule identical to
ants_cfg_all_penetration_gentle.yaml's Stage_0/1, except Stage_2/3 use
w_penetration_gwn instead of w_penetration.
"""
import os
import numpy as np
import pandas as pd
import torch

import config
if os.environ.get("SMIL_DISABLE_PLOTTING"):
    config.PLOT_RESULTS = False

from fitter_3d.utils import load_meshes
from fitter_3d.trainer import SMAL3DFitter, Stage, StageManager

device = "cuda" if torch.cuda.is_available() else "cpu"

MESH_DIR = "diagnostics/moonshot/bench50_clean"
SPECIMENS = [
    "Acanthostichus_aff.brevicornis_CASENT0744328_processed.obj",
    "Acromyrmex_coronatus_CASENT0744365_processed.obj",
    "Solenopsis_invicta_CASENT0744384_processed.obj",
]
OUT_ROOT = os.environ.get("GWN_PEN_OUT_ROOT", "fit3d_results_gwn_gentle")
GWN_PAIRS = [("gaster", "legs")]

STAGE0_NITS = int(os.environ.get("GWN_PEN_STAGE0_NITS", 100))
STAGE1_NITS = int(os.environ.get("GWN_PEN_STAGE1_NITS", 300))
STAGE2_NITS = int(os.environ.get("GWN_PEN_STAGE2_NITS", 1000))
STAGE3_NITS = int(os.environ.get("GWN_PEN_STAGE3_NITS", 1000))

STAGE1_LOSS_WEIGHTS = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.02, w_laplacian=0.01, w_limit=100.0)
# CALIBRATED via fitter_3d/gwn_weight_calibration_probe.py (SLURM job 15521221),
# replacing the first attempt's w_penetration_gwn=0.02/0.05 (copied by analogy from
# the unrelated proximity loss, found to be 70.5% of total weighted loss by the end
# of Stage_3 -- see fit3d_results_gwn_gentle/ for that run's catastrophic output).
#
# The 20-iteration probe found w=0.001 lands at a mean weighted-contribution fraction
# of ~1.2% under Stage_2's base weights and ~2.8% under Stage_3's (Stage_3's lower
# w_chamfer/w_edge/w_normal inflate the SAME raw weight's fraction ~2.3x on their
# own) -- both comfortably single-digit, comparable to edge/normal, not dominating
# chamfer. w=0.005 was ruled out for Stage_3: it opened near 10% and climbed past
# 15% within just 20 iterations, already outside target and trending like the
# original blowup. Deliberately the SAME weight in both stages (not scaled up
# 0.02->0.05 the way the original config was) since Stage_3's own weight profile
# already inflates the fraction without any explicit increase.
#
# NOTE: this was calibrated on a 20-iteration snapshot, not the full 1000-iteration
# schedule -- raw chamfer was still dropping fast while raw GWN loss stayed roughly
# flat, so the fraction likely keeps climbing further into the real run than this
# probe alone shows. Watch this run's own final-iteration loss-component printout
# (the same mechanism that caught the original 70.5%) before trusting the fraction
# stayed single-digit for all 2000 iterations.
STAGE2_LOSS_WEIGHTS = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.005, w_laplacian=0.01, w_limit=100.0,
                            w_penetration_gwn=0.001, penetration_gwn_ramp_iters=200)
STAGE3_LOSS_WEIGHTS = dict(w_chamfer=0.5, w_edge=0.2, w_normal=0.002, w_laplacian=0.001, w_limit=100.0,
                            w_penetration_gwn=0.001, penetration_gwn_ramp_iters=0)


def main():
    mesh_files = [os.path.join(MESH_DIR, s) for s in SPECIMENS]
    mesh_names, target_meshes = load_meshes(mesh_files=mesh_files, device=device)
    print(f"Loaded {len(mesh_names)} specimens: {mesh_names}")

    os.makedirs(OUT_ROOT, exist_ok=True)
    fitter = SMAL3DFitter(batch_size=len(mesh_names), device=device, shape_family=-1)

    stage0 = Stage(nits=STAGE0_NITS, scheme="init_rot_lock", smal_3d_fitter=fitter, target_meshes=target_meshes,
                   mesh_names=mesh_names, name="Stage_0_init", lr=0.05, out_dir=OUT_ROOT, device=device)
    stage0.run()
    stage0.save_npz(labels=mesh_names)

    stage1 = Stage(nits=STAGE1_NITS, scheme="default", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_1_default", lr=0.02, out_dir=OUT_ROOT, device=device,
                    loss_weights=STAGE1_LOSS_WEIGHTS, custom_lrs={"joint_rot": 0.002})
    stage1.run()
    stage1.save_npz(labels=mesh_names)

    manager = StageManager(out_dir=OUT_ROOT, labels=mesh_names)
    stage2 = Stage(nits=STAGE2_NITS, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_2_deform_coarse", lr=0.002, out_dir=OUT_ROOT, device=device,
                    loss_weights=STAGE2_LOSS_WEIGHTS, gwn_penetration_pairs=GWN_PAIRS)
    stage3 = Stage(nits=STAGE3_NITS, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_3_deform_fine", lr=0.0005, out_dir=OUT_ROOT, device=device,
                    loss_weights=STAGE3_LOSS_WEIGHTS, gwn_penetration_pairs=GWN_PAIRS)
    manager.add_stage(stage2)
    manager.add_stage(stage3)
    manager.run()
    manager.plot_losses("losses")

    # ---- three-way comparison against the already-existing step-1 data, IF present ----
    cols = ["specimen", "penetration_num_penetrating", "penetration_fraction_penetrating",
            "penetration_mean_depth_among_penetrating", "f_score@0.01"]
    eval_gwn = pd.read_csv(os.path.join(OUT_ROOT, "Stage_3_deform_fine_eval_metrics.csv"))
    eval_gwn["specimen"] = eval_gwn["specimen"].str.replace(r"\.obj$", "", regex=True)
    print("\n=== gentle-GWN (new w_penetration_gwn) ===")
    print(eval_gwn[cols].to_string(index=False))

    try:
        eval_baseline = pd.read_csv("fit3d_results_all_baseline/Stage_3_deform_fine_eval_metrics.csv")
        eval_proximity = pd.read_csv("fit3d_results_all_gentle/Stage_3_deform_fine_eval_metrics.csv")
    except FileNotFoundError as exc:
        print(f"\nSkipping three-way comparison -- baseline/proximity reference CSVs not found "
              f"locally ({exc}). GWN-only eval metrics above and "
              f"{OUT_ROOT}/Stage_3_deform_fine_eval_metrics.csv are still complete and saved.")
        return

    for df in (eval_baseline, eval_proximity):
        df["specimen"] = df["specimen"].str.replace(r"\.obj$", "", regex=True)

    eval_baseline = eval_baseline[eval_baseline["specimen"].isin(mesh_names)]
    eval_proximity = eval_proximity[eval_proximity["specimen"].isin(mesh_names)]

    print("\n=== baseline (no penetration loss) ===")
    print(eval_baseline[cols].to_string(index=False))
    print("\n=== gentle-proximity (existing w_penetration) ===")
    print(eval_proximity[cols].to_string(index=False))

    combined = eval_baseline[cols].merge(
        eval_proximity[cols], on="specimen", suffixes=("_baseline", "_proximity")
    ).merge(
        eval_gwn[cols].rename(columns={c: f"{c}_gwn" for c in cols if c != "specimen"}), on="specimen"
    )
    combined.to_csv(os.path.join(OUT_ROOT, "three_way_comparison.csv"), index=False)
    print(f"\nWrote {OUT_ROOT}/three_way_comparison.csv")


if __name__ == "__main__":
    main()
