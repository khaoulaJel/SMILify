"""
run_symmetric_chamfer_experiment.py

TASK 7 -- causal test of fitter_3d/trainer.py's new `symmetric_chamfer_sampling`
option (see Stage.__init__'s docstring): does area-weighted-sampling BOTH sides
of the chamfer term, instead of the target-only-sampled-vs-raw-source-vertices
status quo, fix the chamfer vertex-density asymmetry FINDINGS.md step 3
diagnosed (legs 2.36x denser than gaster in the template) -- the SECOND of the
two independent, confirmed-but-untouched-until-now causes of the count/F-score
problem (the first, the proximity-test accuracy asymmetry, was TASK6's subject
and did not resolve cleanly).

Single-variable swap, same discipline as every prior arm in this investigation:
holds TASK5's full config exactly fixed (fitter_3d/ants_cfg_all_offset_gentle_pairscoped_scalecap.yaml
-- pair-scoped gentle w_penetration, w_offset, w_scale/w_trans barriers,
scheme:'all') and changes ONLY symmetric_chamfer_sampling. w_sdf forced to 0
explicitly in both arms (inert here regardless -- optimise.py never loads
sdf_values/source_sdf_values for these configs, and trainer.py's SDF block is
additionally gated on both being non-None -- but stated explicitly per this
project's discipline of not silently inheriting a default rather than
controlling for it).

Runs BOTH arms fresh (neither is a pre-existing result at this specimen scope
-- TASK5's own published numbers are a 50-specimen bench aggregate, not
per-specimen numbers for these 3 specimens specifically, so reusing them would
violate the "matched control, not a differently-scoped historical number"
discipline used throughout this investigation).

Same 3 specimens as TASK6 (gwn_matching_primitive_causal_test), chosen there to
span the mechanism and reused here so this result is directly comparable to
TASK6's, not because they were re-selected for this fix specifically:
Acanthostichus_aff.brevicornis, Acromyrmex_coronatus (strongest proximity-test
disagreement / chamfer-density-implicated pair), Solenopsis_invicta (contrast
case). This answers "does symmetric sampling fix the mechanism", NOT "does it
clear the product F-score bar" -- that needs TASK1-5's 10-specimen x 3-seed
bench and is a natural follow-up if this pilot's signal looks promising, not
run here.
"""
import os

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
OUT_ROOT = os.environ.get("SYMCHAMFER_OUT_ROOT", "fit3d_results_task7_symmetric_chamfer")
PENETRATION_TRAIN_PAIRS = [("gaster", "legs")]

STAGE0_NITS = int(os.environ.get("SYMCHAMFER_STAGE0_NITS", 100))
STAGE1_NITS = int(os.environ.get("SYMCHAMFER_STAGE1_NITS", 300))
STAGE2_NITS = int(os.environ.get("SYMCHAMFER_STAGE2_NITS", 1000))
STAGE3_NITS = int(os.environ.get("SYMCHAMFER_STAGE3_NITS", 1000))

# Verbatim from ants_cfg_all_offset_gentle_pairscoped_scalecap.yaml (TASK5), plus
# explicit w_sdf=0 in every stage -- the only content change from TASK5's own config.
STAGE1_LOSS_WEIGHTS = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.02, w_laplacian=0.01, w_limit=100.0, w_sdf=0.0)
STAGE2_LOSS_WEIGHTS = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.005, w_laplacian=0.01, w_limit=100.0, w_sdf=0.0,
                            w_penetration=0.02, penetration_ramp_iters=200, w_offset=5.0, w_scale=0.052, w_trans=1.0)
STAGE3_LOSS_WEIGHTS = dict(w_chamfer=0.5, w_edge=0.2, w_normal=0.002, w_laplacian=0.001, w_limit=100.0, w_sdf=0.0,
                            w_penetration=0.05, penetration_ramp_iters=0, w_offset=2.0, w_scale=0.052, w_trans=1.0)


def run_arm(symmetric_chamfer_sampling: bool, out_dir: str, specimens=None):
    if specimens is None:
        specimens = SPECIMENS
    mesh_files = [os.path.join(MESH_DIR, s) for s in specimens]
    mesh_names, target_meshes = load_meshes(mesh_files=mesh_files, device=device)
    print(f"\n=== arm: symmetric_chamfer_sampling={symmetric_chamfer_sampling} -> {out_dir} ===")
    print(f"Loaded {len(mesh_names)} specimens: {mesh_names}")

    os.makedirs(out_dir, exist_ok=True)
    fitter = SMAL3DFitter(batch_size=len(mesh_names), device=device, shape_family=-1)

    stage0 = Stage(nits=STAGE0_NITS, scheme="init_rot_lock", smal_3d_fitter=fitter, target_meshes=target_meshes,
                   mesh_names=mesh_names, name="Stage_0_init", lr=0.05, out_dir=out_dir, device=device)
    stage0.run()
    stage0.save_npz(labels=mesh_names)

    stage1 = Stage(nits=STAGE1_NITS, scheme="default", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_1_default", lr=0.02, out_dir=out_dir, device=device,
                    loss_weights=STAGE1_LOSS_WEIGHTS, custom_lrs={"joint_rot": 0.002},
                    symmetric_chamfer_sampling=symmetric_chamfer_sampling)
    stage1.run()
    stage1.save_npz(labels=mesh_names)

    manager = StageManager(out_dir=out_dir, labels=mesh_names)
    stage2 = Stage(nits=STAGE2_NITS, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_2_deform_coarse", lr=0.002, out_dir=out_dir, device=device,
                    loss_weights=STAGE2_LOSS_WEIGHTS, penetration_train_pairs=PENETRATION_TRAIN_PAIRS,
                    symmetric_chamfer_sampling=symmetric_chamfer_sampling)
    stage3 = Stage(nits=STAGE3_NITS, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_3_deform_fine", lr=0.0005, out_dir=out_dir, device=device,
                    loss_weights=STAGE3_LOSS_WEIGHTS, penetration_train_pairs=PENETRATION_TRAIN_PAIRS,
                    symmetric_chamfer_sampling=symmetric_chamfer_sampling)
    manager.add_stage(stage2)
    manager.add_stage(stage3)
    manager.run()
    manager.plot_losses("losses")

    return mesh_names


def main():
    control_dir = os.path.join(OUT_ROOT, "control_raw_src_verts")
    experimental_dir = os.path.join(OUT_ROOT, "experimental_symmetric_sampled")

    mesh_names = run_arm(symmetric_chamfer_sampling=False, out_dir=control_dir)
    run_arm(symmetric_chamfer_sampling=True, out_dir=experimental_dir)

    cols = ["specimen", "penetration_num_penetrating", "penetration_fraction_penetrating",
            "penetration_mean_depth_among_penetrating", "f_score@0.01"]
    eval_control = pd.read_csv(os.path.join(control_dir, "Stage_3_deform_fine_eval_metrics.csv"))
    eval_experimental = pd.read_csv(os.path.join(experimental_dir, "Stage_3_deform_fine_eval_metrics.csv"))
    for df in (eval_control, eval_experimental):
        df["specimen"] = df["specimen"].str.replace(r"\.obj$", "", regex=True)

    print("\n=== control (symmetric_chamfer_sampling=False, TASK5-as-is) ===")
    print(eval_control[cols].to_string(index=False))
    print("\n=== experimental (symmetric_chamfer_sampling=True) ===")
    print(eval_experimental[cols].to_string(index=False))

    combined = eval_control[cols].merge(
        eval_experimental[cols], on="specimen", suffixes=("_control", "_experimental")
    )
    out_path = os.path.join(OUT_ROOT, "control_vs_experimental.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
