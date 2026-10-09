"""
run_bvh_penetration_experiment_seeded_scale.py

TASK 8 -- the field-standard approach (SMPLify-X's BVH + conical distance field,
`fitter_3d/bvh_penetration_loss.py`), tried as an actual training signal for the first
time -- not another compat probe. This was "the original spec's TASK 4", named untried in
every prior deliverable because torch-mesh-isect was believed non-functional on this
project's environment. That belief was correct for one fork and wrong for another (see
DELIVERABLE_penetration_fix_TASK1-5.md's 2026-08-24 update) -- this run is the actual
unblock, not a fourth compat check.

Matching-primitive swap, same discipline as TASK6 (GWN) and every arm since: TASK5's full
config held exactly fixed (w_offset, w_scale, w_trans, scheme:'all', pair-scoped to
gaster-legs), w_sdf=0 explicit, ONE variable changes -- which loss provides the penetration
signal. Control = TASK5-as-is (w_penetration active, the existing proximity/centroid
primitive). Experimental = w_penetration=0, w_penetration_bvh=0.0005 instead (calibrated via
fitter_3d/bvh_weight_calibration_probe.py -- single-digit % of total loss, stage2 mean 1.3%/
max 1.9%, stage3 mean 2.3%/max 4.7%, comfortably below the 70% blowup GWN's first
uncalibrated attempt hit).

Same 10 specimens, same 3 seeds (0/1/2), same random-selection discipline as
fitter_3d/run_symmetric_chamfer_experiment_seeded_scale.py -- reused verbatim for direct
comparability, not re-drawn. Logs both hard (penetration_num_penetrating) and soft
(penetration_soft_num_penetrating) count from the eval CSV automatically (both wired into
compute_eval_metrics unconditionally as of the soft-count addition), and the per-pair CSV
automatically, so any Acromyrmex-style specimen-dependence is visible immediately rather
than after a second round of analysis.

Requires custom_processing/external/torch-mesh-isect-wonjongg on PYTHONPATH -- handled here
via sys.path, not left to the caller to remember.
"""
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "custom_processing/external/torch-mesh-isect-wonjongg"))

import numpy as np
import torch

import config
if os.environ.get("SMIL_DISABLE_PLOTTING"):
    config.PLOT_RESULTS = False

from fitter_3d.utils import load_meshes
from fitter_3d.trainer import SMAL3DFitter, Stage, StageManager

device = "cuda" if torch.cuda.is_available() else "cpu"

MESH_DIR = "diagnostics/moonshot/bench50_clean"
OUT_ROOT = os.environ.get("BVH_SCALE_OUT_ROOT", "fit3d_results_task8_bvh_seeded_scale")
SEEDS = [0, 1, 2]
PENETRATION_TRAIN_PAIRS = [("gaster", "legs")]
W_PENETRATION_BVH = 0.0005  # calibrated -- see module docstring

# Identical specimen set to run_symmetric_chamfer_experiment_seeded_scale.py, reused
# verbatim (not re-drawn) for direct comparability across TASK7/TASK8.
FIXED_SPECIMENS = [
    "Acromyrmex_coronatus_CASENT0744365_processed.obj",
    "Strumigenys_sp._appretiata_group_CASENT0744156_processed.obj",
]
RANDOM_SPECIMENS = [
    "Strumigenys_alberti_CASENT0878103_processed.obj",
    "Centromyrmex_brachycola_CASENT0744052_processed.obj",
    "Acromyrmex_lobicornis_CASENT0878007_processed.obj",
    "Eciton_hamatum_CASENT0744580_processed.obj",
    "Dilobocondyla_fouqueti_CASENT0745576_processed.obj",
    "Cyphomyrmex_cf.minutus_CASENT0744280_processed.obj",
    "Cephalotes_minutus_CASENT0709253_processed.obj",
    "Carebara_trechideros_CASENT0877591_processed.obj",
]
SPECIMENS = FIXED_SPECIMENS + RANDOM_SPECIMENS

# Verbatim from ants_cfg_all_offset_gentle_pairscoped_scalecap.yaml (TASK5), w_sdf=0 explicit,
# identical to run_symmetric_chamfer_experiment.py's STAGE*_LOSS_WEIGHTS.
STAGE1_LOSS_WEIGHTS = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.02, w_laplacian=0.01, w_limit=100.0, w_sdf=0.0)
STAGE2_BASE = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.005, w_laplacian=0.01, w_limit=100.0, w_sdf=0.0,
                    w_offset=5.0, w_scale=0.052, w_trans=1.0)
STAGE3_BASE = dict(w_chamfer=0.5, w_edge=0.2, w_normal=0.002, w_laplacian=0.001, w_limit=100.0, w_sdf=0.0,
                    w_offset=2.0, w_scale=0.052, w_trans=1.0)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[determinism] seeded RNGs with {seed}, deterministic algorithms requested (warn_only)")


def run_arm(use_bvh: bool, out_dir: str):
    mesh_files = [os.path.join(MESH_DIR, s) for s in SPECIMENS]
    mesh_names, target_meshes = load_meshes(mesh_files=mesh_files, device=device)
    print(f"\n=== arm: use_bvh={use_bvh} -> {out_dir} ===")
    print(f"Loaded {len(mesh_names)} specimens: {mesh_names}")

    os.makedirs(out_dir, exist_ok=True)
    fitter = SMAL3DFitter(batch_size=len(mesh_names), device=device, shape_family=-1)

    stage0 = Stage(nits=100, scheme="init_rot_lock", smal_3d_fitter=fitter, target_meshes=target_meshes,
                   mesh_names=mesh_names, name="Stage_0_init", lr=0.05, out_dir=out_dir, device=device)
    stage0.run()

    stage1 = Stage(nits=300, scheme="default", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_1_default", lr=0.02, out_dir=out_dir, device=device,
                    loss_weights=STAGE1_LOSS_WEIGHTS, custom_lrs={"joint_rot": 0.002})
    stage1.run()

    stage2_weights = dict(STAGE2_BASE)
    stage3_weights = dict(STAGE3_BASE)
    stage_kwargs = {}
    if use_bvh:
        stage2_weights["w_penetration_bvh"] = W_PENETRATION_BVH
        stage2_weights["bvh_penetration_ramp_iters"] = 200
        stage3_weights["w_penetration_bvh"] = W_PENETRATION_BVH
        stage3_weights["bvh_penetration_ramp_iters"] = 0
        stage_kwargs["bvh_penetration_pairs"] = PENETRATION_TRAIN_PAIRS
    else:
        stage2_weights["w_penetration"] = 0.02
        stage2_weights["penetration_ramp_iters"] = 200
        stage3_weights["w_penetration"] = 0.05
        stage3_weights["penetration_ramp_iters"] = 0
        stage_kwargs["penetration_train_pairs"] = PENETRATION_TRAIN_PAIRS

    manager = StageManager(out_dir=out_dir, labels=mesh_names)
    stage2 = Stage(nits=1000, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_2_deform_coarse", lr=0.002, out_dir=out_dir, device=device,
                    loss_weights=stage2_weights, **stage_kwargs)
    stage3 = Stage(nits=1000, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_3_deform_fine", lr=0.0005, out_dir=out_dir, device=device,
                    loss_weights=stage3_weights, **stage_kwargs)
    manager.add_stage(stage2)
    manager.add_stage(stage3)
    manager.run()
    manager.plot_losses("losses")


def main():
    for seed in SEEDS:
        for use_bvh, arm_name in [(False, "control"), (True, "experimental")]:
            seed_everything(seed)
            out_dir = os.path.join(OUT_ROOT, f"seed{seed}_{arm_name}")
            print(f"\n{'='*80}\nseed={seed} arm={arm_name} -> {out_dir}\n{'='*80}")
            run_arm(use_bvh=use_bvh, out_dir=out_dir)


if __name__ == "__main__":
    main()
