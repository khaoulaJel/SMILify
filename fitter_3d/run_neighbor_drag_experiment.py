"""
run_neighbor_drag_experiment.py

Tests the neighbor-drag hypothesis directly: does locally down-weighting
mesh smoothness (edge/laplacian) around currently-penetrating vertices let
the gentle penetration loss reduce collision COUNT (not just depth), by
giving those vertices freedom to retreat without dragging clean neighbors
along?

Two arms, matched on COUNT and per-vertex weight at every iteration, differing
only in WHERE the down-weighted vertices are:
  - "local":  down-weight vertices within k=2 mesh-hops of a currently-
              penetrating vertex (k=3, k=5 coverage logged too, for
              reference -- k=2 is what actually drives the down-weight).
              Spatially CLUSTERED.
  - "global": down-weight the SAME NUMBER of vertices, at the SAME "factor"
              weight, but as a FRESH random subset of the whole mesh drawn
              every iteration (replaying "local"'s own logged per-iteration
              touched-vertex COUNT, not a blended average). Spatially
              SCATTERED. This is the control: a positive result here means
              it isn't localness doing the work, just that an equivalent
              amount of down-weighting happened somewhere.

              Resampling fresh every iteration (rather than a fixed random
              subset drawn once) is a deliberate choice: it isolates
              clustered-vs-scattered cleanly, at the cost of giving
              "global" no temporal persistence at any one location, unlike
              "local" (whose target region tends to recur, since the same
              collision persists across iterations). A positive result
              still leaves open whether persistence, not just clustering,
              matters -- see local_smoothness.py's docstring.

Both arms use the identical Stage_0/Stage_1/Stage_2/Stage_3 schedule and
loss weights as fitter_3d/ants_cfg_all_penetration_gentle.yaml, on 3
specimens from diagnostics/moonshot/bench50_clean chosen from the earlier
per-pair persistence check: two with the strongest observed neighbor-drag
signal (Acanthostichus_aff.brevicornis, Acromyrmex_coronatus) and one from
the minority that did NOT show the signal (Solenopsis_invicta), as a
contrast case.

Also reports mesh-integrity metrics (edge_logratio, folded_face_frac) on
both arms throughout -- down-weighting smoothness, even locally, is
exactly the kind of change that can quietly damage geometry while
penetration numbers look better, so this must not be judged on penetration
metrics alone.
"""

import os
import numpy as np
import pandas as pd
import torch

import config
if os.environ.get("NEIGHBOR_DRAG_DISABLE_PLOTTING"):
    # For fast local smoke tests only -- plot_meshes()'s multiprocessing.Pool
    # workers each reimport torch/pytorch3d via 'spawn', which is fine given
    # the real run's multi-hour SLURM budget but too slow for a quick
    # correctness check. The real run leaves this on (default config.PLOT_RESULTS).
    config.PLOT_RESULTS = False

from fitter_3d.utils import load_meshes
from fitter_3d.trainer import SMAL3DFitter, Stage, StageManager
from fitter_3d.local_smoothness import mesh_integrity_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"

MESH_DIR = "diagnostics/moonshot/bench50_clean"
SPECIMENS = [
    "Acanthostichus_aff.brevicornis_CASENT0744328_processed.obj",
    "Acromyrmex_coronatus_CASENT0744365_processed.obj",
    "Solenopsis_invicta_CASENT0744384_processed.obj",
]
K_PRIMARY = 2
LOG_EXTRA_K = [3, 5]
DOWNWEIGHT_FACTOR = 0.1  # touched vertices/edges get 10% of normal smoothness weight
OUT_ROOT = os.environ.get("NEIGHBOR_DRAG_OUT_ROOT", "neighbor_drag_results")

# Overridable via env vars for a fast local smoke test; defaults match
# ants_cfg_all_penetration_gentle.yaml exactly for the real run.
STAGE0_NITS = int(os.environ.get("NEIGHBOR_DRAG_STAGE0_NITS", 100))
STAGE1_NITS = int(os.environ.get("NEIGHBOR_DRAG_STAGE1_NITS", 300))
STAGE2_NITS = int(os.environ.get("NEIGHBOR_DRAG_STAGE2_NITS", 1000))
STAGE3_NITS = int(os.environ.get("NEIGHBOR_DRAG_STAGE3_NITS", 1000))

STAGE1_LOSS_WEIGHTS = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.02, w_laplacian=0.01, w_limit=100.0)
STAGE2_LOSS_WEIGHTS = dict(
    w_chamfer=1.0, w_edge=0.8, w_normal=0.005, w_laplacian=0.01,
    w_penetration=0.02, penetration_ramp_iters=200,
)
STAGE3_LOSS_WEIGHTS = dict(
    w_chamfer=0.5, w_edge=0.2, w_normal=0.002, w_laplacian=0.001,
    w_penetration=0.05, penetration_ramp_iters=0,
)


def build_warmed_up_fitter(target_meshes, mesh_names, out_dir):
    """Fresh SMAL3DFitter + Stage_0_init + Stage_1_default, matching
    ants_cfg_all_penetration_gentle.yaml exactly, so 'local' and 'global'
    arms both start Stage_2 from the same (re-derived) point rather than
    sharing mutated parameter state."""
    fitter = SMAL3DFitter(batch_size=len(mesh_names), device=device, shape_family=-1)

    stage0 = Stage(
        nits=STAGE0_NITS, scheme="init_rot_lock", smal_3d_fitter=fitter, target_meshes=target_meshes,
        mesh_names=mesh_names, name="Stage_0_init", lr=0.05, out_dir=out_dir, device=device,
    )
    stage0.run()
    stage0.save_npz(labels=mesh_names)

    stage1 = Stage(
        nits=STAGE1_NITS, scheme="default", smal_3d_fitter=fitter, target_meshes=target_meshes,
        mesh_names=mesh_names, name="Stage_1_default", lr=0.02, out_dir=out_dir, device=device,
        loss_weights=STAGE1_LOSS_WEIGHTS, custom_lrs={"joint_rot": 0.002},
    )
    stage1.run()
    stage1.save_npz(labels=mesh_names)

    return fitter


def run_arm(arm_name, target_meshes, mesh_names, local_downweight_stage2, local_downweight_stage3):
    out_dir = os.path.join(OUT_ROOT, arm_name)
    os.makedirs(out_dir, exist_ok=True)

    fitter = build_warmed_up_fitter(target_meshes, mesh_names, out_dir)

    manager = StageManager(out_dir=out_dir, labels=mesh_names)
    stage2 = Stage(
        nits=STAGE2_NITS, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
        mesh_names=mesh_names, name="Stage_2_deform_coarse", lr=0.002, out_dir=out_dir, device=device,
        loss_weights=STAGE2_LOSS_WEIGHTS, local_downweight=local_downweight_stage2,
    )
    stage3 = Stage(
        nits=STAGE3_NITS, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
        mesh_names=mesh_names, name="Stage_3_deform_fine", lr=0.0005, out_dir=out_dir, device=device,
        loss_weights=STAGE3_LOSS_WEIGHTS, local_downweight=local_downweight_stage3,
    )
    manager.add_stage(stage2)
    manager.add_stage(stage3)
    manager.run()
    manager.plot_losses("losses")

    return fitter, stage2, stage3


def touched_count_schedule(log_csv_path, k, mesh_names, n_iterations):
    """Pivots a saved local_downweight_log.csv (k==k rows only) into a
    (n_iterations, n_specimens) int array of the exact n_touched vertex
    count "local" mode used each iteration, in mesh_names order -- what
    "global" mode replays (see Stage._compute_local_downweight_vertex_weight):
    a fresh random subset of that same size every iteration, at the same
    "factor" weight, so the two arms are matched on count and per-vertex
    weight and differ only in spatial arrangement (clustered vs. scattered)."""
    df = pd.read_csv(log_csv_path)
    df = df[df["k"] == k]
    pivot = df.pivot_table(index="iteration", columns="specimen", values="n_touched", fill_value=0)
    schedule = np.zeros((n_iterations, len(mesh_names)), dtype=np.int64)
    for it in range(n_iterations):
        if it not in pivot.index:
            continue  # no penetration anywhere this iteration -> 0 touched for every specimen
        for spec_idx, name in enumerate(mesh_names):
            if name in pivot.columns:
                schedule[it, spec_idx] = int(pivot.loc[it, name])
    return schedule


def compute_integrity(fitter, mesh_names, faces_np):
    """edge_logratio / folded_face_frac per specimen, current fitted verts
    vs the SMAL template rest pose (v_template) -- the known-good reference
    every edge/laplacian regularizer is implicitly trying to stay close to."""
    with torch.no_grad():
        final_verts = fitter().cpu().numpy()  # (B, V, 3)
    source_verts = fitter.smal_model.v_template.detach().cpu().numpy()  # (V, 3)

    rows = []
    for i, name in enumerate(mesh_names):
        metrics = mesh_integrity_metrics(source_verts, final_verts[i], faces_np)
        metrics["specimen"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


def main():
    mesh_files = [os.path.join(MESH_DIR, s) for s in SPECIMENS]
    mesh_names, target_meshes = load_meshes(mesh_files=mesh_files, device=device)
    print(f"Loaded {len(mesh_names)} specimens: {mesh_names}")

    os.makedirs(OUT_ROOT, exist_ok=True)

    # ---- LOCAL arm --------------------------------------------------------
    local_dw_stage2 = {"mode": "local", "k": K_PRIMARY, "factor": DOWNWEIGHT_FACTOR, "log_extra_k": LOG_EXTRA_K}
    local_dw_stage3 = {"mode": "local", "k": K_PRIMARY, "factor": DOWNWEIGHT_FACTOR, "log_extra_k": LOG_EXTRA_K}
    print("\n=== Running LOCAL arm ===")
    fitter_local, stage2_local, stage3_local = run_arm(
        "local", target_meshes, mesh_names, local_dw_stage2, local_dw_stage3
    )
    faces_np = stage3_local.faces[0].cpu().numpy()

    # ---- build the matched global schedule from the local arm's own log ---
    # Exact touched-vertex COUNTS (not a blended average weight) -- "global"
    # mode redraws a fresh random subset of this same size every iteration,
    # at the same "factor" weight, so it's matched to "local" on count and
    # per-vertex weight, differing only in spatial arrangement.
    schedule_stage2 = touched_count_schedule(
        os.path.join(OUT_ROOT, "local", "Stage_2_deform_coarse_local_downweight_log.csv"),
        K_PRIMARY, mesh_names, STAGE2_NITS,
    )
    schedule_stage3 = touched_count_schedule(
        os.path.join(OUT_ROOT, "local", "Stage_3_deform_fine_local_downweight_log.csv"),
        K_PRIMARY, mesh_names, STAGE3_NITS,
    )
    print(f"\nGlobal-control schedule built: mean n_touched stage2={schedule_stage2.mean():.2f}, "
          f"stage3={schedule_stage3.mean():.2f} vertices/iteration (out of {schedule_stage2.shape[1]} specimens' "
          f"meshes) -- resampled at random every iteration, same count local used at k={K_PRIMARY}")

    # ---- GLOBAL (control) arm ---------------------------------------------
    global_dw_stage2 = {"mode": "global", "global_schedule": schedule_stage2, "factor": DOWNWEIGHT_FACTOR, "seed": 0}
    global_dw_stage3 = {"mode": "global", "global_schedule": schedule_stage3, "factor": DOWNWEIGHT_FACTOR, "seed": 1}
    print("\n=== Running GLOBAL (control) arm ===")
    fitter_global, stage2_global, stage3_global = run_arm(
        "global", target_meshes, mesh_names, global_dw_stage2, global_dw_stage3
    )

    # ---- mesh-integrity metrics, both arms ---------------------------------
    integrity_local = compute_integrity(fitter_local, mesh_names, faces_np)
    integrity_local["arm"] = "local"
    integrity_global = compute_integrity(fitter_global, mesh_names, faces_np)
    integrity_global["arm"] = "global"
    integrity = pd.concat([integrity_local, integrity_global], ignore_index=True)
    integrity.to_csv(os.path.join(OUT_ROOT, "mesh_integrity_comparison.csv"), index=False)
    print("\n=== Mesh integrity (local vs global) ===")
    print(integrity.to_string(index=False))

    # ---- penetration/F-score comparison, both arms + the existing gentle-of-50 reference
    eval_local = pd.read_csv(os.path.join(OUT_ROOT, "local", "Stage_3_deform_fine_eval_metrics.csv"))
    eval_global = pd.read_csv(os.path.join(OUT_ROOT, "global", "Stage_3_deform_fine_eval_metrics.csv"))
    eval_gentle50 = pd.read_csv("fit3d_results_all_gentle/Stage_3_deform_fine_eval_metrics.csv")
    # eval_gentle50's specimen names come from optimise.py's own os.path.basename()
    # (keeps ".obj"); load_meshes()'s mesh_names strips it -- normalize before matching.
    eval_gentle50["specimen"] = eval_gentle50["specimen"].str.replace(r"\.obj$", "", regex=True)
    eval_gentle50 = eval_gentle50[eval_gentle50["specimen"].isin(mesh_names)]

    cols = ["specimen", "penetration_num_penetrating", "penetration_fraction_penetrating",
            "penetration_mean_depth_among_penetrating", "f_score@0.01"]
    print("\n=== LOCAL arm ===")
    print(eval_local[cols].to_string(index=False))
    print("\n=== GLOBAL (control) arm ===")
    print(eval_global[cols].to_string(index=False))
    print("\n=== reference: gentle arm (of 50), no downweight ===")
    print(eval_gentle50[cols].to_string(index=False))

    combined = eval_local[cols].merge(eval_global[cols], on="specimen", suffixes=("_local", "_global")).merge(
        eval_gentle50[cols].rename(columns={c: f"{c}_gentle50" for c in cols if c != "specimen"}), on="specimen"
    )
    combined.to_csv(os.path.join(OUT_ROOT, "penetration_comparison_summary.csv"), index=False)
    print(f"\nWrote {OUT_ROOT}/penetration_comparison_summary.csv and {OUT_ROOT}/mesh_integrity_comparison.csv")


if __name__ == "__main__":
    main()
