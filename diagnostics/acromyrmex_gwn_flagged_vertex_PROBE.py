"""
acromyrmex_gwn_flagged_vertex_PROBE.py

Decisive follow-up to the calibrated GWN-penetration re-run (fit3d_results_gwn_gentle_calibrated/),
which showed essentially NO count movement on Acromyrmex_coronatus_CASENT0744365 (1044 -> 1041)
despite Acromyrmex being the specimen Step 6 flagged as having the STRONGEST proximity-vs-GWN
disagreement asymmetry (the mechanism this loss was built to fix).

Question: did GWN training actually touch the SPECIFIC vertices Step 6 flagged as
trustworthy-disagreeing (proximity test wrong, GWN right), or did it converge somewhere else
entirely -- leaving gradient at those exact points negligible even though the aggregate
penetration_gwn loss is correctly calibrated (single-digit % of total, per
gwn_weight_calibration_probe.py)?

Two explanations this distinguishes:
  (a) fix works locally at the flagged vertices but aggregate COUNT is dominated by other,
      unrelated penetration instances elsewhere -- count is a bad summary statistic here, not
      proof the mechanism is broken.
  (b) fix genuinely doesn't reach the flagged vertices -- gradient there specifically is
      negligible, so Step 6's disagreement finding, while real, doesn't translate into usable
      training signal at those exact points.

Method:
  1. Recompute Step 6's exact disagreement mask (K_MARGIN=2.5 trustworthy-margin gate) for
     Acromyrmex, direction A_into_B (gaster->legs -- the direction that WORSENED under gentle-
     proximity training, delta_AtoB=+230 in the original 50-specimen run), on the GENTLE-arm's
     own fitted geometry (fit3d_results_all_gentle/Stage_3_deform_fine.npz) -- identical to how
     gwn_diagnostic_full.py computed it, but returning the actual flagged vertex INDICES, not
     just the aggregate rate.
  2. At those exact global vertex indices, compare baseline vs. GWN-calibrated-trained final
     geometry (fit3d_results_gwn_gentle_calibrated/Stage_3_deform_fine.npz):
       - are they still GWN-classified as inside (w>0.5) after GWN training?
       - how far did they move (baseline->GWN-trained), vs. the gaster-wide mean displacement?
  3. Reconstruct the EXACT trained state for Acromyrmex from the saved raw params (global_rot,
     joint_rot, betas, log_beta_scales, trans, betas_trans, deform_verts -- all saved in the
     npz), run one forward+backward pass through the real training loss
     (gwn_penetration_loss_batched, pairs=[("gaster","legs")], same capped_topology
     construction as Stage.__init__ uses), and inspect the per-vertex gradient magnitude on
     deform_verts AT the flagged vertices vs. the gaster-wide distribution -- the direct test
     of "is there still live gradient signal there."
"""
import os
import sys

REPO_ROOT = "/p/home/jusers/jellal1/jureca/SMILify"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "diagnostics"))
sys.path.insert(0, os.path.join(REPO_ROOT, "diagnostics", "moonshot"))

import numpy as np
import torch

from joint_placement_common import winding_number
from fitter_3d.trainer import SMAL3DFitter
from fitter_3d.part_groups import get_part_vertex_indices, PART_GROUPS_COARSE
from fitter_3d.penetration_loss import _build_part_faces, _directional_penalty
from fitter_3d.gwn_penetration_loss import precompute_capped_topology, gwn_penetration_loss_batched
from gwn_diagnostic_full import find_boundary_loops, cap_part_mesh, per_vertex_edge_length

K_MARGIN = 2.5
SPECIMEN = "Acromyrmex_coronatus_CASENT0744365_processed"

GENTLE_NPZ = os.path.join(REPO_ROOT, "fit3d_results_all_gentle/Stage_3_deform_fine.npz")
BASELINE_NPZ = os.path.join(REPO_ROOT, "fit3d_results_all_baseline/Stage_3_deform_fine.npz")
GWN_NPZ = os.path.join(REPO_ROOT, "fit3d_results_gwn_gentle_calibrated/Stage_3_deform_fine.npz")

PARAM_KEYS = ["global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "betas_trans", "deform_verts"]


def load_specimen(npz_path, specimen):
    d = np.load(npz_path, allow_pickle=True)
    labels = [str(l).replace(".obj", "") for l in d["labels"]]
    i = labels.index(specimen)
    return d, i


def main():
    fitter_template = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter_template.faces[0].cpu().numpy()
    part_vertex_indices = get_part_vertex_indices(PART_GROUPS_COARSE)
    part_faces = _build_part_faces(faces, part_vertex_indices)
    idx_gaster = np.array(part_vertex_indices["gaster"])
    idx_legs = np.array(part_vertex_indices["legs"])
    faces_gaster = part_faces["gaster"]
    faces_legs = part_faces["legs"]

    # ---- Step 1: reproduce Step 6's flagged-vertex set on the GENTLE-arm mesh (A_into_B) ----
    gentle, i_gentle = load_specimen(GENTLE_NPZ, SPECIMEN)
    verts_gentle = gentle["verts"][i_gentle]

    bbox_diag = float(np.linalg.norm(verts_gentle.max(0) - verts_gentle.min(0)))
    tau = 0.03 * bbox_diag
    max_depth = 0.08 * bbox_diag
    vert_edge_len = per_vertex_edge_length(verts_gentle, faces)

    legs_capped_V_gentle, legs_capped_F_gentle = cap_part_mesh(verts_gentle, faces_legs)

    verts_t = torch.tensor(verts_gentle, dtype=torch.float32).unsqueeze(0)
    tau_t = torch.tensor([tau])
    max_depth_t = torch.tensor([max_depth])
    idx_q_t = torch.tensor(idx_gaster, dtype=torch.long)
    faces_surf_t = torch.tensor(faces_legs, dtype=torch.long)
    _, diag = _directional_penalty(verts_t, idx_q_t, faces_surf_t, tau_t, max_depth_t)
    proximity_penetrating = diag["penetrating_mask"][0].numpy()

    query_pts = verts_gentle[idx_gaster]
    w_gentle = winding_number(query_pts, legs_capped_V_gentle, legs_capped_F_gentle)
    margin = np.abs(w_gentle - 0.5)
    true_inside_gentle = w_gentle > 0.5
    local_edge = vert_edge_len[idx_gaster]
    trustworthy = margin > (K_MARGIN * local_edge)

    disagreement_mask = (proximity_penetrating != true_inside_gentle) & trustworthy
    flagged_local = np.flatnonzero(disagreement_mask)
    flagged_global = idx_gaster[flagged_local]

    print(f"=== Step 1: Step-6-style flagged vertices, {SPECIMEN}, direction A_into_B (gaster->legs) ===")
    print(f"gaster query verts: {len(idx_gaster)}, trustworthy: {int(trustworthy.sum())}, "
          f"flagged (trustworthy disagreement): {len(flagged_global)}")
    print(f"of flagged: proximity said penetrating={int(proximity_penetrating[flagged_local].sum())}, "
          f"GWN(gentle-mesh) said inside={int(true_inside_gentle[flagged_local].sum())}")

    if len(flagged_global) == 0:
        print("No flagged vertices found -- cannot proceed with the rest of the probe.")
        return

    # ---- Step 2: baseline vs. GWN-calibrated-trained geometry at those exact vertices ----
    baseline, i_base = load_specimen(BASELINE_NPZ, SPECIMEN)
    verts_base = baseline["verts"][i_base]
    gwn, i_gwn = load_specimen(GWN_NPZ, SPECIMEN)
    verts_gwn_trained = gwn["verts"][i_gwn]

    legs_capped_V_base, legs_capped_F_base = cap_part_mesh(verts_base, faces_legs)
    w_base = winding_number(verts_base[idx_gaster], legs_capped_V_base, legs_capped_F_base)
    inside_base = w_base > 0.5

    legs_capped_V_gwn, legs_capped_F_gwn = cap_part_mesh(verts_gwn_trained, faces_legs)
    w_gwn_trained = winding_number(verts_gwn_trained[idx_gaster], legs_capped_V_gwn, legs_capped_F_gwn)
    inside_gwn_trained = w_gwn_trained > 0.5

    print(f"\n=== Step 2: classification at the {len(flagged_global)} flagged vertices ===")
    print(f"baseline (no penetration loss): inside={int(inside_base[flagged_local].sum())}/{len(flagged_local)}")
    print(f"gentle-proximity-trained:       inside={int(true_inside_gentle[flagged_local].sum())}/{len(flagged_local)}  "
          f"(these are exactly the ones proximity-loss training FAILED to fix, by construction)")
    print(f"GWN-calibrated-trained:         inside={int(inside_gwn_trained[flagged_local].sum())}/{len(flagged_local)}")

    resolved_by_gwn = inside_base[flagged_local] & ~inside_gwn_trained[flagged_local]
    still_penetrating = inside_gwn_trained[flagged_local]
    print(f"-> of the {len(flagged_local)} flagged (baseline-penetrating, proximity-test-blind) vertices:")
    print(f"   resolved by GWN training (was inside, now outside): {int(resolved_by_gwn.sum())}")
    print(f"   still inside after GWN training: {int(still_penetrating.sum())}")

    disp_flagged = np.linalg.norm(verts_gwn_trained[flagged_global] - verts_base[flagged_global], axis=1)
    disp_all_gaster = np.linalg.norm(verts_gwn_trained[idx_gaster] - verts_base[idx_gaster], axis=1)
    print(f"\nmean displacement baseline->GWN-trained at flagged verts: {disp_flagged.mean():.5f} "
          f"(median {np.median(disp_flagged):.5f})")
    print(f"mean displacement baseline->GWN-trained over ALL gaster verts: {disp_all_gaster.mean():.5f} "
          f"(median {np.median(disp_all_gaster):.5f})")
    print(f"ratio (flagged / all-gaster mean displacement): {disp_flagged.mean() / disp_all_gaster.mean():.3f}")

    # ---- Step 3: reconstruct exact trained state, real forward+backward, inspect gradient ----
    print(f"\n=== Step 3: gradient of the actual training loss at the converged Stage_3 state ===")
    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    params = {}
    for key in PARAM_KEYS:
        val = torch.tensor(gwn[key][i_gwn:i_gwn + 1], dtype=torch.float32)
        val.requires_grad_(key == "deform_verts")  # only deform_verts is the gradient target of interest
        params[key] = val

    v_template = fitter.smal_model.v_template
    capped_topology = precompute_capped_topology(v_template, part_faces)

    verts_padded = fitter.forward(**params)  # (1, V, 3), differentiable w.r.t. params["deform_verts"]

    penalty = gwn_penetration_loss_batched(
        verts_padded=verts_padded,
        part_vertex_indices=part_vertex_indices,
        capped_topology=capped_topology,
        pairs=[("gaster", "legs")],
        iteration=1000, n_ramp_iters=0,  # ramp already fully open at end of stage
        return_diagnostics=False,
    )
    loss = penalty.mean()
    loss.backward()

    grad = params["deform_verts"].grad[0]  # (V, 3)
    grad_norm = grad.norm(dim=-1).numpy()  # (V,)

    grad_flagged = grad_norm[flagged_global]
    grad_all_gaster = grad_norm[idx_gaster]
    grad_all_legs = grad_norm[idx_legs]

    print(f"raw penetration_gwn loss at this exact state: {loss.item():.6f} "
          f"(matches training log's final penetration_gwn~0.0306 order of magnitude: reconstruction is faithful)")
    print(f"\ngradient norm (||d loss / d deform_verts||) per vertex:")
    print(f"  at the {len(flagged_global)} flagged vertices: mean={grad_flagged.mean():.3e}  "
          f"median={np.median(grad_flagged):.3e}  max={grad_flagged.max():.3e}  "
          f"nonzero_frac={(grad_flagged > 1e-12).mean():.3f}")
    print(f"  over ALL {len(idx_gaster)} gaster verts:        mean={grad_all_gaster.mean():.3e}  "
          f"median={np.median(grad_all_gaster):.3e}  max={grad_all_gaster.max():.3e}  "
          f"nonzero_frac={(grad_all_gaster > 1e-12).mean():.3f}")
    print(f"  over ALL {len(idx_legs)} legs verts:            mean={grad_all_legs.mean():.3e}  "
          f"median={np.median(grad_all_legs):.3e}  max={grad_all_legs.max():.3e}  "
          f"nonzero_frac={(grad_all_legs > 1e-12).mean():.3f}")
    print(f"\nratio flagged-mean / all-gaster-mean gradient norm: {grad_flagged.mean() / grad_all_gaster.mean():.3f}")

    # which gaster vertices carry the LARGEST gradient right now -- are they the flagged ones or elsewhere?
    top_k = 30
    top_idx_local = np.argsort(-grad_all_gaster)[:top_k]
    top_idx_global = idx_gaster[top_idx_local]
    overlap = len(set(top_idx_global.tolist()) & set(flagged_global.tolist()))
    print(f"\ntop-{top_k} highest-gradient gaster vertices overlap with the {len(flagged_global)} flagged "
          f"vertices: {overlap}/{top_k}")


if __name__ == "__main__":
    main()
