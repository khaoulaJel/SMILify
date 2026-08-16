"""Small synthetic compare: edge_mode='shrink' vs 'rest' (PR #4, to-ship porting).

Claim under test (FINAL_REPORT.md §2.2 item 4, moonshot branch): stock's
mesh_edge_loss(target_length=0) is an active shrinkage force -- it penalises edge
length itself, even on a mesh that is already a perfect fit (correct pose, zero
free-form distortion). rest_edge_loss should score such a mesh exactly 0.

Two direct forward-pass checks (no optimisation loop -- that would let the other
free scheme='all' params, e.g. log_beta_scales, confound the edge measurement):

1. Rigid pose-only case: rotate the root joint by a fixed, legitimate amount
   (deform_verts stays 0). A rotation is an isometry, so this mesh IS a perfect fit
   with zero free-form distortion by construction. 'shrink' should still report a
   large nonzero loss (it measures raw edge length, blind to why); 'rest' should
   report ~0.
2. Deform-sensitivity case: starting from that same posed mesh, add a small known
   deform_verts perturbation and check both losses respond -- 'rest' should react
   sharply (that is what it is designed to catch); 'shrink' reacts only through the
   tiny length change the perturbation itself causes, i.e. much less.

CPU-only, no optimiser, single specimen -- deliberately small, not a benchmark.
"""

import torch
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.structures import Meshes

from fitter_3d.trainer import SMAL3DFitter, get_meshes, rest_edge_loss

torch.manual_seed(0)
device = "cpu"


def edge_lengths(verts, faces):
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    return (verts[:, e[:, 0]] - verts[:, e[:, 1]]).norm(dim=-1)


def main():
    smal = SMAL3DFitter(batch_size=1, device=device, shape_family=-1)
    faces = smal.faces.detach()

    with torch.no_grad():
        template_verts = smal().detach()  # zero pose, zero deform
        posed_rot = torch.tensor([[0.3, 0.15, -0.1]])
        posed_verts = smal(global_rot=posed_rot).detach()  # rigid rotation, zero deform

    template_edges = edge_lengths(template_verts, faces[0])
    posed_edges = edge_lengths(posed_verts, faces[0])
    print("--- sanity: rigid rotation preserves edge length ---")
    print(f"mean edge length  template={template_edges.mean():.6f}  posed={posed_edges.mean():.6f}")
    print(f"max abs relative change: {((posed_edges - template_edges).abs() / template_edges).max():.6f}")

    posed_mesh = get_meshes(posed_verts.clone(), faces.clone(), device=device)

    shrink_loss_at_perfect_fit = mesh_edge_loss(posed_mesh).item()
    rest_loss_at_perfect_fit = rest_edge_loss(posed_verts, posed_verts, faces[0]).item()

    print("\n--- check 1: perfect fit (correct pose, zero deform_verts) ---")
    print(f"shrink (mesh_edge_loss, target_length=0): {shrink_loss_at_perfect_fit:.6f}  (expected: large, nonzero)")
    print(f"rest   (rest_edge_loss):                  {rest_loss_at_perfect_fit:.6e}  (expected: ~0)")

    # Check 2: same posed mesh, add a known small deform_verts perturbation.
    with torch.no_grad():
        deform = 0.01 * torch.randn_like(smal.deform_verts)
        distorted_verts = smal(global_rot=posed_rot, deform_verts=deform).detach()

    distorted_mesh = get_meshes(distorted_verts.clone(), faces.clone(), device=device)
    shrink_loss_perturbed = mesh_edge_loss(distorted_mesh).item()
    rest_loss_perturbed = rest_edge_loss(distorted_verts, posed_verts, faces[0]).item()

    print("\n--- check 2: same pose + a small known deform_verts perturbation ---")
    print(f"shrink: {shrink_loss_at_perfect_fit:.6f} -> {shrink_loss_perturbed:.6f}  "
          f"(x{shrink_loss_perturbed / shrink_loss_at_perfect_fit:.3f})")
    print(f"rest:   {rest_loss_at_perfect_fit:.6e} -> {rest_loss_perturbed:.6f}  "
          f"(rest was ~0, now tracks the perturbation directly)")


if __name__ == "__main__":
    main()
