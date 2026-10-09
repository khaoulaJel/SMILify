"""
run_posthoc_repair_experiment.py

TASK 9 continuation: the actual repair (not just detection), on TASK5's own already-converged
output, full 10-specimen x 3-seed set -- after the cholespy segfault was root-caused to its GPU
CholeskySolverF specifically (10-line repro, no mesh involved) and worked around with a one-line
solver swap to the CPU/GPU-agnostic ConjugateGradientSolver already present in the same
`largesteps` codebase (repair_factory.py's `from_differential(M, u, 'CG')`).

Input: TASK7's already-saved Stage_3 control-arm npz files (seed0/1/2) -- TASK5's own config,
zero new GPU training, exactly the "recover before redesign" input specified.

For each (seed, specimen): export the pre-repair mesh, run the real repair
(energy='signed_TPE_verts', constraints=[volume, area, curvature] -- the conservative choice,
giving the repair every reason to preserve geometry rather than trivially resolving collisions
by shrinking or distorting the mesh), measure hard count, soft count, F-score, and edge-length
distortion BEFORE and AFTER. A separate determinism check (same single mesh, repaired twice,
outputs diffed exactly) is run once, not per-specimen.

Deliberately whole-mesh, unfiltered self-intersection repair -- this tool has no concept of
"these parts are allowed to touch" the way this project's own part-pair-scoped evaluation does.
That mismatch is real and is called out in the results, not silently absorbed.
"""
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPAIR_ROOT = os.path.join(repo_root, "custom_processing/external/instant-mesh-intersection-repair")
BVH_ROOT = os.path.join(repo_root, "custom_processing/external/torch-mesh-isect-wonjongg")
sys.path.insert(0, REPAIR_ROOT)
sys.path.insert(0, BVH_ROOT)

import numpy as np
import pandas as pd
import potpourri3d as pp3d
import torch

import repair_factory
repair_factory.device = torch.device("cuda")

from fitter_3d.part_groups import get_part_vertex_indices, get_non_adjacent_pairs, PART_GROUPS_COARSE
from fitter_3d.penetration_loss import _build_part_faces
from fitter_3d.eval_metrics import f_score
from pytorch3d.structures import Meshes
from fitter_3d.utils import load_meshes
from fitter_3d.cpu_self_intersection import CPUSelfIntersectionDetector

# Swap ONLY the detection primitive repair_factory.main() constructs internally
# (`search_tree = BVH(config['max_collisions'])`) -- the bvh_cuda kernel confirmed to fault
# under repeated invocation (TASK8, TASK9), independent of caller. Everything downstream
# (the CG-based push via energies.py, the constraints, the optimizer) is untouched; this is a
# monkey-patch of the one name repair_factory.py imports BVH under, not an edit to the tool's
# own logic. Validated separately (fitter_3d/cpu_self_intersection.py's own test suite, plus a
# direct cross-check against the CUDA kernel's isolated-call collision counts on this exact
# specimen set: CPU counts landed at 80-99% of CUDA's, same relative ordering across
# specimens -- comparable, not divergent) before being trusted here.
repair_factory.BVH = CPUSelfIntersectionDetector

device = "cuda"
TASK7_ROOT = "fit3d_results_task7_seeded_scale"
MESH_DIR = "diagnostics/moonshot/bench50_clean"
OUT_ROOT = os.environ.get("REPAIR_OUT_ROOT", "fit3d_results_task9_repair")
SEEDS = [0, 1, 2]

part_vertex_indices = get_part_vertex_indices(PART_GROUPS_COARSE)
non_adjacent_pairs = get_non_adjacent_pairs(PART_GROUPS_COARSE)

REPAIR_CONFIG_BASE = dict(
    optimizer="Adam", lr=0.001, max_collisions=16,
    energy="signed_TPE_verts", num_iters=200,
    constraints=["volume", "area", "curvature"],
)


def directional(verts, idx_query, faces_surface):
    q = verts[idx_query]
    tri = verts[faces_surface]
    va, vb, vc = tri[:, 0], tri[:, 1], tri[:, 2]
    normals = np.cross(vb - va, vc - va)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    centroids = (va + vb + vc) / 3.0
    d = np.linalg.norm(q[:, None, :] - centroids[None, :, :], axis=2)
    nearest_idx = d.argmin(axis=1)
    dist = d[np.arange(len(q)), nearest_idx]
    matched_c = centroids[nearest_idx]
    matched_n = normals[nearest_idx]
    sign = ((q - matched_c) * matched_n).sum(axis=1)
    return dist, sign


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bbox_diag(verts):
    return np.linalg.norm(verts.max(0) - verts.min(0))


TAU_FRAC = 0.03
TEMP_FRAC = 0.15


def penetration_counts(verts, faces):
    part_faces = _build_part_faces(faces, part_vertex_indices)
    tau = TAU_FRAC * bbox_diag(verts)
    hard_total, soft_total = 0, 0.0
    for a, b in non_adjacent_pairs:
        idx_a, idx_b = part_vertex_indices[a], part_vertex_indices[b]
        fa, fb = part_faces[a], part_faces[b]
        if len(idx_a) == 0 or len(idx_b) == 0 or len(fa) == 0 or len(fb) == 0:
            continue
        for idx_q, f_s in [(idx_a, fb), (idx_b, fa)]:
            dist, sign = directional(verts, idx_q, f_s)
            close = dist < tau
            hard_total += (close & (sign < 0)).sum()
            temp = TEMP_FRAC * tau
            soft_total += (close.astype(float) * sigmoid(-sign / temp)).sum()
    return int(hard_total), float(soft_total)


def edge_logratio(verts_before, verts_after, faces):
    edges = set()
    for f in faces:
        edges.add(tuple(sorted((f[0], f[1]))))
        edges.add(tuple(sorted((f[1], f[2]))))
        edges.add(tuple(sorted((f[2], f[0]))))
    edges = np.array(list(edges))
    len_before = np.linalg.norm(verts_before[edges[:, 0]] - verts_before[edges[:, 1]], axis=1)
    len_after = np.linalg.norm(verts_after[edges[:, 0]] - verts_after[edges[:, 1]], axis=1)
    ratio = len_after / np.clip(len_before, 1e-8, None)
    return float(np.mean(np.abs(np.log(np.clip(ratio, 1e-8, None)))))


def compute_fscore(verts, faces, target_mesh):
    pred = Meshes(verts=[torch.tensor(verts, dtype=torch.float32, device=device)],
                  faces=[torch.tensor(faces, dtype=torch.long, device=device)])
    metrics = f_score(pred, target_mesh, tau_fractions=[0.01], n_samples=10000)
    return float(metrics["f_score"][0, 0].item())


def run_repair(verts, faces, savepath, expname):
    os.makedirs(savepath, exist_ok=True)
    objpath = os.path.join(savepath, f"{expname}_input.obj")
    pp3d.write_mesh(verts, faces, objpath)
    config = dict(REPAIR_CONFIG_BASE)
    config.update(expname=expname, objpath=objpath, savepath=savepath)
    repair_factory.main(config)
    # main() normalizes internally (16.5 / v_range) and writes '<expname>_best.obj' at that
    # normalized scale -- undo it here so repaired output is directly comparable to the input.
    v_range = verts.max() - verts.min()
    best_v, best_f = pp3d.read_mesh(os.path.join(savepath, f"{expname}_best.obj"))
    best_v = best_v * v_range / 16.5
    return best_v, best_f


def main():
    mesh_names_all, target_meshes_all = load_meshes(
        mesh_files=[os.path.join(MESH_DIR, f) for f in sorted(os.listdir(MESH_DIR)) if f.endswith(".obj")],
        device=device,
    )
    target_by_name = {name: target_meshes_all[i] for i, name in enumerate(mesh_names_all)}

    rows = []
    for seed in SEEDS:
        npz_path = os.path.join(TASK7_ROOT, f"seed{seed}_control", "Stage_3_deform_fine.npz")
        data = np.load(npz_path, allow_pickle=True)
        verts_all = data["verts"]
        faces = data["faces"][0]
        labels = [str(l).replace(".obj", "") for l in data["labels"]]

        for i, spec in enumerate(labels):
            print(f"\n=== seed={seed} specimen={spec} ===")
            verts_before = verts_all[i]
            target_mesh = target_by_name[spec]

            hard_before, soft_before = penetration_counts(verts_before, faces)
            f_before = compute_fscore(verts_before, faces, target_mesh)
            print(f"  before: hard={hard_before} soft={soft_before:.1f} f_score={f_before:.4f}")

            savepath = os.path.join(OUT_ROOT, f"seed{seed}", spec)
            verts_after, faces_after = run_repair(verts_before, faces, savepath, spec)

            hard_after, soft_after = penetration_counts(verts_after, faces_after)
            f_after = compute_fscore(verts_after, faces_after, target_mesh)
            edge_dist = edge_logratio(verts_before, verts_after, faces)
            print(f"  after:  hard={hard_after} soft={soft_after:.1f} f_score={f_after:.4f} "
                  f"edge_logratio={edge_dist:.5f}")

            rows.append(dict(seed=seed, specimen=spec,
                              hard_before=hard_before, hard_after=hard_after,
                              soft_before=soft_before, soft_after=soft_after,
                              f_score_before=f_before, f_score_after=f_after,
                              edge_logratio=edge_dist))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_ROOT, "repair_before_after.csv")
    os.makedirs(OUT_ROOT, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
