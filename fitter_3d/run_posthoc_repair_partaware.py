"""
run_posthoc_repair_partaware.py

TASK9's last, bounded follow-up: filter detected self-intersections down to genuinely
non-adjacent part pairs (the same convention penetration_loss.py/bvh_penetration_loss.py
already use) before the repair pushes on them, instead of the whole-mesh-unfiltered behaviour
that produced TASK9's -0.152 mean F-score cost. Same 3 specimens, same energy/constraints,
same everything else -- single-variable swap, one more time.

The face-part mapping is computed ONCE from the shared template topology (fixed across
specimens) and reused for all 3 -- cheap, matching the "try it once, cheaply" scope this was
given.
"""
import os
import numpy as np

import fitter_3d.run_posthoc_repair_experiment as mod
import repair_factory
from fitter_3d.cpu_self_intersection import CPUSelfIntersectionDetector
from fitter_3d.bvh_penetration_loss import build_face_part_ids
from fitter_3d.part_groups import get_part_vertex_indices, get_non_adjacent_pairs, PART_GROUPS_COARSE

SRC_NPZ = "fit3d_results_task7_symmetric_chamfer/control_raw_src_verts/Stage_3_deform_fine.npz"
OUT_ROOT = "fit3d_results_task9_repair_partaware"


def main():
    data = np.load(SRC_NPZ, allow_pickle=True)
    faces = data["faces"][0]

    part_vertex_indices = get_part_vertex_indices(PART_GROUPS_COARSE)
    face_part_id, part_name_to_idx = build_face_part_ids(faces, part_vertex_indices)
    allowed_part_pairs = {
        frozenset((part_name_to_idx[a], part_name_to_idx[b]))
        for a, b in get_non_adjacent_pairs(PART_GROUPS_COARSE)
    }
    print(f"allowed non-adjacent part pairs: {len(allowed_part_pairs)}")

    # Monkeypatch a factory closure, not the bare class -- repair_factory.main() calls
    # BVH(config['max_collisions']) itself, so this needs to accept that single positional
    # arg and inject the fixed face_part_id/allowed_part_pairs on construction.
    def _factory(max_collisions=None):
        return CPUSelfIntersectionDetector(
            max_collisions=max_collisions,
            face_part_id=face_part_id,
            allowed_part_pairs=allowed_part_pairs,
        )

    repair_factory.BVH = _factory

    verts_all = data["verts"]
    labels = [str(l).replace(".obj", "") for l in data["labels"]]
    print(f"specimens: {labels}")

    mesh_names_all, target_meshes_all = mod.load_meshes(
        mesh_files=[os.path.join(mod.MESH_DIR, f) for f in sorted(os.listdir(mod.MESH_DIR)) if f.endswith(".obj")],
        device=mod.device,
    )
    target_by_name = {name: target_meshes_all[i] for i, name in enumerate(mesh_names_all)}

    rows = []
    for i, spec in enumerate(labels):
        print(f"\n{'='*70}\n{spec}\n{'='*70}")
        verts_before = verts_all[i]
        target_mesh = target_by_name[spec]

        hard_b, soft_b = mod.penetration_counts(verts_before, faces)
        f_b = mod.compute_fscore(verts_before, faces, target_mesh)
        print(f"before: hard={hard_b} soft={soft_b:.1f} f_score={f_b:.4f}")

        savepath = os.path.join(OUT_ROOT, spec)
        verts_after, faces_after = mod.run_repair(verts_before, faces, savepath, spec)

        hard_a, soft_a = mod.penetration_counts(verts_after, faces_after)
        f_a = mod.compute_fscore(verts_after, faces_after, target_mesh)
        edge_dist = mod.edge_logratio(verts_before, verts_after, faces)
        print(f"after:  hard={hard_a} soft={soft_a:.1f} f_score={f_a:.4f} edge_logratio={edge_dist:.5f}")

        rows.append(dict(specimen=spec, hard_before=hard_b, hard_after=hard_a,
                          soft_before=soft_b, soft_after=soft_a,
                          f_score_before=f_b, f_score_after=f_a, edge_logratio=edge_dist))

    import pandas as pd
    os.makedirs(OUT_ROOT, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_ROOT, "repair_before_after.csv"), index=False)
    print(f"\nWrote {OUT_ROOT}/repair_before_after.csv")


if __name__ == "__main__":
    main()
