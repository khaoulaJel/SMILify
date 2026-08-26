"""
run_posthoc_repair_3specimen.py

TASK9 continuation: real repair (CPU detection swap + CG-based push) on the original 3
TASK6/TASK7 specimens (Acanthostichus_aff.brevicornis, Acromyrmex_coronatus,
Solenopsis_invicta -- TASK5-config control arm, fit3d_results_task7_symmetric_chamfer),
for direct comparability with those tables. Reduced from the full 10x3-seed set after a
single-specimen timing measurement showed ~997s/specimen (~8+ hours for the full 30-mesh
matrix) -- a real, bounded speed limitation of this approach, not a correctness question.

Also runs a determinism check: repairs one specimen (Acanthostichus) TWICE from the same
input, diffs the two outputs exactly.
"""
import os
import numpy as np
import fitter_3d.run_posthoc_repair_experiment as mod

SRC_NPZ = "fit3d_results_task7_symmetric_chamfer/control_raw_src_verts/Stage_3_deform_fine.npz"
OUT_ROOT = "fit3d_results_task9_repair_3specimen"


def main():
    mesh_names_all, target_meshes_all = mod.load_meshes(
        mesh_files=[os.path.join(mod.MESH_DIR, f) for f in sorted(os.listdir(mod.MESH_DIR)) if f.endswith(".obj")],
        device=mod.device,
    )
    target_by_name = {name: target_meshes_all[i] for i, name in enumerate(mesh_names_all)}

    data = np.load(SRC_NPZ, allow_pickle=True)
    verts_all = data["verts"]
    faces = data["faces"][0]
    labels = [str(l).replace(".obj", "") for l in data["labels"]]
    print(f"specimens: {labels}")

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

    # Determinism check: repair the first specimen a SECOND time from the same input,
    # diff the two outputs exactly.
    spec0 = labels[0]
    print(f"\n{'='*70}\nDETERMINISM CHECK: repairing {spec0} a second time\n{'='*70}")
    verts_a2, faces_a2 = mod.run_repair(verts_all[0], faces, os.path.join(OUT_ROOT, f"{spec0}_run2"), spec0)
    verts_a1, _ = np.load(os.path.join(OUT_ROOT, spec0, f"{spec0}_best.obj").replace(".obj", "_UNUSED")) if False else (None, None)
    import potpourri3d as pp3d
    verts_a1, _ = pp3d.read_mesh(os.path.join(OUT_ROOT, spec0, f"{spec0}_best.obj"))
    v_range = verts_all[0].max() - verts_all[0].min()
    verts_a1 = verts_a1 * v_range / 16.5
    max_abs_diff = np.max(np.abs(verts_a1 - verts_a2))
    mean_abs_diff = np.mean(np.abs(verts_a1 - verts_a2))
    print(f"determinism: max_abs_diff={max_abs_diff:.8f} mean_abs_diff={mean_abs_diff:.8f}")
    print("DETERMINISTIC (bit-identical)" if max_abs_diff == 0 else
          ("effectively deterministic (float noise)" if max_abs_diff < 1e-4 else "NOT deterministic"))


if __name__ == "__main__":
    main()
