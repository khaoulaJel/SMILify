"""
run_density_ratio_probe.py

Bench50 extension of the legs/gaster vertex-density-ratio check computed ad hoc
on the 3 TASK6/7 specimens. Sanity-checked first (not assumed): Stage_2's ratio
agrees with the full Stage_3 fit to within 7.8%/9.3%/-2.3% on those 3 specimens
and preserves rank order, while a cheaper Stage_0/Stage_1 (no-deform) proxy was
tried first and rejected -- up to +149% error and a flipped rank order, since
deform_verts (only active from Stage_2 on) is what actually corrects a
specimen's individual leg/gaster proportions away from the shared template.
So this runs through Stage_2 (deform-coarse) only, not the full Stage_0-3
schedule -- cheaper than a full fit, validated as close enough for this
specific measurement.

Plain baseline weights (no w_penetration/w_offset/w_scale/w_trans) -- this
measures each specimen's own fitted shape/proportions, not a penetration-loss
outcome, so there's no reason to carry TASK5's penetration-specific stack here.

Batched in chunks (default 10) to stay within this GPU's 6GB budget -- batch 50
was not verified to fit and chunking is cheap insurance, not a proven necessity.

Output: diagnostics/density_ratio_bench50.csv, columns specimen, legs_area,
gaster_area, legs_density, gaster_density, ratio.
"""
import os

import numpy as np
import torch

import config
if os.environ.get("SMIL_DISABLE_PLOTTING"):
    config.PLOT_RESULTS = False

from fitter_3d.utils import load_meshes
from fitter_3d.trainer import SMAL3DFitter, Stage
from fitter_3d.part_groups import get_part_vertex_indices, PART_GROUPS_COARSE
from fitter_3d.penetration_loss import _build_part_faces

device = "cuda" if torch.cuda.is_available() else "cpu"

MESH_DIR = "diagnostics/moonshot/bench50_clean"
OUT_DIR = os.environ.get("DENSITY_PROBE_OUT", "fit3d_results_density_ratio_probe")
CHUNK_SIZE = int(os.environ.get("DENSITY_PROBE_CHUNK", 10))

STAGE0_NITS = int(os.environ.get("DENSITY_PROBE_STAGE0_NITS", 100))
STAGE1_NITS = int(os.environ.get("DENSITY_PROBE_STAGE1_NITS", 300))
STAGE2_NITS = int(os.environ.get("DENSITY_PROBE_STAGE2_NITS", 1000))

STAGE1_LOSS_WEIGHTS = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.02, w_laplacian=0.01, w_limit=100.0, w_sdf=0.0)
STAGE2_LOSS_WEIGHTS = dict(w_chamfer=1.0, w_edge=0.8, w_normal=0.005, w_laplacian=0.01, w_limit=100.0, w_sdf=0.0)

part_vertex_indices = get_part_vertex_indices(PART_GROUPS_COARSE)


def mesh_area(v, f):
    tri = v[f]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum()


def run_chunk(mesh_files, out_dir):
    mesh_names, target_meshes = load_meshes(mesh_files=mesh_files, device=device)
    os.makedirs(out_dir, exist_ok=True)
    fitter = SMAL3DFitter(batch_size=len(mesh_names), device=device, shape_family=-1)

    stage0 = Stage(nits=STAGE0_NITS, scheme="init_rot_lock", smal_3d_fitter=fitter, target_meshes=target_meshes,
                   mesh_names=mesh_names, name="Stage_0_init", lr=0.05, out_dir=out_dir, device=device)
    stage0.run()

    stage1 = Stage(nits=STAGE1_NITS, scheme="default", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_1_default", lr=0.02, out_dir=out_dir, device=device,
                    loss_weights=STAGE1_LOSS_WEIGHTS, custom_lrs={"joint_rot": 0.002})
    stage1.run()

    stage2 = Stage(nits=STAGE2_NITS, scheme="all", smal_3d_fitter=fitter, target_meshes=target_meshes,
                    mesh_names=mesh_names, name="Stage_2_deform_coarse", lr=0.002, out_dir=out_dir, device=device,
                    loss_weights=STAGE2_LOSS_WEIGHTS)
    stage2.run()

    verts = stage2.smal_3d_fitter().detach().cpu().numpy()
    faces = stage2.faces[0].detach().cpu().numpy()
    part_faces = _build_part_faces(faces, part_vertex_indices)

    rows = []
    for i, name in enumerate(mesh_names):
        v = verts[i]
        legs_area = mesh_area(v, part_faces["legs"])
        gaster_area = mesh_area(v, part_faces["gaster"])
        legs_density = len(part_vertex_indices["legs"]) / legs_area
        gaster_density = len(part_vertex_indices["gaster"]) / gaster_area
        rows.append((name, legs_area, gaster_area, legs_density, gaster_density, legs_density / gaster_density))
    return rows


def main():
    mesh_files = sorted(
        os.path.join(MESH_DIR, f) for f in os.listdir(MESH_DIR) if f.endswith(".obj")
    )
    print(f"Found {len(mesh_files)} specimens, running in chunks of {CHUNK_SIZE}")

    all_rows = []
    for c in range(0, len(mesh_files), CHUNK_SIZE):
        chunk = mesh_files[c:c + CHUNK_SIZE]
        chunk_out = os.path.join(OUT_DIR, f"chunk_{c // CHUNK_SIZE}")
        print(f"\n=== chunk {c // CHUNK_SIZE}: {len(chunk)} specimens -> {chunk_out} ===")
        all_rows.extend(run_chunk(chunk, chunk_out))

    out_csv = os.path.join(OUT_DIR, "density_ratio_bench50.csv") if os.path.isdir(OUT_DIR) else "density_ratio_bench50.csv"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "density_ratio_bench50.csv")
    with open(out_csv, "w") as f:
        f.write("specimen,legs_area,gaster_area,legs_density,gaster_density,ratio\n")
        for name, la, ga, ld, gd, r in all_rows:
            f.write(f"{name},{la},{ga},{ld},{gd},{r}\n")
    print(f"\nWrote {out_csv} ({len(all_rows)} specimens)")


if __name__ == "__main__":
    main()
