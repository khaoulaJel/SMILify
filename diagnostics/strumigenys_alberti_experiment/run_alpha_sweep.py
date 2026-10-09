"""Direct CGAL alpha_wrap sweep on Strumigenys_alberti's pre_reconstruction.obj, tracking gap_after
for ALL genuine (normal_dot<-0.3) near-touch candidates found by probe_topk_near_touches.py - not
just the single (mistracked, artifact) landmark the production bisection guards. Answers: at what
alpha does each real near-touch actually get preserved, and what does that cost (vertex/face
count, runtime)?

Usage: python3 run_alpha_sweep.py   (run in the pytorch3d conda env - has trimesh/scipy/embreex)
"""
import os
import subprocess
import time
import numpy as np
import trimesh
from scipy.spatial import cKDTree

INPUT_OBJ = "/home/nao48500/SMILify/diagnostics/eyeball_groundtruth_20260817/06_FIDFAIL_Strumigenys_alberti/pre_reconstruction.obj"
BINARY = "/home/nao48500/SMILify/custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap"
WORK_DIR = "/home/nao48500/SMILify/diagnostics/strumigenys_alberti_experiment/sweep_work"
os.makedirs(WORK_DIR, exist_ok=True)

mesh = trimesh.load(INPUT_OBJ, process=False)
diag_length = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
print(f"diag_length={diag_length:.6g}")

# Genuine near-touch candidates from probe_topk_near_touches.py (normal_dot < -0.3), as
# (label, point_a, point_b, pre_gap)
CANDIDATES = [
    ("cand_leftfarcluster_A",  (-553.329, 373.323, 162.388), (-550.746, 376.307, 163.258), 4.041),
    ("cand_rightfarcluster_A", (500.403, -248.654, 17.331),  (501.690, -248.217, 21.930),  4.796),
    ("cand_rightfarcluster_B", (556.372, 15.116, -344.669),  (559.884, 18.150, -346.722),  5.075),
    ("cand_midlower",          (90.885, -464.780, -268.590), (93.982, -461.545, -265.883), 5.234),
    ("cand_rightfarcluster_C", (358.671, -281.712, -277.352),(358.083, -276.261, -277.261),5.483),
    ("cand_nearcore",          (-89.012, 8.958, 121.246),    (-87.956, 13.613, 124.310),   5.672),
    ("cand_leftfarcluster_B",  (-552.949, 376.465, 168.640), (-550.746, 376.307, 163.258), 5.818),
    ("cand_leftfarcluster_C",  (-528.255, 426.411, 142.535), (-526.154, 421.787, 145.641), 5.953),
]
OFFSET_ALPHA_RATIO = 2.5

# alpha values to test: current production alpha (relative_alpha=104.48) up to much finer
relative_alphas_to_test = [104.48, 200, 400, 800, 1600, 2500]

results = []
for rel_alpha in relative_alphas_to_test:
    alpha = diag_length / rel_alpha
    offset = alpha / OFFSET_ALPHA_RATIO
    relative_offset = diag_length / offset
    out_label = f"sweep_relalpha{int(rel_alpha)}"
    cmd = (
        f"cd {WORK_DIR} && {BINARY} {INPUT_OBJ} {rel_alpha} {relative_offset}"
    )
    t0 = time.time()
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    input_base = os.path.splitext(os.path.basename(INPUT_OBJ))[0]
    predicted_path = os.path.join(WORK_DIR, f"{input_base}_{int(rel_alpha)}_{int(relative_offset)}.off")
    if result.returncode != 0 or not os.path.isfile(predicted_path):
        print(f"relative_alpha={rel_alpha}: FAILED exit={result.returncode} elapsed={elapsed:.1f}s "
              f"stderr_tail={result.stderr[-300:]}")
        continue
    out_path = os.path.join(WORK_DIR, f"{out_label}.off")
    os.replace(predicted_path, out_path)

    out_mesh = trimesh.load(out_path, process=False)
    tree = cKDTree(out_mesh.vertices)

    gaps_after = {}
    for label, pa, pb, pre_gap in CANDIDATES:
        _, ia = tree.query(pa)
        _, ib = tree.query(pb)
        gap_after = float(np.linalg.norm(out_mesh.vertices[ia] - out_mesh.vertices[ib]))
        verdict = "fused" if ia == ib else ("suspect" if gap_after < pre_gap * 0.3 else "preserved")
        gaps_after[label] = (gap_after, verdict)

    n_preserved = sum(1 for v in gaps_after.values() if v[1] == "preserved")
    print(f"relative_alpha={rel_alpha:>7.1f} alpha={alpha:.4g} elapsed={elapsed:.1f}s "
          f"verts={len(out_mesh.vertices)} faces={len(out_mesh.faces)} "
          f"watertight={out_mesh.is_watertight} n_components={out_mesh.body_count} "
          f"n_candidates_preserved={n_preserved}/{len(CANDIDATES)}")
    for label, (gap_after, verdict) in gaps_after.items():
        print(f"    {label}: gap_after={gap_after:.4g} verdict={verdict}")

    results.append({
        "relative_alpha": rel_alpha, "alpha": alpha, "elapsed": elapsed,
        "verts": len(out_mesh.vertices), "faces": len(out_mesh.faces),
        "watertight": out_mesh.is_watertight, "n_components": out_mesh.body_count,
        "n_preserved": n_preserved, "gaps_after": gaps_after,
    })

print()
print("=== SUMMARY ===")
for r in results:
    print(f"relative_alpha={r['relative_alpha']:>7.1f} n_preserved={r['n_preserved']}/{len(CANDIDATES)} "
          f"verts={r['verts']:>8d} elapsed={r['elapsed']:.1f}s watertight={r['watertight']}")
