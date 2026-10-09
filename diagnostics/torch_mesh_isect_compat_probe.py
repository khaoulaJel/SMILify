"""
torch_mesh_isect_compat_probe.py

GPU smoke test for the BVH triangle-triangle collision extension (`bvh_cuda`,
from either EthanFifle/torch-mesh-intersection or the original
vchoutas/torch-mesh-isect). Confirms the compiled extension not only IMPORTS
but produces the CORRECT collision verdict at actual kernel-launch time --
compiled-CUDA-op ABI mismatches (this project pins torch 2.3.1/cu118; the
original repo was tested against torch 1.0/CUDA 10.0, the fork against torch
2.0/CUDA 11.7) are often invisible on a plain import and only surface when a
kernel actually runs.

Not a training-loss integration test -- this only answers "does the compiled
BVH extension run and classify triangle-triangle intersection correctly on
this environment's GPU," the practical prerequisite before spending any time
wiring it into fitter_3d/trainer.py as a third arm alongside centroid-
proximity and GWN.

Geometry (single 4-triangle "mesh", run through BVH.forward once):
  - triangle 0: (0,0,0),(1,0,0),(0,1,0) in the z=0 plane.
  - triangle 1: (0.2,0.2,-0.5),(0.3,0.2,0.5),(0.2,0.3,0.5) -- one vertex
    below the z=0 plane, two above. Deliberately NOT edge/vertex-touching:
    the segment where triangle 1 pierces the z=0 plane runs from
    (0.25, 0.2, 0) to (0.2, 0.25, 0) (both computed analytically below,
    linear interpolation along edges v0-v1 and v0-v2 at t=0.5), and both
    endpoints satisfy x>0, y>0, x+y<1 -- strictly inside triangle 0's
    interior, not on its boundary. This is an unambiguous "pierces through
    the middle" intersection, chosen specifically to rule out a
    boundary/edge-case false negative before trusting any FAIL from this
    probe as a real extension bug.
  - triangle 2, triangle 3: two triangles far away (centered near (10,10,10)
    and (20,20,20)) that intersect neither triangle 0/1 nor each other.

Expected verdict: the collision-pair output contains (0,1) or (1,0) and
nothing involving faces 2 or 3. A second pass translates triangle 1 far away
(same shape, offset) and re-checks that the collision then disappears --
rules out a BVH that always reports a fixed/stale collision.
"""
import sys

import torch

print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FAIL: no CUDA device visible to torch -- cannot test a CUDA kernel.")
    sys.exit(1)
print(f"device: {torch.cuda.get_device_name(0)}")

try:
    from mesh_intersection.bvh_search_tree import BVH
except ImportError as exc:
    print(f"FAIL: could not import mesh_intersection.bvh_search_tree.BVH ({exc})")
    sys.exit(1)

device = "cuda"


def make_triangles(offset_t1=(0.0, 0.0, 0.0)):
    t0 = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
    )
    t1 = torch.tensor(
        [[0.2, 0.2, -0.5], [0.3, 0.2, 0.5], [0.2, 0.3, 0.5]], device=device
    ) + torch.tensor(offset_t1, device=device)
    t2 = torch.tensor(
        [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]], device=device
    )
    t3 = torch.tensor(
        [[20.0, 20.0, 20.0], [21.0, 20.0, 20.0], [20.0, 21.0, 20.0]], device=device
    )
    return torch.stack([t0, t1, t2, t3], dim=0).unsqueeze(0)  # [1, 4, 3, 3]


def collision_pairs(triangles):
    bvh = BVH(max_collisions=8)
    out = bvh(triangles)
    out = out.detach().cpu().numpy() if torch.is_tensor(out) else out
    pairs = set()
    for row in out.reshape(-1, 2):
        a, b = int(row[0]), int(row[1])
        if a >= 0 and b >= 0:
            pairs.add(tuple(sorted((a, b))))
    return pairs


ok = True

print("--- pass 0: two EXACTLY coincident triangles (maximal, unambiguous overlap) ---")
t0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
dup_pairs = collision_pairs(torch.stack([t0, t0.clone()], dim=0).unsqueeze(0))
print(f"collision pairs: {sorted(dup_pairs)}")
if (0, 1) not in dup_pairs:
    print("FAIL: two IDENTICAL, fully-coincident triangles were not flagged as colliding.")
    print("      This is the most extreme possible positive case -- if this fails, the")
    print("      kernel is not detecting intersections at all on this build, independent")
    print("      of any geometric edge case in the other passes below.")
    ok = False
else:
    print("PASS: exact duplicate correctly flagged as colliding.")

print("\nanalytic ground truth: triangle 1 pierces the z=0 plane along the")
print("segment (0.25, 0.20, 0.0) -- (0.20, 0.25, 0.0), both points strictly")
print("inside triangle 0's interior (x>0, y>0, x+y<1) -- a genuine, non-")
print("boundary 3D intersection, independent of the BVH extension.")

print("\n--- pass 1: triangle 1 intersecting triangle 0 ---")
pairs = collision_pairs(make_triangles())
print(f"collision pairs: {sorted(pairs)}")
if (0, 1) not in pairs:
    print("FAIL: expected collision (0,1) not detected.")
    ok = False
spurious = {p for p in pairs if p != (0, 1)}
if spurious:
    print(f"FAIL: unexpected collision pairs involving faces 2/3: {spurious}")
    ok = False
if (0, 1) in pairs and not spurious:
    print("PASS: correct collision detected, no spurious collisions.")

print("\n--- pass 2: triangle 1 translated away, collision should vanish ---")
pairs2 = collision_pairs(make_triangles(offset_t1=(50.0, 50.0, 50.0)))
print(f"collision pairs: {sorted(pairs2)}")
if pairs2:
    print(f"FAIL: expected no collisions once separated, got {pairs2}")
    ok = False
else:
    print("PASS: no spurious collision once triangles are separated.")

print(f"\n=== OVERALL: {'PASS' if ok else 'FAIL'} ===")
sys.exit(0 if ok else 1)
