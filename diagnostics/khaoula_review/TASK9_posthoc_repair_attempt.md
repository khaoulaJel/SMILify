# TASK 9 — post-hoc repair (the actual SIGGRAPH 2025 method, not just its detection primitive)

Attempts the structurally-different avenue TASK6/7/8 all lacked: a deterministic, static-mesh
repair on TASK5's own already-converged output, with no training-time gradient competition at
all -- proposed specifically because TASK6 (GWN, rough gradients), TASK7 (symmetric-chamfer,
seed instability), and TASK8 (BVH-as-training-signal, training-graph CUDA fault) all failed for
reasons specific to optimizing *during* differentiable fitting, which this approach was meant to
sidestep entirely.

## What was set up

`wonjongg/instant-mesh-intersection-repair` cloned in full (not just its `torch-mesh-isect`
submodule, already built and confirmed working for detection in TASK8). Its two additional
dependencies (`potpourri3d`, `cholespy` -- both prebuilt wheels, no compilation needed) installed
cleanly. `largesteps` (the Laplacian/differential-coordinate reparameterization the tool uses for
stable large-step mesh optimization, Nicolet et al. 2021) ships bundled in the repo, no separate
install. 10 specimens exported as `.obj` from TASK7's already-saved seed0-control Stage_3 npz
(TASK5's own config, zero new GPU training -- exactly the "recover before redesign" input this
task called for).

## What happened

**Segfault, not a Python-catchable exception, on the very first repair attempt** -- and,
decisively, also on the tool's own bundled example mesh (`configs/misc.yaml` /
`data/misc/celtic_knot.obj`) with its own recommended default settings. That rules out anything
specific to the ant mesh, the earlier BVH work, or this project's own code before any further
digging was needed.

`python -X faulthandler` (prints a stack trace through a segfault, when possible) pinned it
exactly: `largesteps/solvers.py:34`, inside `cholespy.CholeskySolverF.__init__`, called with a
GPU sparse-tensor's indices/values (`M.indices()[0]`, `M.indices()[1]`, `M.values()` from a CUDA
`torch.sparse_coo_tensor`). Minimal, decisive repro (10 lines, no mesh, no repair code, no other
dependency involved):

```python
solver = CholeskySolverF(4, rows, cols, vals, MatrixType.COO)  # rows/cols/vals on CPU: works
solver = CholeskySolverF(4, rows, cols, vals, MatrixType.COO)  # same, on CUDA: segfaults instantly
```

CPU tensors: works. CUDA tensors, same trivial 4x4 identity-like system: segfaults immediately,
no mesh or geometry involved at all. **This is not a scale, config, or data problem — it's the
pip-installed `cholespy` wheel's GPU code path being broken on this environment**, confirmed in
under 10 lines rather than after a long chase.

## Reading

Same underlying class of problem as GWN, both `torch-mesh-isect` forks, and now this: a
prebuilt CUDA extension, built and tested against a materially older stack than this project's
(torch 2.5.1+cu124, Ampere/compute_86), failing in a way specific to this environment. What's
different and worth naming explicitly: **every third-party GPU-accelerated geometry-processing
dependency this investigation has tried to adopt has hit some version of this same wall.** That's
no longer a coincidence about any one library — it's a property of trying to bring in
niche, research-code CUDA extensions (rarely maintained past their original paper) onto this
project's specific, comparatively recent stack. Rebuilding `cholespy` from source against this
environment is a real, scoped possible next step (unlike stepping through someone else's
undocumented training-graph CUDA fault, as TASK8 hit) -- not attempted here, since it's a new
build effort of unknown size (cholespy's own native dependencies, e.g. SuiteSparse), not a
quick fix, and this task's job was to attempt the repair, not to become a third build-debugging
detour in the same session.

## Follow-up: the solver swap, and what it actually revealed

Rather than rebuilding `cholespy`, swapped to `largesteps.solvers.ConjugateGradientSolver` --
already present in the same file, pure PyTorch, no compiled dependency. Genuinely a one-line
change: `from_differential(M, u, 'Cholesky')` -> `from_differential(M, u, 'CG')`
(`repair_factory.py`'s `main()`, the only call site actually used).

**The swap works.** Re-ran the tool's own bundled example (`configs/misc.yaml`) with no other
changes: 809 self-intersecting face pairs at the start, 0 by iteration 28, clean exit. The
segfault is fully resolved -- confirms it really was `cholespy`'s GPU path specifically, nothing
else in the pipeline.

**Running the real repair on the actual ant mesh (TASK7's seed0-control output, same specimens
this whole task set has used) hits a DIFFERENT, already-seen fault**: the exact TASK8 crash,
`Cuda failure .../bvh_cuda_op.cu:990: 'an illegal memory access was encountered'`. Reproduced
three times, independently, before concluding anything:

1. Inside this project's own `bvh_penetration_loss_batched` integration, combined with the rest
   of a real training step (TASK8's original finding).
2. Inside `repair_factory.py`'s own optimization loop -- a completely different, much simpler
   caller (BVH search + one energy term + constraints, no other losses, no training-step graph
   at all) -- same crash, same file:line.
3. In a fresh Python process with zero prior GPU allocations (no target meshes loaded, no other
   computation), specifically to rule out a GPU-memory-pressure confound from work done earlier
   in the same process. Same crash again.

**This changes TASK8's own diagnosis.** TASK8 concluded the fault was specific to combining the
BVH-derived loss with other losses in one training graph. Reproduction #2 rules that out directly
-- there is no other loss here, no training graph, just BVH called repeatedly inside an
optimization loop. The real pattern, now supported by three independent, differently-coded
callers: **single/isolated BVH calls on this real mesh work reliably (confirmed repeatedly across
TASK8 and this task); calling it repeatedly inside ANY iterative loop, regardless of what else is
in that loop, eventually crashes.** Consistent with a resource leak inside the compiled kernel
across repeated invocations (each call allocating something the previous one didn't release)
rather than a graph-composition or memory-pressure issue -- plausible, not confirmed; stepping
further into the kernel itself is the open-ended debugging this project has consistently declined
to do without new direction, and still declines here.

## Second follow-up: swap the detection primitive, not just the solver

The repair pipeline has two separable pieces: detect which triangles intersect, then push to
resolve them. Only detection touches the faulting `bvh_cuda` kernel. `fitter_3d/cpu_self_intersection.py`
implements a CPU-only, no-CUDA drop-in for `mesh_intersection.bvh_search_tree.BVH` (same
`__call__` contract) -- broad-phase via `trimesh`'s AABB tree (`rtree`-backed), narrow-phase via
6 vectorized Moller-Trumbore segment-triangle tests per candidate pair (each triangle's 3 edges
against the other, both directions). One real bug caught and fixed before trusting it: the first
version flagged every ordinarily-adjacent face pair (two triangles sharing a mesh vertex, normal
manifold topology) as "intersecting" -- ~100,000+ spurious pairs per specimen. Fixed by excluding
candidate pairs that share a vertex (exact float equality is the correct check here, not a
tolerance hack, since `verts[faces]`-exploded triangles reuse the identical bit values for a
shared vertex index).

**Validated two ways before trusting it on real data**, matching this project's own "probe
before editing" discipline: (1) the exact same synthetic suite used to validate the CUDA kernel
(coincident triangles, hand-verified interior-piercing pair, separated-pair negative control) --
same result pattern, including the same known coincident-triangle blind spot any edge-crossing
test has; (2) a direct cross-check against the CUDA kernel's own isolated-call collision counts
on the real 10-specimen warmed-up state (TASK8's methodology) -- CPU counts landed at 80-99% of
CUDA's across all 10 specimens, same relative ordering, not divergent.

Swapped into the repair pipeline via a one-line monkeypatch (`repair_factory.BVH =
CPUSelfIntersectionDetector`) -- zero edits to the tool's own logic, exactly matching "swap the
detection step, keep everything else unchanged."

## The real result

Ran to completion on the original 3 TASK6/TASK7 specimens (Acanthostichus, Acromyrmex_coronatus,
Solenopsis_invicta -- TASK5-config control arm) plus a determinism check (Acanthostichus,
repaired twice from the same input).

| specimen | hard: before->after | soft: before->after | F-score: before->after | edge distortion |
|---|---|---|---|---:|
| Acanthostichus | 755->434 (-42.5%) | 1194->685 (-42.6%) | 0.890->0.741 (**-0.149**) | 0.034 |
| Acromyrmex_coronatus | 964->**1352** (**+40.2%**) | 1293->1434 (+10.9%) | 0.673->0.544 (**-0.129**) | 0.054 |
| Solenopsis_invicta | 305->209 (-31.5%) | 394->243 (-38.3%) | 0.681->0.502 (**-0.179**) | 0.057 |

**Determinism: confirmed, exactly.** Two independent runs of Acanthostichus from the same input
produced bit-identical output (max abs diff across every vertex: 0.0). The one part of the
structural pitch that was a pure procedural claim, not an empirical bet, holds completely.

**The outcome does not.** Mean F-score cost: **-0.152** -- far worse than TASK5's own -0.01 bar,
worse than TASK6's worst single-specimen case (-0.078), worse than every other arm across this
entire investigation. Count doesn't even reliably improve: Acromyrmex regresses by +40.2%,
a real effect, not noise (2 specimens improve, the third gets substantially worse -- the same
"squeezed balloon" shape TASK1/TASK4 found for training-time approaches, now showing up in a
completely training-instability-free setting). Aggregate hard count nets out to a barely-there
-1.4% only because the three specimens partly cancel, which masks the real per-specimen
variance rather than indicating any stability.

**Reading**: this tool does whole-mesh, unfiltered self-intersection repair -- it has no concept
of "these parts are supposed to touch," the mismatch flagged as a real risk when this task was
first scoped. It's resolving legitimate anatomical contact (leg-to-thorax attachment seams,
etc.) alongside genuine problem penetration, indiscriminately, and the volume/area/curvature
constraints available in this tool aren't enough to prevent that from badly distorting the fit.
This is a property of the METHOD as configured here (whole-mesh, not part-pair-aware), not of
the CPU detector (validated correct and comparable to the CUDA kernel) or the CG solver
(validated deterministic and correct on the toy example). A part-pair-scoped version of this
same pipeline -- filtering detected pairs down to genuinely anatomically-non-adjacent parts,
the same way this project's own `penetration_loss.py`/`bvh_penetration_loss.py` already do --
was the natural next design, tried immediately below rather than left as a hypothesis.

## Final follow-up: the part-aware filter -- tried, and it made the outcome worse

`fitter_3d/cpu_self_intersection.py`'s detector extended with an optional `face_part_id`/
`allowed_part_pairs` filter (same `non_adjacent_pairs` convention as `penetration_loss.py`),
applied AFTER raw detection and BEFORE the pairs reach the CG-based push -- one more
single-variable swap, same 3 specimens, same energy/constraints, everything else unchanged.
Sanity-checked first: 48-69% of each specimen's raw detected pairs are same-part or
adjacent-part touches, excluded by the filter -- a real, substantial fraction, not a no-op.

| specimen | hard: before->after | F-score: before->after | edge distortion |
|---|---|---|---:|
| Acanthostichus | 755->141 (-81.3%) | 0.885->**0.222** (**-0.662**) | **0.318** |
| Acromyrmex_coronatus | 964->642 (-33.4%) | 0.678->0.477 (-0.201) | 0.079 |
| Solenopsis_invicta | 305->70 (-77.0%) | 0.678->0.562 (-0.115) | 0.042 |

**Count improved dramatically on all 3** (unlike the unfiltered version, where Acromyrmex
regressed +40.2%) -- the filter did concentrate the repair onto genuine problem pairs, exactly
as the mechanistic hypothesis predicted. **But mean F-score cost is -0.326, more than double
the unfiltered version's -0.152**, and Acanthostichus effectively collapses (edge distortion
0.318, an order of magnitude worse than any other result in this task, F-score falling from
0.885 to 0.222). Reading: with fewer, deeper, more severe pairs to resolve and the same
200-iteration budget, the optimizer pushes far harder per remaining pair -- concentrating scope
concentrated force, and the volume/area/curvature constraints available in this tool aren't
strong enough to contain it. The hypothesis wasn't wrong about WHERE the distortion was coming
from (whole-mesh unfiltered repair does resolve legitimate contact it shouldn't touch); it was
wrong that removing those pairs would reduce total distortion -- instead it redirected the same
optimization pressure onto fewer targets, making each one's local damage worse.

## Verdict

**Not usable as a clean fix, on outcome grounds, tried two ways, both worse than the
training-time alternatives.** All infrastructure blockers this task uncovered are resolved:
`cholespy`'s segfault (fixed), the CUDA BVH kernel's repeated-invocation fault (bypassed via a
validated CPU detector), determinism (confirmed, perfectly, in both configurations). The
structural argument for why post-hoc repair should sidestep training-time instability was
correct and is directly demonstrated -- there is no seed noise, no gradient roughness, no
training-graph interaction anywhere in either result. But the actual repair, whole-mesh
(-0.152 mean F-score) or part-aware (-0.326 mean F-score, one specimen collapsing outright),
produces the worst surface-fit cost of any arm tried across TASK1-9 either way, and the more
targeted version is worse, not better. This closes post-hoc repair -- not on infrastructure
grounds, and not on a first-attempt technicality, but because the one mechanistic refinement
with real backing was tried and made the core problem worse. No further variant of this
approach is recommended without a fundamentally different constraint/optimizer design (e.g. a
per-iteration displacement cap, not attempted here), which is a new method, not a tenth attempt
at this one.
