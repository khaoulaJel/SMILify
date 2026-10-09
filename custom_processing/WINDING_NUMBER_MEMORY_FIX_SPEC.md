# `close_holes_via_winding_number` memory fix — scoped spec

## Status
**Implemented** (batch-and-flush variant, see below) as of 2026-08-06. `close_holes_via_winding_number`
now welds every `tile_batch_size` (default 300) surface-producing tiles into a persistent running
bmesh immediately, discarding each batch's raw arrays afterward, instead of accumulating every
tile's raw output for the whole loop and welding once at the end. Peak raw-accumulation memory is
now bounded by `tile_batch_size x per-tile size`, not `total_tiles x per-tile size`. The
`tile_memory_budget_fraction` guard was adapted to check ACTUAL memory consumed (via
`_available_memory_bytes()`) after every batch flush, rather than extrapolating from partial
tile-count averages as the earlier stopgap version did - a stronger check now that there's a
real per-flush checkpoint to measure at. `tile_memory_check_every` was removed (no longer
meaningful now that the guard runs on every flush).

## Validation results (2026-08-06)

Re-ran Cephalotes_depressus_CASENT0744569 (the specimen the old stopgap guard correctly
blocked, needing 6,814 tiles / projected 6.05GB under the old one-shot design) through the
fixed code. Result: **the fix works exactly as designed, but it reveals a different, more
fundamental problem for this specimen.**

The guard tripped again - but after only the *first* batch (300/6,814 tiles), not deep into
the run: the running mesh already held 5,028,363 verts and had consumed 5.86GB, exceeding the
5.11GB budget. This is not a batching-granularity issue (a smaller `tile_batch_size` would trip
sooner, at a smaller number, but the underlying total is the same) - it means the tiles at this
specimen's required resolution (`voxel_pitch=0.2893`, forced fine by its own tight
self-approach gap) are producing genuinely large amounts of *distinct, non-overlapping* new
surface, not mostly-duplicate halo geometry that welding would shrink away. Extrapolating
linearly from the first batch (5.03M verts / 300 tiles) across all 6,814 tiles implies a final
mesh on the order of ~114M vertices - a size no amount of accumulation-strategy cleverness
fixes, because it's not an accumulation artifact. It's what "represent this specimen's entire
surface at the resolution its own gap-safety factor demands" actually costs in vertices, and
that cost exceeds this machine's memory regardless of how carefully the algorithm holds it.

**Conclusion**: batch-and-flush is confirmed working correctly - it converted an
architecture-level unbounded-accumulation risk into a real, checkable, per-flush measurement,
and gave a precise, actionable diagnostic instead of a crash. But for Cephalotes specifically,
the ceiling is a genuine resource-vs-required-resolution mismatch, in the same family as
Metapone's, not the accumulation bug this fix targets. Cephalotes and Metapone both remain
open/unresolved on this machine at their currently-computed target resolution; the fix here
did not change that verdict for either, though it would help any *other* specimen whose
failure actually was accumulation-shaped (large tile count, but tiles mostly overlapping/
redundant near-duplicate geometry) rather than intrinsically-large-output-shaped.

---

## Original status note (superseded, kept for history)
Previously: a stopgap safety guard (`tile_memory_budget_fraction` / `tile_memory_check_every`
params, in `close_holes_via_winding_number`) converted the crash this document describes into a
clear `RuntimeError` instead of a silent kernel OOM-kill, without fixing the underlying design.
This document originally scoped the real fix as future work; that work is now done (see Status
above).

## Confirmed problem (not hypothetical)

Running `close_holes_via_winding_number` as `manifold_external`'s fallback on
`Metapone_emersoni_CASENT0745558` (4,583 occupied tiles), the process was killed by the Linux
OOM-killer at tile 3,900/4,583, `anon-rss` 14.3GB, on a 15GB machine. Confirmed via `dmesg`:

```
Out of memory: Killed process 2209 (blender) total-vm:25622324kB, anon-rss:14972200kB
```

No traceback, no warning — the log simply stops mid-tile-loop.

## Root cause

```python
all_v_chunks = []
all_f_chunks = []
for i, (tx, ty, tz) in enumerate(occupied):
    ...
    tv, tf, _, _ = skimage_measure.marching_cubes(...)
    all_v_chunks.append(tv)
    all_f_chunks.append(tf + v_offset)
    v_offset += len(tv)
# only concatenated + welded ONCE, after the entire loop finishes:
raw_v = np.concatenate(all_v_chunks, axis=0)
raw_f = np.concatenate(all_f_chunks, axis=0)
```

Every tile's raw marching-cubes output (before dedup/welding) is held in memory as a separate
numpy array for the *entire* remainder of the loop. Peak memory scales with **total raw
tile output across the whole specimen**, not with any bounded working set. For a specimen
needing thousands of tiles (large surface area / fine voxel_pitch, e.g. from the gap-safety
resolution boost), this has no ceiling.

## Why this wasn't caught earlier

`close_holes_via_winding_number` had never been run, in this investigation, as a fallback for a
specimen whose `manifold_external` attempt needed thousands of tiles. Its own tile budget
(`max_tiles=6000` default, adaptive growth of `tile_voxels` to stay under it) limits tile
*count*, but does nothing to bound *per-tile output size accumulation* — a specimen can sit
right at the tile-count budget and still accumulate unbounded raw geometry if each tile
produces substantial surface.

## Proposed fix: incremental/streaming tile welding

**Key insight**: duplicate vertices from tile overlap only occur in the halo region between
*spatially adjacent* tiles (bounded by `tile_halo_voxels`). A tile far from the current one in
the tile grid cannot have overlapping geometry with it. This means the current design's global
"accumulate everything, weld once at the end" pass is solving a much bigger problem than
necessary — welding only ever needs to happen locally.

### Design sketch

1. Maintain a single running `bmesh` across the tile loop instead of `all_v_chunks`/
   `all_f_chunks` lists.
2. For each tile, after marching cubes produces `(tv, tf)`:
   - Add `tv`/`tf` as new geometry to the running bmesh.
   - Weld only the **newly-added vertices** against vertices already in the bmesh that fall
     within `weld_threshold_factor * voxel_pitch` of them — not a global weld, a spatially
     scoped one (a `cKDTree` rebuilt periodically over "recent" vertices, or bmesh's own
     `bmesh.ops.remove_doubles` scoped to the new tile's vertices plus a halo-sized
     neighborhood, mirrors the existing boundary-seam-reweld pass's own scoping logic).
3. Periodically (e.g. every N tiles, or when a tile is spatially "far enough" from the active
   processing front) consider a region **finalized** — no longer a merge candidate for future
   tiles — and drop it from whatever "recent vertices" structure is used for welding, so that
   structure's own size stays bounded too, not just the raw accumulation.
4. Final result is the running bmesh directly; no second global concatenation/weld pass needed.

### Complexity/tradeoff

Trades some CPU (many small weld operations instead of one large one) for bounded peak memory.
Given the existing tile-seam reweld pass already demonstrates scoping a weld operation to a
vertex subset (`boundary_verts` in the current code) rather than the whole mesh, this is not a
new technique for this codebase - it is extending an already-used, already-validated approach
from "post-hoc cleanup of a fully-materialized mesh" to "the primary accumulation mechanism."

### Alternative (simpler, less complete): batch-and-flush

Instead of true streaming, accumulate tiles in fixed-size batches (e.g. 200-500 tiles), weld
each batch against the running result, then discard that batch's raw arrays before starting the
next. Simpler to implement than full incremental welding, bounds peak memory to
O(batch_size) + O(running_result_size) rather than O(1) + O(running_result_size), but is a much
smaller change to the existing loop structure. Worth prototyping first given lower implementation
risk - may be sufficient in practice even if less elegant.

## Validation plan (do not skip)

Given this whole investigation's throughline is "verify, don't assume":

1. Re-run the exact failing case (Metapone, as `manifold_external`'s fallback) and confirm peak
   memory stays bounded (track via `/proc/<pid>/status` `VmRSS` sampling during the run, or the
   `_available_memory_bytes()` helper already added).
2. Re-run at least 2-3 of the specimens already validated under the OLD (non-streaming) code
   path and confirm **identical or equivalent output** (vertex/face counts,
   `mesh_integrity_report`, and ideally `verify_reconstruction_fidelity` against the old output)
   - the memory fix must not silently change reconstruction quality.
3. Re-run the full `manifold_repair_batch`/gap-safety validation specimens through this changed
   path at least once to confirm the fusion-detection/fallback machinery still works correctly
   against the new welding implementation's output.

## Explicitly out of scope for this fix

- Metapone is not expected to be "fixed" by this alone in the sense of getting a
  high-fidelity result - the memory fix makes it *possible to complete without crashing*, not
  necessarily *fast* or *at full target resolution*. Whether Metapone's `winding_number`
  reconstruction, once it can actually finish, produces a fidelity-check-passing result is a
  separate question to verify once this fix exists, not to assume.
