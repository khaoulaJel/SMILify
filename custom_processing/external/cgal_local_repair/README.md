# Local geometry-preserving topology repair — research phase (2026-08-17)

Standalone CGAL/C++ diagnostic tools and controlled experiments investigating whether
scan-defect boundary loops (holes, duplicate seams, tangled boundaries, component-interleaving
regions) in the ant-specimen meshes used by `custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py`
can be safely repaired locally, as an alternative/complement to global Alpha Wrap reconstruction.

**This is a research/diagnostic tree. Nothing here is wired into the production pipeline.**
No file under `custom_processing/` (outside this directory) was modified as part of this work.

Full experiment output, CSVs, reports, and diagnostic mesh exports are preserved in
`diagnostics/local_repair_experiment_20260817/`.

## Build

Same pattern as the pre-existing `custom_processing/external/cgal_alpha_wrap/Alpha_wrap_3` example:
CMake + `create_single_source_cgal_program` per `.cpp`, pointed at the already-cloned CGAL checkout.

```
cd custom_processing/external/cgal_local_repair
mkdir -p build && cd build
cmake -DCGAL_DIR=../../cgal_alpha_wrap/CGAL ..
make
```

`build/` is git-ignored (matches the repo's existing `build/` rule).

## Status: two primitives frozen and validated. Everything else is a deliberate, evidenced rejection — not a failed repair attempt.

---

## Frozen primitive 1 — A-only duplicate-seam stitching

**What it does**: stitches exact-duplicate-position boundary-vertex pairs left by
`orient_polygon_soup`'s non-manifold-vertex splitting, restricted to pairs that pass an explicit
classifier (`duplicate_seam_characterization.cpp`) into class **A** ("likely legitimate
duplicate/same-surface"), with a validated coherent border-chain match (`chain_match_length >= 3`,
capped at 150 — see "known limitation" below), same connected component, and outside a generous
exclusion radius around the 8 known genuine Strumigenys near-touch coordinates.

**Mechanism** (`explicit_A_only_stitch.cpp`): does **not** use CGAL's global/unrestricted
`stitch_borders(mesh)` (found to touch B/C-classified pairs) and does **not** use the public
`stitch_borders(cycle_representatives, mesh)` restricted overload (found to hang even on a
2-triangle synthetic case — a bug in this CGAL version's cycle bookkeeping, not something to build
on). Instead: `PMP::stitch_boundary_cycles()` (whole-loop-pure candidates only — a loop is only
offered if *every* vertex on its native boundary cycle is in the safe set, not just some) followed
by `PMP::internal::collect_duplicated_stitchable_boundary_edges()` restricted to the remaining safe
candidate halfedges, feeding the result to the confirmed-working explicit-pairs
`PMP::stitch_borders(mesh, hedge_pairs)`.

**Validated results** (both specimens, mesh-copy only):

| | Strumigenys | Mayriella |
|---|---|---|
| stitched | 4 | 55 |
| boundary edges | 2974→2966 | 2332→2222 |
| boundary loops | 63→62 | 188→188 |
| non-manifold vertices | 0→0 | 0→0 |
| components | 1→1 | 15→15 |
| self-intersections | 5529→5489 | 19007→18461 |
| coordinate displacement | 0 | 0 |
| B/C pairs touched | 0 | 0 |
| cross-component merges | 0 | 0 |
| genuine near-touch gaps altered | 0/8 | n/a |

Reports: `diagnostics/local_repair_experiment_20260817/{strumigenys_alberti,mayriella}_A_only_stitch_report.txt`

**Known limitation, not a bug**: the majority of class-A duplicate pairs (79–99% depending on
specimen) are *same-direction* chain matches — two loops whose boundary walks visit matched
positions in the same rotational order. CGAL's stitching precondition
(`stitch_borders.h`, `equal(source(he),target(other_he)) && equal(target(he),source(other_he))`)
requires the *reversed* relationship; same-direction pairs fail it by construction, and correctly
so — stitching them as-is would produce a locally non-orientable result. This is not something this
primitive should paper over: it's the source-verified reason those chains stay unstitched, and
recovering them would need face-reorientation (a separate, larger, unvalidated operation) — see
"Rejected: giant same-direction chains" below.

---

## Frozen primitive 2 — unambiguous >2-way multiplicity stitching

**What it does**: resolves genuine >2-way exact-duplicate-position vertex clusters (left by
`orient_polygon_soup`) into stitchable pairs, but **only** when the cluster decomposes into a
*perfect matching* — every member has exactly one full predecessor+successor identity match with
another member, with no member left ambiguous or unmatched. Ambiguous or non-pairwise-decomposable
clusters (e.g. a genuine 4-valent cyclic border crossing through one point) are rejected outright.

**Mechanism**: `multiplicity_cluster_analysis.cpp` finds and classifies every >2-way
position cluster on the pristine mesh (Strumigenys: 7 clusters, 1 unambiguous; Mayriella: 110
clusters, 11 unambiguous). `multiway_chain_extension.cpp` extends each unambiguous seed pair
outward (both rotational directions) to recover the short (1-hop each direction — these are small,
self-contained touch points, not fragments of the giant chains) matched chain needed to produce
real stitchable edges. Stitched via the same validated two-stage mechanism as primitive 1.

**Validated results**:

| | Strumigenys | Mayriella |
|---|---|---|
| unambiguous clusters used | 1/7 | 11/110 |
| stitched | 4 | 44 |
| boundary edges | 2974→2966 | 2332→2244 |
| boundary loops | 63→61 | 188→166 |
| non-manifold vertices | 0→0 | 0→0 |
| components | 1→1 | 15→15 |
| self-intersections | 5529→5497 | 19007→18842 |
| coordinate displacement | 0 | 0 |
| B/C pairs touched | 0 | 0 |
| cross-component merges | 0 | 0 |

Reports: `diagnostics/local_repair_experiment_20260817/{strumigenys_alberti,mayriella}_multiway_stitch_report.txt`

Both primitives were re-run against a fresh clean build immediately before this freeze; results
were bit-for-bit identical to the originally validated runs (deterministic, reproducible).

---

## Defect taxonomy and decision architecture

```
DEFECT CLASSIFICATION (per boundary loop, on the pristine read+orient+build mesh)
   │
   ├── SIMPLE + near-planar + min_dist=∞
   │      → REPAIR: existing validated local hole filling (triangulate_hole)
   │
   ├── clean 2-way duplicate seam (class A, coherent chain, reversed orientation)
   │      → REPAIR: primitive 1 (A-only stitching)
   │
   ├── unambiguous >2-way multiplicity cluster (perfect one-ring matching)
   │      → REPAIR: primitive 2 (multiplicity stitching)
   │
   ├── giant same-direction duplicate chain
   │      → REJECT/ESCALATE — CGAL orientation precondition correctly refuses it;
   │        recovery needs face-reorientation, a separate unvalidated operation
   │
   ├── tangled/self-intersecting boundary (fold / complex missing-data region)
   │      → REJECT/ESCALATE — mechanism understood via three independent negative
   │        tests (see below), no safe local operation found
   │
   ├── component-interleaving (loop footprint overlaps another component)
   │      → REJECT/ESCALATE — zero geometric contact evidence between components
   │        in every tested case; spatial proximity alone is not connectivity
   │
   └── anything else / insufficient evidence
          → REJECT/ESCALATE
   ↓
VALIDATION: boundary edges/loops, non-manifold V/E, components, degenerate faces,
self-intersections, coordinate displacement, locality, near-touch preservation,
B/C preservation
   ↓
ACCEPT / REJECT / ESCALATE
```

### Why the rejected classes are deliberate safety decisions, not failed repairs

Each rejection below is the result of a specific, evidenced test failing — not an unexplored gap.

**Giant same-direction chains** — full, complete, unambiguous, zero-multiplicity correspondence
across the whole native loop confirmed by direct chain walk; CGAL's own orientation precondition
(source-verified, not inferred) is the reason it can't be stitched as-is. See
`diagnostics/local_repair_experiment_20260817/{strumigenys_alberti,mayriella}_giant_chain_steps.csv`.

**Tangled/fold boundaries** — three independent mechanisms tested and rejected in sequence on
Strumigenys loop30 (and cross-checked on 1-2 additional loops each time):
1. *Sheet-transition reconnection*: sheet identity (normal + spatial clustering) is real and
   detectable, but proposed reconnection chords span 57-61% of the defect's own bounding diagonal —
   inventing long-distance connectivity with no supporting evidence. Rejected.
2. *Localized incidence fusion*: dual-graph face-adjacency crossings between the two sheets exist
   (29 for loop30) but are spread across the *entire* spatial extent of the defect (fusion-vertex
   bbox diagonal ≈ loop bbox diagonal), not concentrated at a small separable neck. Rejected.
3. *Intersection/crease structure*: actual geometric triangle-triangle intersections computed
   (exact CGAL construction) in the local context; the largest connected intersection curve covers
   ~2% (loop30) to ~9% (loop25) of the defect's spatial extent — nowhere near enough to organize
   the tangle. Rejected.
   Conclusion: the two flanking surface sheets are locally valid and non-intersecting; the tangled
   boundary is best interpreted as the boundary of a complex missing-data / scanner-occlusion
   region, not a mesh topology defect.

**Component-interleaving** — tested via the same evidence hierarchy (existing topology > adjacency
> geometric intersection > component identity > normals). Existing topology/adjacency are
trivially absent between different connected components by construction. Geometric intersection
was tested directly and found to be **zero** across all three known Mayriella cases (0/1597 for the
dominant case, patch2/loop4, despite 1597 same-component intersecting pairs nearby). Spatial
co-location alone (the only tier where "interleaving" exists) is explicitly not treated as
connectivity evidence. Same missing-data-boundary interpretation as the tangled class.

See prior conversation turns / diagnostic files under `diagnostics/local_repair_experiment_20260817/`
for the full instrumented evidence behind each of the above (per-tool CSVs and `.txt` reports,
named by specimen and loop id).

## Explicitly not done in this phase

- No production code touched (`custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py`
  and all other pipeline files are untouched by this work).
- No hole filling, component bridging, self-intersection cleanup, smoothing, remeshing, coordinate
  perturbation, or Alpha Wrap was run as part of any accepted primitive.
- No 50-specimen validation/regression run yet — next step, pending explicit go-ahead.
